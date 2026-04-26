"""Main transition loop.

The simulator is the only place that mutates ``ClusterMindState``. It owns
the canonical step ordering from PRD §8:

    observe -> action validation -> apply action -> scenario events
    -> chaos events -> job progress -> deadline tick -> thermal update
    -> cooling stress -> energy ledger -> hidden degradation -> failures
    -> cascades -> cluster health -> guardrails -> reward -> recorder
    -> next observation

We purposefully keep the simulator function-style (long but flat) — every
phase is a small private method, so it's easy to read top-to-bottom and
spot exactly where each PRD-listed concern lives.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from clustermind import failures, scheduler, thermal
from clustermind.chaos import ChaosAgent
from clustermind.guardrails import GuardrailManager
from clustermind.models import (
    ActionType,
    ChaosActionType,
    ChaosEvent,
    ClusterMindAction,
    ClusterMindObservation,
    ClusterMindState,
    CoolingZone,
    FlightRecord,
    GPUNode,
    GuardrailViolation,
    INTENSITY_MULTIPLIER,
    IntensityLevel,
    Job,
    JobPriority,
    JobStatus,
    JobView,
    LegalActionDescriptor,
    NodeStatus,
    NodeView,
    PRIORITY_WEIGHT,
    RewardBreakdown,
    ScenarioConfig,
    ZoneStatus,
    ZoneView,
)
from clustermind.recorder import FlightRecorder
from clustermind.rewards import compute_reward
from clustermind.scenarios import build_scenario


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 20
BASE_GPU_WORK_RATE = 5.0
THROTTLE_FACTOR = 0.55
MIGRATION_PROGRESS_PENALTY_FRAC = 0.05
MIGRATION_ENERGY_COST = 0.03
MIGRATION_HEAT_COST = 3.0
MIGRATION_COOLING_RELIEF = 4.0
MAINTENANCE_ENERGY_COST = 0.05
INSPECT_ENERGY_COST = 0.005
SHUTDOWN_ENERGY_COST = 0.01
JOB_ENERGY_BASE_FACTOR = 0.06


def _job_thermal_penalty(temp: float) -> float:
    if temp < 80:
        return 1.0
    if temp < 90:
        return 0.75
    return 0.45


def _node_health_factor(node: GPUNode) -> float:
    return {
        NodeStatus.HEALTHY: 1.0,
        NodeStatus.WARNING: 0.85,
        NodeStatus.OVERHEATED: 0.5,
        NodeStatus.MAINTENANCE: 0.0,
        NodeStatus.FAILED: 0.0,
        NodeStatus.SHUTDOWN: 0.0,
    }[node.status]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ClusterSimulator:
    """Stateful simulator. One instance per env instance."""

    def __init__(self):
        self.state: ClusterMindState = ClusterMindState()
        self.config: ScenarioConfig = ScenarioConfig(
            name="demand_spike", curriculum_level=1, description=""
        )
        self.recorder = FlightRecorder()
        self.guardrails = GuardrailManager()
        self.chaos = ChaosAgent()
        self.rng = random.Random()
        self._failure_rng = random.Random()
        self._inspect_rng = random.Random()
        self.arrival_schedule: List[Dict[str, Any]] = []
        self.episode_id: Optional[str] = None
        self.last_action_text: str = ""
        self.cluster_health_history: List[float] = []
        self.history_metrics: Dict[str, int] = {}
        self.latest_chaos_event: Optional[ChaosEvent] = None
        self.last_failures: List[str] = []
        self.last_cascades: List[str] = []
        self.last_event_log: List[str] = []
        self.last_invalid_action: bool = False
        self.last_useful_action: bool = False
        self.last_action_type: ActionType = ActionType.NO_OP
        self.last_intensity: Optional[IntensityLevel] = None
        self.last_inspect_target: Optional[str] = None
        self.deadline_misses_so_far: int = 0
        self._max_steps: int = DEFAULT_MAX_STEPS

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario: str = "demand_spike",
        curriculum_level: int = 1,
        seed: Optional[int] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        episode_id: Optional[str] = None,
    ) -> ClusterMindObservation:
        self.rng = random.Random(seed)
        self._failure_rng = random.Random((seed or 0) + 7919)
        self._inspect_rng = random.Random((seed or 0) + 6113)

        config, nodes, zones, initial_jobs, arrival_schedule = build_scenario(
            scenario, curriculum_level=curriculum_level, seed=seed, max_steps=max_steps
        )
        self.config = config
        self.arrival_schedule = arrival_schedule
        self._max_steps = max_steps

        self.state = ClusterMindState(
            scenario=scenario,
            curriculum_level=curriculum_level,
            seed=seed,
            nodes=nodes,
            cooling_zones=zones,
            jobs=list(initial_jobs),
            energy_remaining=config.energy_budget,
            energy_budget_per_step=config.energy_budget_per_step,
            cluster_health=1.0,
            cascade_count=0,
            outage_count=0,
            episode_id=episode_id,
            step_count=0,
        )

        self.recorder.reset(episode_id=episode_id)
        self.guardrails.reset()
        self.chaos.reset(severity_multiplier=config.chaos_severity_multiplier)

        self.cluster_health_history = [1.0]
        self.history_metrics = {}
        self.last_action_text = "reset"
        self.deadline_misses_so_far = 0
        self.episode_id = episode_id

        return self._build_observation(
            done=False,
            reward=None,
            last_action_result="reset",
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: ClusterMindAction) -> Tuple[ClusterMindObservation, float, bool, Dict[str, Any]]:
        self.state.step_count += 1
        step = self.state.step_count
        state_before = self._snapshot_state_for_reward()

        # Phase 1 — apply primary action.
        invalid, reason, useful, action_message = self._apply_primary_action(action, step)
        self.last_invalid_action = invalid
        self.last_useful_action = useful
        self.last_action_type = action.action_type
        self.last_intensity = action.intensity if action.action_type == ActionType.INCREASE_COOLING else None
        self.last_inspect_target = action.node_id if action.action_type == ActionType.INSPECT_NODE else None

        # Phase 2 — scheduled job arrivals from the scenario.
        self._apply_scheduled_arrivals(step)

        # Phase 3 — chaos.
        chaos_action = self.chaos.decide(
            self.state,
            self._weakness_history(),
            step,
            enabled=self.config.chaos_enabled,
        )
        self.latest_chaos_event = None
        if chaos_action != ChaosActionType.NO_CHAOS:
            event = self.chaos.apply(chaos_action, self.state, step)
            self.state.chaos_events.append(event)
            self.chaos.commit(chaos_action, step)
            self.latest_chaos_event = event

        # Phase 4 — progress running jobs and increment waiting/delays.
        self._tick_jobs(step)

        # Phase 5 — thermal & cooling updates.
        self._tick_thermal()

        # Phase 6 — energy ledger (cooling, residual job energy beyond Phase 1).
        self._tick_energy_residuals()

        # Phase 7 — hidden degradation accrual.
        self._tick_degradation()

        # Phase 8 — failures (probabilistic) and cascade propagation.
        prev_outage = state_before.outage_count
        self._tick_failures(step)

        # Phase 9 — maintenance timer.
        self._tick_maintenance()

        # Phase 10 — refresh visible alerts and node statuses.
        self._refresh_alerts()

        # Phase 11 — cluster health update.
        self._update_cluster_health()

        # Phase 12 — guardrails.
        self._tick_guardrails(step, action_invalid=invalid)

        # Phase 13 — reward.
        breakdown = compute_reward(
            state_before,
            self.state,
            invalid_action=invalid,
            useful_action=useful,
            step=step,
        )

        # Phase 14 — recorder.
        self.last_failures = list(self.last_failures)
        self.last_cascades = list(self.last_cascades)
        chaos_label = self.latest_chaos_event.action.value if self.latest_chaos_event else None
        self.recorder.record(
            FlightRecord(
                record_id=f"{self.episode_id or 'ep'}_step_{step:03d}",
                step=step,
                observation_summary=self._lightweight_summary(),
                primary_action=self._action_summary(action, invalid, reason, action_message),
                chaos_action=chaos_label,
                reward_breakdown=breakdown.model_dump(),
                guardrail_flags=[v.name for v in self.state.guardrail_violations if v.step == step],
                failure_events=list(self.last_failures),
                cascade_events=list(self.last_cascades),
                failure_risk={n.node_id: failures.compute_failure_probability(n) for n in self.state.nodes},
                event_log=list(self.last_event_log),
            )
        )

        # Phase 15 — observation.
        terminated = self._termination_check()
        self.last_action_text = action_message

        info = {
            "reward_breakdown": breakdown.model_dump(),
            "guardrail_flags": [v.name for v in self.state.guardrail_violations if v.step == step],
            "guardrail_penalty": breakdown.guardrail_violation_penalty,
            "invalid_action_reason": reason if invalid else None,
            "chaos_action": chaos_label,
            "chaos_event": self.latest_chaos_event.model_dump() if self.latest_chaos_event else None,
            "failure_events": list(self.last_failures),
            "cascade_events": list(self.last_cascades),
            "flight_record_id": f"{self.episode_id or 'ep'}_step_{step:03d}",
            "metrics_snapshot": self._metrics_snapshot(),
            "useful_action": useful,
            "event_log": list(self.last_event_log),
        }

        obs = self._build_observation(
            done=terminated,
            reward=breakdown.total,
            last_action_result=action_message,
        )
        return obs, breakdown.total, terminated, info

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def _apply_primary_action(
        self, action: ClusterMindAction, step: int
    ) -> Tuple[bool, Optional[str], bool, str]:
        self.last_failures = []
        self.last_cascades = []
        self.last_event_log = []

        if action.action_type == ActionType.NO_OP:
            self.state.no_op_count += 1
            self.last_event_log.append("no_op")
            return False, None, False, "NO_OP"

        if action.action_type == ActionType.ALLOCATE_JOB:
            return self._action_allocate(action, step)
        if action.action_type == ActionType.DELAY_JOB:
            return self._action_delay(action, step)
        if action.action_type == ActionType.THROTTLE_NODE:
            return self._action_throttle(action, step)
        if action.action_type == ActionType.INCREASE_COOLING:
            return self._action_cooling(action, step)
        if action.action_type == ActionType.RUN_MAINTENANCE:
            return self._action_maintenance(action, step)
        if action.action_type == ActionType.MIGRATE_JOB:
            return self._action_migrate(action, step)
        if action.action_type == ActionType.INSPECT_NODE:
            return self._action_inspect(action, step)
        if action.action_type == ActionType.SHUTDOWN_NODE:
            return self._action_shutdown(action, step)

        return self._mark_invalid("unknown_action_type")

    def _mark_invalid(self, reason: str, message: Optional[str] = None) -> Tuple[bool, str, bool, str]:
        self.state.invalid_action_count += 1
        self.last_event_log.append(f"invalid_action:{reason}")
        return True, reason, False, message or f"invalid:{reason}"

    # -------- per-action helpers --------

    def _find_job(self, job_id: Optional[str]) -> Optional[Job]:
        if not job_id:
            return None
        for j in self.state.jobs:
            if j.job_id == job_id:
                return j
        return None

    def _find_node(self, node_id: Optional[str]) -> Optional[GPUNode]:
        if not node_id:
            return None
        for n in self.state.nodes:
            if n.node_id == node_id:
                return n
        return None

    def _find_zone(self, zone_id: Optional[str]) -> Optional[CoolingZone]:
        if not zone_id:
            return None
        for z in self.state.cooling_zones:
            if z.zone_id == zone_id:
                return z
        return None

    def _zone_for_node(self, node: GPUNode) -> Optional[CoolingZone]:
        return self._find_zone(node.zone_id)

    def _action_allocate(self, action: ClusterMindAction, step: int):
        job = self._find_job(action.job_id)
        node = self._find_node(action.node_id or action.target_node_id)
        if job is None:
            return self._mark_invalid("job_not_found", f"ALLOCATE_JOB→missing job {action.job_id}")
        if job.status != JobStatus.QUEUED:
            return self._mark_invalid("job_not_queued", f"ALLOCATE_JOB→{job.job_id} status={job.status}")
        if node is None:
            return self._mark_invalid("node_not_found", f"ALLOCATE_JOB→missing node {action.node_id}")
        if not scheduler.is_node_feasible(node, job, self.state.energy_remaining):
            return self._mark_invalid(
                "node_not_feasible",
                f"ALLOCATE_JOB→{node.node_id} infeasible for {job.job_id}",
            )

        job.assigned_node = node.node_id
        job.status = JobStatus.RUNNING
        node.allocated_gpus = min(node.total_gpus, node.allocated_gpus + job.gpu_required)
        node.assigned_jobs.append(job.job_id)
        node.utilization = min(1.0, node.allocated_gpus / max(1, node.total_gpus))
        cost = job.energy_cost * job.gpu_required * 0.5
        self.state.energy_remaining = max(0.0, self.state.energy_remaining - cost)
        self.last_event_log.append(f"allocate:{job.job_id}->{node.node_id}")
        return False, None, False, f"ALLOCATE_JOB {job.job_id}->{node.node_id}"

    def _action_delay(self, action: ClusterMindAction, step: int):
        job = self._find_job(action.job_id)
        if job is None:
            return self._mark_invalid("job_not_found", f"DELAY_JOB→missing {action.job_id}")
        if job.status not in (JobStatus.QUEUED, JobStatus.DELAYED):
            return self._mark_invalid("job_not_delayable", f"DELAY_JOB→{job.job_id} not delayable")
        job.status = JobStatus.DELAYED
        job.delay_count += 1
        job.deadline_remaining = max(0, job.deadline_remaining - 1)
        self.history_metrics["DELAY_JOB"] = self.history_metrics.get("DELAY_JOB", 0) + 1
        self.last_event_log.append(f"delay:{job.job_id}")
        return False, None, False, f"DELAY_JOB {job.job_id}"

    def _action_throttle(self, action: ClusterMindAction, step: int):
        node = self._find_node(action.node_id)
        if node is None:
            return self._mark_invalid("node_not_found", f"THROTTLE_NODE→missing {action.node_id}")
        if node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
            return self._mark_invalid("node_inactive", f"THROTTLE_NODE→{node.node_id} inactive")
        node.throttled = True
        useful = node.temperature > 82 or node.hidden_degradation > 0.4
        self.last_event_log.append(f"throttle:{node.node_id}")
        return False, None, useful, f"THROTTLE_NODE {node.node_id}"

    def _action_cooling(self, action: ClusterMindAction, step: int):
        zone = self._find_zone(action.zone_id)
        if zone is None:
            return self._mark_invalid("zone_not_found", f"INCREASE_COOLING→missing {action.zone_id}")
        intensity = action.intensity or IntensityLevel.MEDIUM
        cost = thermal.cooling_energy_cost(zone)
        if self.state.energy_remaining < cost * 0.5:
            return self._mark_invalid("not_enough_energy", "INCREASE_COOLING→insufficient energy")
        thermal.apply_cooling_action(zone, intensity)
        # Energy is paid in the next thermal residual phase.
        if intensity == IntensityLevel.HIGH:
            self.history_metrics["INCREASE_COOLING_high"] = self.history_metrics.get("INCREASE_COOLING_high", 0) + 1
        useful = any(n.temperature > 78 for n in self.state.nodes if n.zone_id == zone.zone_id)
        self.last_event_log.append(f"cooling:{zone.zone_id}:{intensity.value}")
        return False, None, useful, f"INCREASE_COOLING {zone.zone_id} ({intensity.value})"

    def _action_maintenance(self, action: ClusterMindAction, step: int):
        node = self._find_node(action.node_id)
        if node is None:
            return self._mark_invalid("node_not_found", f"RUN_MAINTENANCE→missing {action.node_id}")
        if node.status == NodeStatus.FAILED or node.status == NodeStatus.SHUTDOWN:
            return self._mark_invalid("node_inactive", f"RUN_MAINTENANCE→{node.node_id} inactive")
        if node.status == NodeStatus.MAINTENANCE:
            return self._mark_invalid("already_in_maintenance", f"RUN_MAINTENANCE→{node.node_id} already in maintenance")
        # Re-queue any running jobs on this node.
        for job in self.state.jobs:
            if job.assigned_node == node.node_id and job.status == JobStatus.RUNNING:
                job.status = JobStatus.QUEUED
                job.assigned_node = None
                job.progress = max(0.0, job.progress - job.remaining_work * 0.05)
        node.assigned_jobs = []
        node.allocated_gpus = 0
        failures.apply_maintenance_to_node(node)
        self.state.energy_remaining = max(0.0, self.state.energy_remaining - MAINTENANCE_ENERGY_COST)
        useful = node.hidden_degradation > 0.05  # was something to clean up
        self.last_event_log.append(f"maintenance:{node.node_id}")
        return False, None, useful, f"RUN_MAINTENANCE {node.node_id}"

    def _action_migrate(self, action: ClusterMindAction, step: int):
        job = self._find_job(action.job_id)
        source = self._find_node(action.source_node_id)
        target = self._find_node(action.target_node_id)
        if job is None:
            return self._mark_invalid("job_not_found", f"MIGRATE_JOB→missing {action.job_id}")
        if job.status != JobStatus.RUNNING:
            return self._mark_invalid("job_not_running", f"MIGRATE_JOB→{job.job_id} not running")
        if source is None or job.assigned_node != source.node_id:
            return self._mark_invalid("source_mismatch", f"MIGRATE_JOB→source mismatch")
        if target is None:
            return self._mark_invalid("target_not_found", f"MIGRATE_JOB→missing target")
        if target.status not in (NodeStatus.HEALTHY, NodeStatus.WARNING):
            return self._mark_invalid("target_inactive", f"MIGRATE_JOB→target {target.node_id} inactive")
        if target.free_gpus < job.gpu_required:
            return self._mark_invalid("target_capacity", f"MIGRATE_JOB→target {target.node_id} no capacity")

        source.allocated_gpus = max(0, source.allocated_gpus - job.gpu_required)
        if job.job_id in source.assigned_jobs:
            source.assigned_jobs.remove(job.job_id)
        target.allocated_gpus = min(target.total_gpus, target.allocated_gpus + job.gpu_required)
        target.assigned_jobs.append(job.job_id)
        job.assigned_node = target.node_id

        job.progress = max(0.0, job.progress - job.remaining_work * MIGRATION_PROGRESS_PENALTY_FRAC)
        source.temperature = max(30.0, source.temperature - MIGRATION_COOLING_RELIEF)
        target.temperature = min(110.0, target.temperature + MIGRATION_HEAT_COST)
        self.state.energy_remaining = max(0.0, self.state.energy_remaining - MIGRATION_ENERGY_COST)
        useful = source.temperature > 82 or source.hidden_degradation > 0.4 or job.priority in (JobPriority.CRITICAL, JobPriority.HIGH)
        self.last_event_log.append(f"migrate:{job.job_id}:{source.node_id}->{target.node_id}")
        return False, None, useful, f"MIGRATE_JOB {job.job_id} {source.node_id}->{target.node_id}"

    def _action_inspect(self, action: ClusterMindAction, step: int):
        node = self._find_node(action.node_id)
        if node is None:
            return self._mark_invalid("node_not_found", f"INSPECT_NODE→missing {action.node_id}")
        estimate = failures.inspect_node(node, self.state.curriculum_level, self._inspect_rng, step)
        self.state.energy_remaining = max(0.0, self.state.energy_remaining - INSPECT_ENERGY_COST)
        useful = estimate > 0.30
        self.last_event_log.append(f"inspect:{node.node_id}={estimate:.2f}")
        return False, None, useful, f"INSPECT_NODE {node.node_id} (est {estimate:.2f})"

    def _action_shutdown(self, action: ClusterMindAction, step: int):
        node = self._find_node(action.node_id)
        if node is None:
            return self._mark_invalid("node_not_found", f"SHUTDOWN_NODE→missing {action.node_id}")
        if node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
            return self._mark_invalid("node_already_off", f"SHUTDOWN_NODE→{node.node_id} already off")
        failed_ids, paused_ids = failures.shutdown_node(node, self.state.jobs, step)
        self.state.energy_remaining = max(0.0, self.state.energy_remaining - SHUTDOWN_ENERGY_COST)
        useful = node.hidden_degradation > 0.7 or node.temperature > 95
        self.last_event_log.append(f"shutdown:{node.node_id}")
        if failed_ids:
            self.last_event_log.append(f"shutdown_failed_jobs:{','.join(failed_ids)}")
        return False, None, useful, f"SHUTDOWN_NODE {node.node_id}"

    # ------------------------------------------------------------------
    # Tick phases
    # ------------------------------------------------------------------

    def _apply_scheduled_arrivals(self, step: int) -> None:
        for entry in self.arrival_schedule:
            if entry.get("step") == step:
                self.state.jobs.extend(entry.get("jobs", []))
                self.last_event_log.append(f"job_arrivals:{len(entry.get('jobs', []))}")

    def _tick_jobs(self, step: int) -> None:
        for job in self.state.jobs:
            if job.status == JobStatus.QUEUED:
                job.waiting_steps += 1
                continue
            if job.status == JobStatus.RUNNING:
                node = self._find_node(job.assigned_node)
                if node is None or node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
                    # Will be tidied during failure handling.
                    continue
                throttle = THROTTLE_FACTOR if node.throttled else 1.0
                thermal_p = _job_thermal_penalty(node.temperature)
                health_p = _node_health_factor(node)
                progress_gain = job.gpu_required * BASE_GPU_WORK_RATE * throttle * thermal_p * health_p
                job.progress = min(job.remaining_work, job.progress + progress_gain)
                job_energy = job.energy_cost * job.gpu_required * JOB_ENERGY_BASE_FACTOR
                self.state.energy_remaining = max(0.0, self.state.energy_remaining - job_energy)
                if job.progress >= job.remaining_work:
                    job.status = JobStatus.COMPLETED
                    job.completed_at_step = step
                    self.state.completed_jobs.append(job.job_id)
                    if node and job.job_id in node.assigned_jobs:
                        node.assigned_jobs.remove(job.job_id)
                    if node:
                        node.allocated_gpus = max(0, node.allocated_gpus - job.gpu_required)
                        node.utilization = min(1.0, node.allocated_gpus / max(1, node.total_gpus))
            # Deadline tick for everything still active.
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.DELAYED):
                job.deadline_remaining -= 1
                if job.deadline_remaining < 0 and job.status != JobStatus.COMPLETED:
                    job.status = JobStatus.FAILED
                    self.state.failed_jobs.append(job.job_id)
                    self.last_event_log.append(f"deadline_miss:{job.job_id}")
                    self.deadline_misses_so_far += 1
                    # Free up the node.
                    if job.assigned_node:
                        node = self._find_node(job.assigned_node)
                        if node:
                            node.allocated_gpus = max(0, node.allocated_gpus - job.gpu_required)
                            if job.job_id in node.assigned_jobs:
                                node.assigned_jobs.remove(job.job_id)
                            node.utilization = min(1.0, node.allocated_gpus / max(1, node.total_gpus))
                        job.assigned_node = None

    def _tick_thermal(self) -> None:
        by_id = {n.node_id: n for n in self.state.nodes}
        for node in self.state.nodes:
            zone = self._zone_for_node(node)
            if zone is None:
                continue
            thermal.update_node_temperature(node, zone, by_id)
        for zone in self.state.cooling_zones:
            thermal.passive_cooling_recovery(zone)

    def _tick_energy_residuals(self) -> None:
        # Charge cooling for the step.
        for zone in self.state.cooling_zones:
            self.state.energy_remaining = max(
                0.0, self.state.energy_remaining - thermal.cooling_energy_cost(zone)
            )

    def _tick_degradation(self) -> None:
        for node in self.state.nodes:
            latency = "latency_alert" in node.visible_alerts
            failures.update_hidden_degradation(node, latency_alert=latency)

    def _tick_failures(self, step: int) -> None:
        prev_outage = self.state.outage_count
        prior_failure_steps = [n.last_failure_step for n in self.state.nodes if n.last_failure_step is not None]

        # Probabilistic failure draw.
        for node in list(self.state.nodes):
            if node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE):
                continue
            p = failures.compute_failure_probability(node)
            node.hidden_failure_probability = p
            if self._failure_rng.random() < p:
                failed_ids, paused_ids = failures.trigger_failure(node, self.state.jobs, step)
                self.state.outage_count += 1
                self.last_failures.append(node.node_id)
                self.last_event_log.append(f"node_failed:{node.node_id}")
                if failed_ids:
                    self.last_event_log.append(f"failed_jobs:{','.join(failed_ids)}")
                # Cascade detection: this failure happened within 3 steps of an
                # earlier failure → count as cascade. Keeps "linked failures"
                # (PRD §14.10) distinct from independent stochastic outages.
                if any(0 <= step - prior_step <= 3 for prior_step in prior_failure_steps):
                    self.state.cascade_count += 1
                    self.last_cascades.append(f"linked:{node.node_id}@step{step}")

        # Neighbour load shock from any newly-failed node.
        if self.state.outage_count > prev_outage:
            by_id = {n.node_id: n for n in self.state.nodes}
            for nid in list(self.last_failures):
                src = by_id.get(nid)
                if src is None:
                    continue
                affected = failures.propagate_cascade(src, by_id)
                if affected:
                    self.last_cascades.append(f"shock:{nid}->{','.join(affected)}")

    def _tick_maintenance(self) -> None:
        for node in self.state.nodes:
            failures.step_maintenance_timer(node)

    def _refresh_alerts(self) -> None:
        for node in self.state.nodes:
            thermal.refresh_visible_alerts(node)

    def _update_cluster_health(self) -> None:
        if not self.state.nodes:
            self.state.cluster_health = 1.0
            self.cluster_health_history.append(1.0)
            return
        active = sum(1 for n in self.state.nodes if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN))
        warning = sum(1 for n in self.state.nodes if n.status == NodeStatus.WARNING)
        overheated = sum(1 for n in self.state.nodes if n.status == NodeStatus.OVERHEATED)
        total = len(self.state.nodes)
        score = (active - 0.3 * warning - 0.5 * overheated) / total
        self.state.cluster_health = max(0.0, min(1.0, score))
        self.cluster_health_history.append(self.state.cluster_health)

    def _tick_guardrails(self, step: int, action_invalid: bool) -> None:
        avg_temp = thermal.average_temperature(self.state.nodes)
        any_alert = any(n.visible_alerts for n in self.state.nodes)
        completed_value = sum(j.reward_value for j in self.state.jobs if j.status == JobStatus.COMPLETED)
        available_value = sum(j.reward_value for j in self.state.jobs)
        self.guardrails.context.push(
            action_type=self.last_action_type,
            intensity=self.last_intensity,
            avg_temp=avg_temp,
            node_id_inspected=self.last_inspect_target,
            any_alert=any_alert,
            invalid=action_invalid,
            completed_value=completed_value,
            available_value=max(1e-6, available_value),
        )
        violations = self.guardrails.evaluate(self.state, step)
        for v in violations:
            self.state.guardrail_violations.append(v)

    def _termination_check(self) -> bool:
        if self.state.step_count >= self._max_steps:
            return True
        if self.state.energy_remaining <= 0.0 and not any(
            j.status in (JobStatus.RUNNING, JobStatus.QUEUED) for j in self.state.jobs
        ):
            return True
        active_nodes = sum(1 for n in self.state.nodes if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN))
        if active_nodes == 0:
            self.last_event_log.append("all_nodes_offline")
            return True
        return False

    # ------------------------------------------------------------------
    # Snapshots / observation builders
    # ------------------------------------------------------------------

    def _snapshot_state_for_reward(self) -> ClusterMindState:
        # We need the *previous* outage/cascade counts and previous job statuses
        # so reward components can diff. A shallow Pydantic copy is sufficient.
        return ClusterMindState(
            scenario=self.state.scenario,
            curriculum_level=self.state.curriculum_level,
            seed=self.state.seed,
            nodes=[n.model_copy() for n in self.state.nodes],
            cooling_zones=[z.model_copy() for z in self.state.cooling_zones],
            jobs=[j.model_copy() for j in self.state.jobs],
            energy_remaining=self.state.energy_remaining,
            energy_budget_per_step=self.state.energy_budget_per_step,
            cluster_health=self.state.cluster_health,
            cascade_count=self.state.cascade_count,
            outage_count=self.state.outage_count,
            completed_jobs=list(self.state.completed_jobs),
            failed_jobs=list(self.state.failed_jobs),
            invalid_action_count=self.state.invalid_action_count,
            no_op_count=self.state.no_op_count,
            episode_id=self.state.episode_id,
            step_count=self.state.step_count - 1,
        )

    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        last_action_result: Optional[str],
    ) -> ClusterMindObservation:
        nodes_view = [self._node_view(n) for n in self.state.nodes]
        zones_view = [self._zone_view(z) for z in self.state.cooling_zones]
        jobs_view = [self._job_view(j) for j in self.state.jobs]
        legal = self._compute_legal_actions()
        alerts = []
        for node in self.state.nodes:
            for a in node.visible_alerts:
                alerts.append(f"{node.node_id}:{a}")
        for zone in self.state.cooling_zones:
            if zone.cooling_efficiency < 0.7:
                alerts.append(f"{zone.zone_id}:cooling_degraded")
        guardrail_warnings = [
            f"{v.name}:{v.severity.value}"
            for v in self.state.guardrail_violations
            if v.step == self.state.step_count
        ]
        avg_temp = thermal.average_temperature(self.state.nodes)
        queue = sum(1 for j in self.state.jobs if j.status == JobStatus.QUEUED)
        running = sum(1 for j in self.state.jobs if j.status == JobStatus.RUNNING)
        queue_pressure = queue / max(1, queue + running + 1)
        return ClusterMindObservation(
            done=done,
            reward=reward,
            step=self.state.step_count,
            max_steps=self._max_steps,
            scenario=self.state.scenario,
            curriculum_level=self.state.curriculum_level,
            cluster_health=self.state.cluster_health,
            energy_remaining=self.state.energy_remaining,
            queue_pressure=queue_pressure,
            average_temperature=avg_temp,
            active_outages=sum(1 for n in self.state.nodes if n.status == NodeStatus.FAILED),
            cascade_count=self.state.cascade_count,
            nodes=nodes_view,
            cooling_zones=zones_view,
            jobs=jobs_view,
            alerts=alerts,
            legal_actions=legal,
            last_action_result=last_action_result,
            guardrail_warnings=guardrail_warnings,
        )

    def _node_view(self, node: GPUNode) -> NodeView:
        # Inspection estimate is only revealed for the most recent inspection (per PRD §11).
        estimate = node.inspection_estimate if node.inspection_step is not None else None
        return NodeView(
            node_id=node.node_id,
            zone_id=node.zone_id,
            total_gpus=node.total_gpus,
            free_gpus=node.free_gpus,
            allocated_gpus=node.allocated_gpus,
            temperature=round(node.temperature, 2),
            utilization=round(node.utilization, 3),
            status=node.status,
            assigned_jobs=list(node.assigned_jobs),
            visible_alerts=list(node.visible_alerts),
            throttled=node.throttled,
            maintenance_timer=node.maintenance_timer,
            inspection_estimate=estimate,
        )

    def _zone_view(self, z: CoolingZone) -> ZoneView:
        return ZoneView(
            zone_id=z.zone_id,
            cooling_power=round(z.cooling_power, 3),
            cooling_efficiency=round(z.cooling_efficiency, 3),
            cooling_stress=round(z.cooling_stress, 3),
            intensity=z.intensity,
            status=z.status,
            node_ids=list(z.node_ids),
        )

    def _job_view(self, j: Job) -> JobView:
        progress_pct = 0.0
        if j.remaining_work > 0:
            progress_pct = round(min(1.0, j.progress / j.remaining_work), 3)
        return JobView(
            job_id=j.job_id,
            job_type=j.job_type,
            priority=j.priority,
            gpu_required=j.gpu_required,
            deadline_remaining=max(-1, j.deadline_remaining),
            waiting_steps=j.waiting_steps,
            progress_pct=progress_pct,
            status=j.status,
            assigned_node=j.assigned_node,
            reward_value=round(j.reward_value, 3),
        )

    def _compute_legal_actions(self) -> List[LegalActionDescriptor]:
        legal: List[LegalActionDescriptor] = []
        legal.append(LegalActionDescriptor(action_type=ActionType.NO_OP, description="do nothing"))
        # ALLOCATE_JOB if any queued job can fit somewhere.
        for job in self.state.jobs:
            if job.status != JobStatus.QUEUED:
                continue
            if scheduler.feasible_nodes(self.state.nodes, job, self.state.energy_remaining):
                legal.append(LegalActionDescriptor(
                    action_type=ActionType.ALLOCATE_JOB,
                    description=f"allocate {job.job_id} (priority={job.priority.value})",
                ))
                break
        # DELAY_JOB if any queued/running job to delay.
        if any(j.status in (JobStatus.QUEUED, JobStatus.DELAYED) for j in self.state.jobs):
            legal.append(LegalActionDescriptor(action_type=ActionType.DELAY_JOB, description="delay a queued job"))
        # THROTTLE_NODE if any active node not throttled.
        if any(n.is_active and not n.throttled for n in self.state.nodes):
            legal.append(LegalActionDescriptor(action_type=ActionType.THROTTLE_NODE, description="throttle a node"))
        # INCREASE_COOLING always available if energy allows.
        if any(thermal.cooling_energy_cost(z) * 0.5 < self.state.energy_remaining for z in self.state.cooling_zones):
            legal.append(LegalActionDescriptor(action_type=ActionType.INCREASE_COOLING, description="raise zone cooling"))
        # RUN_MAINTENANCE if any node not in maintenance.
        if any(n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE) for n in self.state.nodes):
            legal.append(LegalActionDescriptor(action_type=ActionType.RUN_MAINTENANCE, description="run maintenance on a node"))
        # MIGRATE_JOB if any running job and a feasible target exists.
        running = [j for j in self.state.jobs if j.status == JobStatus.RUNNING]
        if running:
            legal.append(LegalActionDescriptor(action_type=ActionType.MIGRATE_JOB, description="migrate a running job"))
        # INSPECT_NODE always legal if node exists.
        if self.state.nodes:
            legal.append(LegalActionDescriptor(action_type=ActionType.INSPECT_NODE, description="inspect hidden degradation"))
        # SHUTDOWN_NODE always legal if active node.
        if any(n.is_active for n in self.state.nodes):
            legal.append(LegalActionDescriptor(action_type=ActionType.SHUTDOWN_NODE, description="shutdown a node"))
        return legal

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _weakness_history(self) -> Dict[str, int]:
        history = dict(self.history_metrics)
        # Crude derived signals for chaos scoring.
        history["ignored_warnings"] = sum(
            1 for n in self.state.nodes if "temp_warning" in n.visible_alerts
        )
        history["zone_overload"] = sum(
            1 for z in self.state.cooling_zones if z.cooling_stress > 0.4
        )
        history["no_corrective"] = self.state.no_op_count
        return history

    def _action_summary(self, action: ClusterMindAction, invalid: bool, reason: Optional[str], message: str) -> Dict[str, Any]:
        out = {
            "action_type": action.action_type.value,
            "job_id": action.job_id,
            "node_id": action.node_id,
            "source_node_id": action.source_node_id,
            "target_node_id": action.target_node_id,
            "zone_id": action.zone_id,
            "intensity": action.intensity.value if action.intensity else None,
            "invalid": invalid,
            "reason": reason,
            "message": message,
        }
        return out

    def _lightweight_summary(self) -> Dict[str, Any]:
        return {
            "step": self.state.step_count,
            "scenario": self.state.scenario,
            "cluster_health": round(self.state.cluster_health, 3),
            "energy_remaining": round(self.state.energy_remaining, 3),
            "active_outages": sum(1 for n in self.state.nodes if n.status == NodeStatus.FAILED),
            "cascade_count": self.state.cascade_count,
            "average_temperature": round(thermal.average_temperature(self.state.nodes), 2),
            "completed_jobs": len(self.state.completed_jobs),
            "failed_jobs": len(self.state.failed_jobs),
        }

    def _metrics_snapshot(self) -> Dict[str, Any]:
        completed_critical = sum(
            1 for j in self.state.jobs
            if j.status == JobStatus.COMPLETED and j.priority in (JobPriority.CRITICAL, JobPriority.HIGH)
        )
        total_critical = sum(
            1 for j in self.state.jobs if j.priority in (JobPriority.CRITICAL, JobPriority.HIGH)
        )
        return {
            "completed_jobs": len(self.state.completed_jobs),
            "failed_jobs": len(self.state.failed_jobs),
            "completed_critical": completed_critical,
            "total_critical": max(1, total_critical),
            "outage_count": self.state.outage_count,
            "cascade_count": self.state.cascade_count,
            "cluster_health": self.state.cluster_health,
            "average_temperature": thermal.average_temperature(self.state.nodes),
            "energy_remaining": self.state.energy_remaining,
            "invalid_actions": self.state.invalid_action_count,
            "deadline_misses": self.deadline_misses_so_far,
            "guardrail_violations": len(self.state.guardrail_violations),
        }
