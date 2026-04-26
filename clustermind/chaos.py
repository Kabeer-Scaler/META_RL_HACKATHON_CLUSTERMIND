"""Adversarial chaos agent.

The chaos agent picks bounded, weakness-driven events to stress the cluster.
PRD §17 enforces budgets so chaos cannot just slot-machine the agent into
defeat. Every choice flows from observed weaknesses — there is no
"if scenario == triple_crisis: force collapse" anywhere in this file.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from clustermind.models import (
    ChaosActionType,
    ChaosEvent,
    ClusterMindState,
    CoolingZone,
    GPUNode,
    Job,
    JobPriority,
    JobStatus,
    JobType,
    NodeStatus,
)


MAX_CHAOS_PER_EPISODE = 3
MIN_GAP_BETWEEN_CHAOS = 3
MIN_CLUSTER_HEALTH_FOR_CHAOS = 0.25


class ChaosAgent:
    def __init__(self, severity_multiplier: float = 1.0, rng: Optional[random.Random] = None):
        self.severity_multiplier = severity_multiplier
        self.rng = rng or random.Random()
        self.events_used = 0
        self.last_step: Optional[int] = None
        self.last_action: Optional[ChaosActionType] = None

    def reset(self, severity_multiplier: Optional[float] = None) -> None:
        if severity_multiplier is not None:
            self.severity_multiplier = severity_multiplier
        self.events_used = 0
        self.last_step = None
        self.last_action = None

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def decide(
        self,
        state: ClusterMindState,
        agent_history: Dict[str, int],
        step: int,
        enabled: bool,
    ) -> ChaosActionType:
        if not enabled:
            return ChaosActionType.NO_CHAOS
        if state.cluster_health < MIN_CLUSTER_HEALTH_FOR_CHAOS:
            return ChaosActionType.NO_CHAOS
        if self.events_used >= MAX_CHAOS_PER_EPISODE:
            return ChaosActionType.NO_CHAOS
        if self.last_step is not None and step - self.last_step < MIN_GAP_BETWEEN_CHAOS:
            return ChaosActionType.NO_CHAOS

        candidates = self._score_candidates(state, agent_history)
        # Forbid same-as-last and zero-impossibility cases.
        candidates = [
            (action, score) for action, score in candidates
            if action != self.last_action and score > 0.0
        ]
        if not candidates:
            return ChaosActionType.NO_CHAOS
        # Soft-stochastic pick weighted by score so traces are reproducible
        # under seed but not deterministic per scenario.
        actions, scores = zip(*candidates)
        chosen = self.rng.choices(actions, weights=scores, k=1)[0]
        return chosen

    def commit(self, action: ChaosActionType, step: int) -> None:
        if action == ChaosActionType.NO_CHAOS:
            return
        self.events_used += 1
        self.last_step = step
        self.last_action = action

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        action: ChaosActionType,
        state: ClusterMindState,
        step: int,
    ) -> ChaosEvent:
        severity = max(0.1, 0.4 + 0.15 * state.curriculum_level) * self.severity_multiplier
        target: Optional[str] = None
        detail = ""

        if action == ChaosActionType.INJECT_DEMAND_SPIKE:
            new_jobs = []
            for k in range(self.rng.randint(2, 4)):
                jid = f"chaos_job_{step}_{k}"
                new_jobs.append(
                    Job(
                        job_id=jid,
                        job_type=self.rng.choice(list(JobType)),
                        priority=self.rng.choice(
                            [JobPriority.HIGH, JobPriority.MEDIUM, JobPriority.CRITICAL]
                        ),
                        gpu_required=self.rng.randint(2, 5),
                        deadline_remaining=self.rng.randint(4, 9),
                        remaining_work=float(self.rng.randint(40, 80)),
                        thermal_load=self.rng.uniform(0.9, 1.4),
                        energy_cost=self.rng.uniform(0.02, 0.05),
                        reward_value=self.rng.uniform(0.5, 1.0),
                        arrived_at_step=step,
                    )
                )
            state.jobs.extend(new_jobs)
            detail = f"Injected {len(new_jobs)} new jobs."

        elif action == ChaosActionType.DROP_COOLING_EFFICIENCY:
            zone = self._weakest_zone(state.cooling_zones)
            if zone is not None:
                drop = 0.20 + 0.10 * severity
                zone.cooling_efficiency = max(0.20, zone.cooling_efficiency - drop)
                target = zone.zone_id
                detail = f"Dropped cooling efficiency in {zone.zone_id} by {drop:.2f}."

        elif action == ChaosActionType.INCREASE_HIDDEN_DEGRADATION:
            chosen = self._pick_silent_node(state.nodes)
            if chosen is not None:
                bump = 0.15 + 0.05 * severity
                chosen.hidden_degradation = min(1.0, chosen.hidden_degradation + bump)
                target = chosen.node_id
                detail = f"Hidden degradation +{bump:.2f} on {chosen.node_id}."

        elif action == ChaosActionType.ADD_VIP_JOB:
            vip = Job(
                job_id=f"vip_chaos_{step}",
                job_type=JobType.INFERENCE,
                priority=JobPriority.CRITICAL,
                gpu_required=self.rng.randint(4, 6),
                deadline_remaining=self.rng.randint(4, 6),
                remaining_work=70.0,
                thermal_load=1.2,
                energy_cost=0.04,
                reward_value=1.2,
                arrived_at_step=step,
            )
            state.jobs.append(vip)
            target = vip.job_id
            detail = "Added critical VIP job with tight deadline."

        elif action == ChaosActionType.REDUCE_ENERGY_BUDGET:
            cut = 0.15 + 0.10 * severity
            state.energy_remaining = max(0.0, state.energy_remaining - cut)
            detail = f"Reduced remaining energy by {cut:.2f}."

        elif action == ChaosActionType.DELAY_MAINTENANCE:
            chosen = self._weakest_node(state.nodes)
            if chosen is not None and chosen.maintenance_timer > 0:
                chosen.maintenance_timer += 1
                target = chosen.node_id
                detail = f"Delayed maintenance on {chosen.node_id} by 1 step."

        elif action == ChaosActionType.TRIGGER_LATENCY_ALERT:
            chosen = self._weakest_zone_node(state)
            if chosen is not None:
                chosen.visible_alerts = list(set(chosen.visible_alerts + ["latency_alert"]))
                chosen.hidden_degradation = min(1.0, chosen.hidden_degradation + 0.06)
                target = chosen.node_id
                detail = f"Latency alert on {chosen.node_id}; degradation +0.06."

        return ChaosEvent(
            step=step,
            action=action,
            target=target,
            severity=severity,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Scoring & helpers
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        state: ClusterMindState,
        history: Dict[str, int],
    ) -> List[Tuple[ChaosActionType, float]]:
        scores: List[Tuple[ChaosActionType, float]] = []

        cooling_spam = history.get("INCREASE_COOLING_high", 0)
        delay_count = history.get("DELAY_JOB", 0)
        no_corrective = history.get("no_corrective", 0)
        zone_overload = history.get("zone_overload", 0)
        ignored_warnings = history.get("ignored_warnings", 0)

        # REDUCE_ENERGY_BUDGET — strong against cooling spam.
        scores.append(
            (
                ChaosActionType.REDUCE_ENERGY_BUDGET,
                0.5 + min(2.0, cooling_spam * 0.5),
            )
        )

        # INCREASE_HIDDEN_DEGRADATION when agent is ignoring warnings.
        scores.append(
            (
                ChaosActionType.INCREASE_HIDDEN_DEGRADATION,
                0.5 + min(2.0, ignored_warnings * 0.7),
            )
        )

        # DROP_COOLING_EFFICIENCY on hot/overloaded zones.
        hot_zone_pressure = sum(
            1 for n in state.nodes if n.temperature > 80.0
        )
        scores.append(
            (
                ChaosActionType.DROP_COOLING_EFFICIENCY,
                0.4 + 0.2 * hot_zone_pressure,
            )
        )

        # ADD_VIP_JOB if agent delays a lot.
        scores.append(
            (
                ChaosActionType.ADD_VIP_JOB,
                0.4 + min(2.0, delay_count * 0.5),
            )
        )

        # TRIGGER_LATENCY_ALERT when one zone is overloaded.
        scores.append(
            (
                ChaosActionType.TRIGGER_LATENCY_ALERT,
                0.4 + min(2.0, zone_overload * 0.6),
            )
        )

        # INJECT_DEMAND_SPIKE — a generic stressor early in episode.
        if state.outage_count == 0:
            scores.append((ChaosActionType.INJECT_DEMAND_SPIKE, 0.6))

        # DELAY_MAINTENANCE only meaningful if anything is in maintenance.
        if any(n.maintenance_timer > 0 for n in state.nodes):
            scores.append((ChaosActionType.DELAY_MAINTENANCE, 0.5 + 0.2 * no_corrective))

        return scores

    def _weakest_zone(self, zones: List[CoolingZone]) -> Optional[CoolingZone]:
        if not zones:
            return None
        return min(zones, key=lambda z: z.cooling_efficiency * (1.0 - 0.3 * z.cooling_stress))

    def _pick_silent_node(self, nodes: List[GPUNode]) -> Optional[GPUNode]:
        active = [
            n for n in nodes
            if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE)
        ]
        if not active:
            return None
        # Prefer nodes that look fine but already have some hidden degradation.
        active.sort(key=lambda n: -n.hidden_degradation)
        return active[0]

    def _weakest_node(self, nodes: List[GPUNode]) -> Optional[GPUNode]:
        active = [n for n in nodes if n.status != NodeStatus.FAILED]
        if not active:
            return None
        return max(active, key=lambda n: n.hidden_degradation + n.temperature / 100.0)

    def _weakest_zone_node(self, state: ClusterMindState) -> Optional[GPUNode]:
        zone = self._weakest_zone(state.cooling_zones)
        if zone is None:
            return None
        zone_nodes = [n for n in state.nodes if n.zone_id == zone.zone_id]
        if not zone_nodes:
            return None
        return max(zone_nodes, key=lambda n: n.utilization)
