"""Heuristic baseline agents.

Each baseline operates on an :class:`ClusterMindObservation` and must return a
:class:`ClusterMindAction`. They share a common ``act(obs, env_state=None)``
signature so :func:`run_episode` can drive them uniformly.

PRD §21 prescribes five baselines plus Random; we add Random as the trivial
floor. The baselines are *deliberately* asymmetric in their failure modes so
the storytelling demo (greedy collapse vs. trained recovery) holds.
"""

from __future__ import annotations

import random
from typing import List, Optional

from clustermind.models import (
    ActionType,
    ClusterMindAction,
    ClusterMindObservation,
    IntensityLevel,
    JobPriority,
    JobStatus,
    JobView,
    NodeStatus,
    NodeView,
    PRIORITY_WEIGHT,
    ZoneView,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_score(job: JobView) -> float:
    priority = PRIORITY_WEIGHT[job.priority]
    urgency = 1.0 / (max(0, job.deadline_remaining) + 1)
    return 0.4 * priority + 0.3 * urgency + 0.2 * job.reward_value + 0.1 * min(job.waiting_steps / 10, 1.0)


def _feasible_node_for_job(nodes: List[NodeView], job: JobView, energy_remaining: float):
    if energy_remaining <= 0.05:
        return []
    return [
        n for n in nodes
        if n.free_gpus >= job.gpu_required
        and n.status in (NodeStatus.HEALTHY, NodeStatus.WARNING)
        and n.temperature < 95
        and n.maintenance_timer == 0
    ]


def _hottest_zone(zones: List[ZoneView], nodes: List[NodeView]) -> Optional[ZoneView]:
    if not zones:
        return None
    avg_by_zone = {}
    for zone in zones:
        zone_nodes = [n for n in nodes if n.zone_id == zone.zone_id]
        if not zone_nodes:
            continue
        avg_by_zone[zone.zone_id] = sum(n.temperature for n in zone_nodes) / len(zone_nodes)
    if not avg_by_zone:
        return None
    hot_id = max(avg_by_zone, key=avg_by_zone.get)
    return next((z for z in zones if z.zone_id == hot_id), None)


def _queued_jobs(jobs: List[JobView]) -> List[JobView]:
    return [j for j in jobs if j.status == JobStatus.QUEUED]


def _running_jobs(jobs: List[JobView]) -> List[JobView]:
    return [j for j in jobs if j.status == JobStatus.RUNNING]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaselineAgent:
    name: str = "BaseAgent"

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class RandomAgent(BaselineAgent):
    name = "RandomAgent"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        legal_types = [la.action_type for la in obs.legal_actions] or [ActionType.NO_OP]
        choice = self.rng.choice(legal_types)
        if choice == ActionType.ALLOCATE_JOB and obs.jobs and obs.nodes:
            queued = _queued_jobs(obs.jobs)
            if queued:
                job = self.rng.choice(queued)
                feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
                if feasible:
                    node = self.rng.choice(feasible)
                    return ClusterMindAction(action_type=choice, job_id=job.job_id, node_id=node.node_id)
        if choice == ActionType.DELAY_JOB:
            queued = _queued_jobs(obs.jobs)
            if queued:
                return ClusterMindAction(action_type=choice, job_id=self.rng.choice(queued).job_id)
        if choice == ActionType.THROTTLE_NODE:
            actives = [n for n in obs.nodes if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN)]
            if actives:
                return ClusterMindAction(action_type=choice, node_id=self.rng.choice(actives).node_id)
        if choice == ActionType.INCREASE_COOLING and obs.cooling_zones:
            zone = self.rng.choice(obs.cooling_zones)
            intensity = self.rng.choice(list(IntensityLevel))
            return ClusterMindAction(action_type=choice, zone_id=zone.zone_id, intensity=intensity)
        if choice == ActionType.RUN_MAINTENANCE:
            actives = [
                n for n in obs.nodes
                if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE)
            ]
            if actives:
                return ClusterMindAction(action_type=choice, node_id=self.rng.choice(actives).node_id)
        if choice == ActionType.MIGRATE_JOB:
            running = _running_jobs(obs.jobs)
            if running:
                job = self.rng.choice(running)
                feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
                feasible = [n for n in feasible if n.node_id != job.assigned_node]
                if feasible:
                    target = self.rng.choice(feasible)
                    return ClusterMindAction(
                        action_type=choice,
                        job_id=job.job_id,
                        source_node_id=job.assigned_node,
                        target_node_id=target.node_id,
                    )
        if choice == ActionType.INSPECT_NODE and obs.nodes:
            return ClusterMindAction(action_type=choice, node_id=self.rng.choice(obs.nodes).node_id)
        if choice == ActionType.SHUTDOWN_NODE:
            actives = [n for n in obs.nodes if n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN)]
            if actives:
                return ClusterMindAction(action_type=choice, node_id=self.rng.choice(actives).node_id)
        return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# GreedyThroughputAgent
# ---------------------------------------------------------------------------

class GreedyThroughputAgent(BaselineAgent):
    """Highest job_score → biggest free node. Ignores heat. Storytelling baseline."""

    name = "GreedyThroughputAgent"

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        queued = _queued_jobs(obs.jobs)
        if not queued:
            return ClusterMindAction(action_type=ActionType.NO_OP)
        queued.sort(key=_job_score, reverse=True)
        for job in queued:
            feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
            if not feasible:
                continue
            best = max(feasible, key=lambda n: (n.free_gpus, -n.temperature))
            return ClusterMindAction(
                action_type=ActionType.ALLOCATE_JOB, job_id=job.job_id, node_id=best.node_id
            )
        return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# ConservativeAutoscalerAgent
# ---------------------------------------------------------------------------

class ConservativeAutoscalerAgent(BaselineAgent):
    name = "ConservativeAutoscalerAgent"

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        # Cool any node above 75C.
        for zone in obs.cooling_zones:
            zone_nodes = [n for n in obs.nodes if n.zone_id == zone.zone_id]
            if any(n.temperature > 75 for n in zone_nodes) and zone.intensity != IntensityLevel.HIGH:
                return ClusterMindAction(
                    action_type=ActionType.INCREASE_COOLING,
                    zone_id=zone.zone_id,
                    intensity=IntensityLevel.HIGH,
                )
        # Throttle warning nodes.
        for n in obs.nodes:
            if n.status == NodeStatus.WARNING and not n.throttled:
                return ClusterMindAction(action_type=ActionType.THROTTLE_NODE, node_id=n.node_id)
        # Delay low-priority queued jobs.
        for j in _queued_jobs(obs.jobs):
            if j.priority in (JobPriority.LOW, JobPriority.MEDIUM):
                return ClusterMindAction(action_type=ActionType.DELAY_JOB, job_id=j.job_id)
        # Allocate critical/high jobs to safe nodes.
        for j in sorted(_queued_jobs(obs.jobs), key=_job_score, reverse=True):
            if j.priority not in (JobPriority.HIGH, JobPriority.CRITICAL):
                continue
            feasible = _feasible_node_for_job(obs.nodes, j, obs.energy_remaining)
            safe = [n for n in feasible if n.temperature < 75 and n.utilization < 0.6]
            pool = safe or feasible
            if pool:
                node = max(pool, key=lambda n: (95 - n.temperature, n.free_gpus))
                return ClusterMindAction(
                    action_type=ActionType.ALLOCATE_JOB, job_id=j.job_id, node_id=node.node_id
                )
        return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# ThermalAwareHeuristicAgent
# ---------------------------------------------------------------------------

class ThermalAwareHeuristicAgent(BaselineAgent):
    name = "ThermalAwareHeuristicAgent"

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        hot_zone = _hottest_zone(obs.cooling_zones, obs.nodes)
        if hot_zone is not None:
            zone_nodes = [n for n in obs.nodes if n.zone_id == hot_zone.zone_id]
            # Trigger on the hottest node in the zone, not the average — idle
            # peers shouldn't suppress a real thermal alarm.
            max_temp = max((n.temperature for n in zone_nodes), default=0.0)
            if max_temp > 78 and hot_zone.intensity != IntensityLevel.HIGH:
                return ClusterMindAction(
                    action_type=ActionType.INCREASE_COOLING,
                    zone_id=hot_zone.zone_id,
                    intensity=IntensityLevel.HIGH,
                )
        # Migrate or throttle hot/warning nodes.
        for n in obs.nodes:
            if n.temperature >= 80 and n.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
                running_on_n = [j for j in _running_jobs(obs.jobs) if j.assigned_node == n.node_id]
                if running_on_n:
                    job = max(running_on_n, key=_job_score)
                    feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
                    feasible = [m for m in feasible if m.node_id != n.node_id and m.temperature < 80]
                    if feasible:
                        target = max(feasible, key=lambda m: (m.free_gpus, -m.temperature))
                        return ClusterMindAction(
                            action_type=ActionType.MIGRATE_JOB,
                            job_id=job.job_id,
                            source_node_id=n.node_id,
                            target_node_id=target.node_id,
                        )
                if not n.throttled:
                    return ClusterMindAction(action_type=ActionType.THROTTLE_NODE, node_id=n.node_id)
        # Allocate the best-scoring job.
        queued = _queued_jobs(obs.jobs)
        if queued:
            queued.sort(key=_job_score, reverse=True)
            for job in queued:
                feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
                cool = [n for n in feasible if n.temperature < 80]
                pool = cool or feasible
                if not pool:
                    continue
                node = max(pool, key=lambda n: (95 - n.temperature, n.free_gpus))
                return ClusterMindAction(
                    action_type=ActionType.ALLOCATE_JOB, job_id=job.job_id, node_id=node.node_id
                )
        return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# BackfillAgent
# ---------------------------------------------------------------------------

class BackfillAgent(BaselineAgent):
    """Slurm-style backfill: only run a low-priority job when no high-priority job is starved."""

    name = "BackfillAgent"

    def act(self, obs: ClusterMindObservation) -> ClusterMindAction:
        queued = _queued_jobs(obs.jobs)
        if not queued:
            return ClusterMindAction(action_type=ActionType.NO_OP)
        high = sorted(
            [j for j in queued if j.priority in (JobPriority.HIGH, JobPriority.CRITICAL)],
            key=_job_score, reverse=True,
        )
        for job in high:
            feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
            if feasible:
                node = max(feasible, key=lambda n: (95 - n.temperature, n.free_gpus))
                return ClusterMindAction(
                    action_type=ActionType.ALLOCATE_JOB, job_id=job.job_id, node_id=node.node_id
                )
        # No critical/high feasible — backfill with low/medium only if no high is *waiting*.
        low = sorted(
            [j for j in queued if j.priority in (JobPriority.LOW, JobPriority.MEDIUM)],
            key=_job_score, reverse=True,
        )
        if not high:
            for job in low:
                feasible = _feasible_node_for_job(obs.nodes, job, obs.energy_remaining)
                if feasible:
                    node = max(feasible, key=lambda n: (95 - n.temperature, n.free_gpus))
                    return ClusterMindAction(
                        action_type=ActionType.ALLOCATE_JOB, job_id=job.job_id, node_id=node.node_id
                    )
        # If high jobs are starved, prefer thermal-aware action to relieve pressure.
        hot_zone = _hottest_zone(obs.cooling_zones, obs.nodes)
        if hot_zone is not None and hot_zone.intensity != IntensityLevel.HIGH:
            zone_nodes = [n for n in obs.nodes if n.zone_id == hot_zone.zone_id]
            if zone_nodes and (sum(n.temperature for n in zone_nodes) / len(zone_nodes)) > 80:
                return ClusterMindAction(
                    action_type=ActionType.INCREASE_COOLING,
                    zone_id=hot_zone.zone_id,
                    intensity=IntensityLevel.MEDIUM,
                )
        return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_BASELINES = {
    "RandomAgent": RandomAgent,
    "GreedyThroughputAgent": GreedyThroughputAgent,
    "ConservativeAutoscalerAgent": ConservativeAutoscalerAgent,
    "ThermalAwareHeuristicAgent": ThermalAwareHeuristicAgent,
    "BackfillAgent": BackfillAgent,
}


def make_baseline(name: str, **kwargs) -> BaselineAgent:
    if name not in ALL_BASELINES:
        raise KeyError(f"Unknown baseline {name}. Choose from {list(ALL_BASELINES)}.")
    return ALL_BASELINES[name](**kwargs)
