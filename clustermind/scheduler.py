"""Scheduling helpers — pure functions that score jobs and nodes.

These power both the heuristic baselines and the legality oracle inside
the simulator. None of them mutate state. PRD §14.1–14.3 spell out the
formulas; this module is a faithful implementation, kept dependency-free
so it can be reused by tests and the LLM action-validation path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from clustermind.models import (
    CoolingZone,
    GPUNode,
    Job,
    JobPriority,
    JobStatus,
    NodeStatus,
    PRIORITY_WEIGHT,
)


# ---------------------------------------------------------------------------
# Job scoring
# ---------------------------------------------------------------------------

def job_score(job: Job) -> float:
    """Composite priority/urgency score per PRD §14.1.

    score = 0.40 * priority + 0.30 * urgency + 0.20 * reward + 0.10 * waiting
    """

    priority = PRIORITY_WEIGHT[job.priority]
    urgency = 1.0 / float(max(0, job.deadline_remaining) + 1)
    reward = max(0.0, min(1.0, job.reward_value))
    waiting = min(job.waiting_steps / 10.0, 1.0)
    return 0.40 * priority + 0.30 * urgency + 0.20 * reward + 0.10 * waiting


def rank_jobs(jobs: List[Job]) -> List[Job]:
    queued = [j for j in jobs if j.status == JobStatus.QUEUED]
    return sorted(queued, key=job_score, reverse=True)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def is_node_feasible(
    node: GPUNode,
    job: Job,
    energy_remaining: float,
) -> bool:
    """PRD §14.2 feasibility filter — only impossible actions are rejected."""

    if node.free_gpus < job.gpu_required:
        return False
    if node.status not in (NodeStatus.HEALTHY, NodeStatus.WARNING):
        return False
    if node.temperature >= 95.0:
        return False
    if node.maintenance_timer > 0:
        return False
    if energy_remaining <= job.energy_cost:
        return False
    return True


def feasible_nodes(
    nodes: List[GPUNode],
    job: Job,
    energy_remaining: float,
) -> List[GPUNode]:
    return [n for n in nodes if is_node_feasible(n, job, energy_remaining)]


# ---------------------------------------------------------------------------
# Node placement scoring
# ---------------------------------------------------------------------------

def _zone_lookup(zones: List[CoolingZone]):
    return {z.zone_id: z for z in zones}


def _zone_utilization(zone_id: str, nodes: List[GPUNode]) -> float:
    z_nodes = [n for n in nodes if n.zone_id == zone_id]
    if not z_nodes:
        return 0.0
    return sum(n.utilization for n in z_nodes) / len(z_nodes)


def node_score(
    node: GPUNode,
    job: Job,
    zones: List[CoolingZone],
    nodes: List[GPUNode],
) -> float:
    """PRD §14.3 placement score, kept clamped to [0, 1] component-wise."""

    thermal_headroom = max(0.0, min(1.0, (95.0 - node.temperature) / 65.0))
    fit_quality = 1.0 - abs(node.free_gpus - job.gpu_required) / max(1, node.total_gpus)
    fit_quality = max(0.0, min(1.0, fit_quality))
    estimated_failure_risk = min(1.0, max(0.0, node.hidden_degradation * 0.6 + max(0.0, node.temperature - 80) / 30 * 0.4))
    risk_safety = 1.0 - estimated_failure_risk
    zone = _zone_lookup(zones).get(node.zone_id)
    zone_util = _zone_utilization(node.zone_id, nodes)
    zone_balance = max(0.0, 1.0 - zone_util)
    if zone is not None:
        energy_efficiency = max(0.0, 1.0 - (zone.energy_cost_multiplier - 0.5) / 1.5)
    else:
        energy_efficiency = 0.5
    return (
        0.35 * thermal_headroom
        + 0.25 * fit_quality
        + 0.15 * risk_safety
        + 0.15 * zone_balance
        + 0.10 * energy_efficiency
    )


def best_node(
    nodes: List[GPUNode],
    job: Job,
    zones: List[CoolingZone],
    energy_remaining: float,
) -> Optional[GPUNode]:
    candidates = feasible_nodes(nodes, job, energy_remaining)
    if not candidates:
        return None
    return max(candidates, key=lambda n: node_score(n, job, zones, nodes))


# ---------------------------------------------------------------------------
# Strategy presets used by baselines
# ---------------------------------------------------------------------------

def greedy_choice(nodes, job, zones, energy_remaining):
    """Pick feasible node with the most free GPUs (ignores heat)."""

    candidates = feasible_nodes(nodes, job, energy_remaining)
    if not candidates:
        return None
    return max(candidates, key=lambda n: (n.free_gpus, -n.temperature))


def thermal_aware_choice(nodes, job, zones, energy_remaining, hot_thresh: float = 85.0):
    candidates = feasible_nodes(nodes, job, energy_remaining)
    if not candidates:
        return None
    cool = [n for n in candidates if n.temperature < hot_thresh]
    pool = cool or candidates
    return max(pool, key=lambda n: node_score(n, job, zones, nodes))


def conservative_choice(nodes, job, zones, energy_remaining):
    candidates = feasible_nodes(nodes, job, energy_remaining)
    safe = [n for n in candidates if n.temperature < 75.0 and n.utilization < 0.6]
    pool = safe or candidates
    if not pool:
        return None
    return max(pool, key=lambda n: node_score(n, job, zones, nodes))


def backfill_choice(
    nodes,
    job,
    other_pending_jobs: List[Job],
    zones,
    energy_remaining,
):
    """Slurm-style backfill — only run a low-priority job if doing so does not
    starve a higher-priority job that is waiting.
    """

    higher = [j for j in other_pending_jobs if PRIORITY_WEIGHT[j.priority] > PRIORITY_WEIGHT[job.priority]]
    if higher and PRIORITY_WEIGHT[job.priority] < 0.75:
        # Don't backfill if a critical/high job is starved and would fit somewhere.
        for hjob in higher:
            if feasible_nodes(nodes, hjob, energy_remaining):
                return None
    return thermal_aware_choice(nodes, job, zones, energy_remaining)
