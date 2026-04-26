"""Hidden degradation, failure probability, and cascade propagation.

Per PRD §14.8–14.10 the agent cannot directly observe a node's true
``hidden_degradation`` or ``hidden_failure_probability``. Inspection returns
a noisy estimate, maintenance scrubs degradation back down, and node failure
shocks neighbours so cascades emerge from dynamics rather than a script.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from clustermind.models import (
    GPUNode,
    Job,
    JobStatus,
    NodeStatus,
)


INSPECTION_NOISE = {
    1: 0.02,
    2: 0.05,
    3: 0.10,
    4: 0.15,
    5: 0.20,
}


def update_hidden_degradation(node: GPUNode, latency_alert: bool = False) -> None:
    """Accrue degradation per the PRD heuristic. Maintenance applies separately."""

    if node.status == NodeStatus.FAILED:
        # Already failed nodes accumulate nothing.
        return
    if node.status == NodeStatus.SHUTDOWN:
        # Shutdown nodes recover slowly.
        node.hidden_degradation = max(0.0, node.hidden_degradation - 0.01)
        return
    if node.status == NodeStatus.MAINTENANCE:
        # Maintenance handled separately when applied.
        return

    if node.temperature > 80:
        node.hidden_degradation += 0.03
    if node.temperature > 90:
        node.hidden_degradation += 0.07
    if node.utilization > 0.90:
        node.hidden_degradation += 0.02
    if latency_alert:
        node.hidden_degradation += 0.02

    node.hidden_degradation = max(0.0, min(1.0, node.hidden_degradation))


def apply_maintenance_to_node(node: GPUNode) -> None:
    node.hidden_degradation = max(0.0, node.hidden_degradation - 0.25)
    node.temperature = max(30.0, node.temperature - 8.0)
    node.status = NodeStatus.MAINTENANCE
    node.maintenance_timer = 2
    node.throttled = False


def step_maintenance_timer(node: GPUNode) -> None:
    if node.status == NodeStatus.MAINTENANCE:
        node.maintenance_timer = max(0, node.maintenance_timer - 1)
        if node.maintenance_timer <= 0:
            node.status = NodeStatus.HEALTHY
            node.allocated_gpus = 0
            node.assigned_jobs = []


def inspect_node(node: GPUNode, level: int, rng: random.Random, step: int) -> float:
    noise = INSPECTION_NOISE.get(level, 0.10)
    estimate = node.hidden_degradation + rng.gauss(0.0, noise)
    estimate = max(0.0, min(1.0, estimate))
    node.inspection_estimate = estimate
    node.inspection_step = step
    return estimate


# ---------------------------------------------------------------------------
# Failure probability
# ---------------------------------------------------------------------------

def compute_failure_probability(node: GPUNode) -> float:
    """Per-step failure probability.

    PRD §14.9 specifies a 0.05 floor, but with 10 nodes × 20 steps that makes
    episodes stochastically unsurvivable regardless of policy. We keep the
    *shape* (degradation > thermal > utilisation) but bring the floor down to
    0.005 so that failures actually correlate with bad management.
    """

    if node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE):
        return 0.0
    base = 0.005
    deg = 0.35 * node.hidden_degradation
    therm = 0.30 * max(0.0, node.temperature - 80) / 30.0
    util = 0.15 * max(0.0, node.utilization - 0.90) / 0.10
    p = base + deg + therm + util
    return max(0.0, min(0.95, p))


# ---------------------------------------------------------------------------
# Failure resolution
# ---------------------------------------------------------------------------

def trigger_failure(
    node: GPUNode,
    jobs: List[Job],
    step: int,
) -> Tuple[List[str], List[str]]:
    """Mark node failed; return (failed_job_ids, paused_job_ids)."""

    failed_jobs: List[str] = []
    paused_jobs: List[str] = []
    node.status = NodeStatus.FAILED
    node.last_failure_step = step
    node.utilization = 0.0
    node.allocated_gpus = 0
    node.throttled = False

    for job in jobs:
        if job.assigned_node != node.node_id or job.status not in (JobStatus.RUNNING, JobStatus.QUEUED):
            continue
        if job.priority.value in ("critical", "high"):
            # Pause and re-queue critical/high jobs so they can be migrated/re-allocated.
            job.status = JobStatus.QUEUED
            job.assigned_node = None
            paused_jobs.append(job.job_id)
        else:
            job.status = JobStatus.FAILED
            failed_jobs.append(job.job_id)
    node.assigned_jobs = []
    return failed_jobs, paused_jobs


def propagate_cascade(
    failed_node: GPUNode,
    by_id: Dict[str, GPUNode],
) -> List[str]:
    """Apply load shock to neighbours; returns affected node IDs."""

    affected: List[str] = []
    for nid in failed_node.neighbors:
        n = by_id.get(nid)
        if n is None or n.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
            continue
        n.utilization = max(0.0, min(1.0, n.utilization + 0.10))
        n.temperature = min(110.0, n.temperature + 5.0)
        n.hidden_degradation = max(0.0, min(1.0, n.hidden_degradation + 0.04))
        affected.append(nid)
    return affected


def shutdown_node(node: GPUNode, jobs: List[Job], step: int) -> Tuple[List[str], List[str]]:
    failed_jobs: List[str] = []
    paused_jobs: List[str] = []
    if node.status == NodeStatus.SHUTDOWN:
        return failed_jobs, paused_jobs
    node.status = NodeStatus.SHUTDOWN
    node.utilization = 0.0
    node.allocated_gpus = 0
    node.temperature = max(30.0, node.temperature - 12.0)
    node.throttled = False
    for job in jobs:
        if job.assigned_node != node.node_id:
            continue
        if job.priority.value in ("critical", "high"):
            job.status = JobStatus.QUEUED
            job.assigned_node = None
            paused_jobs.append(job.job_id)
        else:
            job.status = JobStatus.FAILED
            failed_jobs.append(job.job_id)
    node.assigned_jobs = []
    return failed_jobs, paused_jobs
