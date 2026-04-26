"""Reward calculation per PRD §19.

The reward is a *weighted decomposition*: every component is computed each
step and bundled into a :class:`RewardBreakdown`. The simulator clips the
total to [-1, 1] for stability when GRPO/PPO sees per-step returns.

Crucially, the formulas reference dynamics signals (cluster health, deadline
slack, energy use, guardrail penalties) — never agent identity, scenario
name, or fixed step indices. PRD §32 forbids those shortcuts.
"""

from __future__ import annotations

from typing import Iterable, List

from clustermind.models import (
    ClusterMindState,
    GPUNode,
    GuardrailViolation,
    Job,
    JobPriority,
    JobStatus,
    NodeStatus,
    RewardBreakdown,
)


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------

def _critical_completion(jobs: Iterable[Job]) -> float:
    completed = sum(
        j.reward_value for j in jobs
        if j.status == JobStatus.COMPLETED and j.priority in (JobPriority.CRITICAL, JobPriority.HIGH)
    )
    return min(1.0, completed)


def _normal_completion(jobs: Iterable[Job]) -> float:
    completed = sum(
        j.reward_value for j in jobs
        if j.status == JobStatus.COMPLETED and j.priority in (JobPriority.MEDIUM, JobPriority.LOW)
    )
    return min(1.0, completed * 0.6)


def _deadline_score(jobs: Iterable[Job]) -> float:
    relevant = [j for j in jobs if j.status in (JobStatus.RUNNING, JobStatus.QUEUED)]
    if not relevant:
        return 0.0
    healthy = sum(1 for j in relevant if j.deadline_remaining > 2)
    return healthy / len(relevant)


def _cluster_health(state: ClusterMindState) -> float:
    return max(0.0, min(1.0, state.cluster_health))


def _thermal_safety(nodes: List[GPUNode]) -> float:
    if not nodes:
        return 1.0
    over = sum(1 for n in nodes if n.temperature >= 90)
    warn = sum(1 for n in nodes if 80 <= n.temperature < 90)
    cool = sum(1 for n in nodes if n.temperature < 80)
    return max(0.0, (cool * 1.0 + warn * 0.5) / max(1, len(nodes)))


def _recovery(nodes: List[GPUNode], prev_outage: int, current_outage: int) -> float:
    if prev_outage > current_outage:
        return min(1.0, (prev_outage - current_outage) * 0.5)
    return 0.0


def _energy_efficiency(state: ClusterMindState) -> float:
    return max(0.0, min(1.0, state.energy_remaining))


def _useful_inspect_or_maint(action_was_useful: bool) -> float:
    return 1.0 if action_was_useful else 0.0


# ---------------------------------------------------------------------------
# Per-step penalties
# ---------------------------------------------------------------------------

def _outage_penalty(prev_outage: int, current_outage: int) -> float:
    fresh = max(0, current_outage - prev_outage)
    return min(1.0, fresh * 0.5)


def _cascade_penalty(prev_cascade: int, current_cascade: int) -> float:
    fresh = max(0, current_cascade - prev_cascade)
    return min(1.0, fresh * 0.6)


def _missed_critical_deadline(jobs: Iterable[Job]) -> float:
    missed = sum(
        1 for j in jobs
        if j.status == JobStatus.FAILED and j.priority in (JobPriority.CRITICAL, JobPriority.HIGH)
    )
    return min(1.0, missed * 0.4)


def _guardrail_penalty(violations: List[GuardrailViolation], step: int) -> float:
    fresh = [v for v in violations if v.step == step]
    return min(1.0, sum(v.penalty for v in fresh))


def _invalid_penalty(invalid: bool) -> float:
    return 1.0 if invalid else 0.0


def _no_progress_penalty(jobs: Iterable[Job]) -> float:
    queued = [j for j in jobs if j.status == JobStatus.QUEUED]
    running = [j for j in jobs if j.status == JobStatus.RUNNING]
    if not queued and not running:
        return 0.0
    if running:
        return 0.0
    if queued and len(queued) > 0:
        return min(1.0, 0.05 * len(queued))
    return 0.0


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def compute_reward(
    state_before: ClusterMindState,
    state_after: ClusterMindState,
    *,
    invalid_action: bool,
    useful_action: bool,
    step: int,
) -> RewardBreakdown:
    breakdown = RewardBreakdown(
        critical_job_completion=_critical_completion(state_after.jobs),
        normal_job_completion=_normal_completion(state_after.jobs),
        deadline_score=_deadline_score(state_after.jobs),
        cluster_health_score=_cluster_health(state_after),
        thermal_safety_score=_thermal_safety(state_after.nodes),
        recovery_score=_recovery(state_after.nodes, state_before.outage_count, state_after.outage_count),
        energy_efficiency_score=_energy_efficiency(state_after),
        useful_inspection_or_maintenance=_useful_inspect_or_maint(useful_action),
        outage_penalty=_outage_penalty(state_before.outage_count, state_after.outage_count),
        cascade_penalty=_cascade_penalty(state_before.cascade_count, state_after.cascade_count),
        missed_critical_deadline_penalty=_missed_critical_deadline(state_after.jobs),
        guardrail_violation_penalty=_guardrail_penalty(state_after.guardrail_violations, step),
        invalid_action_penalty=_invalid_penalty(invalid_action),
        no_progress_penalty=_no_progress_penalty(state_after.jobs),
    )
    total = (
        0.25 * breakdown.critical_job_completion
        + 0.15 * breakdown.normal_job_completion
        + 0.15 * breakdown.deadline_score
        + 0.15 * breakdown.cluster_health_score
        + 0.10 * breakdown.thermal_safety_score
        + 0.10 * breakdown.recovery_score
        + 0.05 * breakdown.energy_efficiency_score
        + 0.05 * breakdown.useful_inspection_or_maintenance
        - 0.40 * breakdown.outage_penalty
        - 0.35 * breakdown.cascade_penalty
        - 0.20 * breakdown.missed_critical_deadline_penalty
        - 0.15 * breakdown.guardrail_violation_penalty
        - 0.10 * breakdown.invalid_action_penalty
        - 0.10 * breakdown.no_progress_penalty
    )
    breakdown.total = max(-1.0, min(1.0, total))
    return breakdown
