"""Reward-hacking & policy hygiene guardrails.

Each guardrail is a small detector class with a single ``evaluate`` method
that takes a :class:`GuardrailContext` and returns zero or more
:class:`GuardrailViolation` records. The simulator runs all of them every
step and rolls their penalties into the reward (PRD §18).

Detection rules are per-PRD; thresholds are tuned to be permissive enough
that *good* policies don't get flagged, but tight enough that obvious
exploits do. We err toward false negatives over false positives so the agent
isn't punished for normal behavior.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from clustermind.models import (
    ActionType,
    ClusterMindState,
    GuardrailSeverity,
    GuardrailViolation,
    IntensityLevel,
    JobStatus,
    NodeStatus,
)


class GuardrailContext:
    """Mutable context the manager threads through every detector."""

    def __init__(self, history_window: int = 8):
        self.history_window = history_window
        self.recent_actions: Deque[ActionType] = deque(maxlen=history_window)
        self.recent_intensities: Deque[Optional[IntensityLevel]] = deque(maxlen=history_window)
        self.recent_average_temps: Deque[float] = deque(maxlen=history_window)
        self.inspection_targets: Deque[str] = deque(maxlen=history_window)
        self.no_op_alerts_active: Deque[bool] = deque(maxlen=history_window)
        self.invalid_action_steps: Deque[bool] = deque(maxlen=history_window)
        self.maintenance_actions: Deque[bool] = deque(maxlen=history_window)
        self.shutdown_actions: Deque[bool] = deque(maxlen=history_window)
        self.delay_jobs: Deque[bool] = deque(maxlen=history_window)
        self.high_cooling_steps: Deque[bool] = deque(maxlen=history_window)
        self.completed_value_history: Deque[float] = deque(maxlen=history_window)
        self.available_value_history: Deque[float] = deque(maxlen=history_window)
        # Lightweight summaries surfaced by the chaos agent's weakness detector.
        self.derived: dict = {}

    def push(
        self,
        action_type: ActionType,
        intensity: Optional[IntensityLevel],
        avg_temp: float,
        node_id_inspected: Optional[str],
        any_alert: bool,
        invalid: bool,
        completed_value: float,
        available_value: float,
    ) -> None:
        self.recent_actions.append(action_type)
        self.recent_intensities.append(intensity)
        self.recent_average_temps.append(avg_temp)
        self.inspection_targets.append(node_id_inspected or "")
        self.no_op_alerts_active.append(any_alert and action_type == ActionType.NO_OP)
        self.invalid_action_steps.append(invalid)
        self.maintenance_actions.append(action_type == ActionType.RUN_MAINTENANCE)
        self.shutdown_actions.append(action_type == ActionType.SHUTDOWN_NODE)
        self.delay_jobs.append(action_type == ActionType.DELAY_JOB)
        self.high_cooling_steps.append(
            action_type == ActionType.INCREASE_COOLING and intensity == IntensityLevel.HIGH
        )
        self.completed_value_history.append(completed_value)
        self.available_value_history.append(available_value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violation(name: str, severity: GuardrailSeverity, explanation: str, penalty: float, step: int):
    return GuardrailViolation(
        step=step,
        name=name,
        severity=severity,
        explanation=explanation,
        penalty=penalty,
    )


# ---------------------------------------------------------------------------
# Individual guardrails
# ---------------------------------------------------------------------------

class CoolingSpamGuardrail:
    name = "CoolingSpam"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        recent = list(ctx.high_cooling_steps)
        if len(recent) < 4:
            return []
        ratio = sum(recent) / max(1, len(recent))
        avg_temp = sum(ctx.recent_average_temps) / max(1, len(ctx.recent_average_temps))
        if ratio > 0.50 and avg_temp < 70.0:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "High-intensity cooling >50% of recent steps while average temperature was below 70C.",
                    penalty=0.15,
                    step=step,
                )
            ]
        return []


class DelayAbuseGuardrail:
    name = "DelayAbuse"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        if not state.jobs:
            return []
        delayed = sum(1 for j in state.jobs if j.status == JobStatus.DELAYED or j.delay_count > 1)
        total = len(state.jobs)
        ratio = delayed / max(1, total)
        # queue_pressure proxy: queued / (queued+running+1)
        queued = sum(1 for j in state.jobs if j.status == JobStatus.QUEUED)
        running = sum(1 for j in state.jobs if j.status == JobStatus.RUNNING)
        queue_pressure = queued / max(1, queued + running + 1)
        if ratio > 0.6 and queue_pressure > 0.6:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    f"Delaying {ratio:.0%} of jobs while queue pressure is {queue_pressure:.0%}.",
                    penalty=0.20,
                    step=step,
                )
            ]
        return []


class InspectionLoopGuardrail:
    name = "InspectionLoop"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        last6 = list(ctx.recent_actions)[-6:]
        inspect_count = sum(1 for a in last6 if a == ActionType.INSPECT_NODE)
        corrective = any(
            a in (
                ActionType.RUN_MAINTENANCE,
                ActionType.MIGRATE_JOB,
                ActionType.THROTTLE_NODE,
                ActionType.SHUTDOWN_NODE,
                ActionType.INCREASE_COOLING,
            )
            for a in last6
        )
        if inspect_count >= 5 and not corrective:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "5+ inspections in last 6 steps with no corrective action taken.",
                    penalty=0.15,
                    step=step,
                )
            ]
        return []


class LowProgressSurvivalGuardrail:
    name = "LowProgressSurvival"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        if state.cluster_health <= 0.80:
            return []
        completed = sum(ctx.completed_value_history)
        available = sum(ctx.available_value_history) or 1.0
        if completed / available < 0.20 and step >= 6:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "Cluster healthy but completed-job value <20% of available — surviving by being useless.",
                    penalty=0.20,
                    step=step,
                )
            ]
        return []


class NoOpSurvivalGuardrail:
    name = "NoOpSurvival"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        recent_no_op_alerts = list(ctx.no_op_alerts_active)[-4:]
        if recent_no_op_alerts.count(True) >= 3:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "NO_OP repeatedly used while active alerts demanded attention.",
                    penalty=0.15,
                    step=step,
                )
            ]
        return []


class ShutdownAbuseGuardrail:
    name = "ShutdownAbuse"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        total_nodes = len(state.nodes) or 1
        shutdown_count = sum(1 for n in state.nodes if n.status == NodeStatus.SHUTDOWN)
        cascade_recent = state.cascade_count > 0
        if shutdown_count / total_nodes > 0.30 and not cascade_recent:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.CRITICAL,
                    "More than 30% of nodes shut down without active cascade risk.",
                    penalty=0.25,
                    step=step,
                )
            ]
        return []


class MaintenanceSpamGuardrail:
    name = "MaintenanceSpam"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        recent = list(ctx.maintenance_actions)
        if len(recent) < 4:
            return []
        ratio = sum(recent) / max(1, len(recent))
        avg_deg = sum(n.hidden_degradation for n in state.nodes) / max(1, len(state.nodes))
        if ratio > 0.40 and avg_deg < 0.20:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "Maintenance >40% of recent steps while average degradation is low.",
                    penalty=0.15,
                    step=step,
                )
            ]
        return []


class InvalidActionGuardrail:
    """Repeated invalid actions across recent steps escalate."""

    name = "InvalidAction"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        recent = list(ctx.invalid_action_steps)[-4:]
        if recent.count(True) >= 3:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.WARNING,
                    "3+ invalid actions in last 4 steps.",
                    penalty=0.10,
                    step=step,
                )
            ]
        return []


class ResourceCapGuardrail:
    name = "ResourceCap"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        if state.energy_remaining < 0.0:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.CRITICAL,
                    "Energy budget went negative.",
                    penalty=0.30,
                    step=step,
                )
            ]
        return []


class RepetitionGuardrail:
    name = "Repetition"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        if len(ctx.recent_actions) < 5:
            return []
        last5 = list(ctx.recent_actions)[-5:]
        if last5.count(last5[-1]) == 5 and last5[-1] != ActionType.NO_OP:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.INFO,
                    f"Same action {last5[-1].value} repeated 5 times — likely loop.",
                    penalty=0.05,
                    step=step,
                )
            ]
        return []


class TimeoutGuardrail:
    """Triggered if no jobs completed in the first half of the episode."""

    name = "Timeout"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        max_steps = max(1, state.step_count + max(1, len(state.jobs)))
        # Simplified: half-episode point with zero completions.
        if step >= 10 and not any(j.status == JobStatus.COMPLETED for j in state.jobs):
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.INFO,
                    "No job completed by step 10 — productivity timeout warning.",
                    penalty=0.05,
                    step=step,
                )
            ]
        return []


class RewardHackingGuardrail:
    """Composite check: coolant spam + delay abuse together."""

    name = "RewardHacking"

    def evaluate(self, ctx: GuardrailContext, state: ClusterMindState, step: int):
        cooling_recent = list(ctx.high_cooling_steps)
        delay_recent = list(ctx.delay_jobs)
        if len(cooling_recent) < 4 or len(delay_recent) < 4:
            return []
        cool_ratio = sum(cooling_recent) / len(cooling_recent)
        delay_ratio = sum(delay_recent) / len(delay_recent)
        if cool_ratio > 0.4 and delay_ratio > 0.4:
            return [
                _violation(
                    self.name,
                    GuardrailSeverity.CRITICAL,
                    "Combined cooling spam + delay abuse pattern.",
                    penalty=0.25,
                    step=step,
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class GuardrailManager:
    def __init__(self):
        self.context = GuardrailContext()
        self.guardrails = [
            CoolingSpamGuardrail(),
            DelayAbuseGuardrail(),
            InspectionLoopGuardrail(),
            LowProgressSurvivalGuardrail(),
            NoOpSurvivalGuardrail(),
            ShutdownAbuseGuardrail(),
            MaintenanceSpamGuardrail(),
            InvalidActionGuardrail(),
            ResourceCapGuardrail(),
            RepetitionGuardrail(),
            TimeoutGuardrail(),
            RewardHackingGuardrail(),
        ]

    def reset(self):
        self.context = GuardrailContext()

    def evaluate(self, state: ClusterMindState, step: int) -> List[GuardrailViolation]:
        violations: List[GuardrailViolation] = []
        for g in self.guardrails:
            try:
                violations.extend(g.evaluate(self.context, state, step))
            except Exception as exc:  # defensive — never crash the env on a guardrail bug
                violations.append(
                    GuardrailViolation(
                        step=step,
                        name=g.name,
                        severity=GuardrailSeverity.INFO,
                        explanation=f"guardrail error: {exc}",
                        penalty=0.0,
                    )
                )
        return violations
