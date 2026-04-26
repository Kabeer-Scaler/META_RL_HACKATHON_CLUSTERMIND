
"""Composable evaluation graders.

Each grader inspects either a per-step transition or an episode summary and
emits a 0..1 score plus a letter grade (A/B/C/D/F per PRD §20). Graders here
are independent of the simulator so they can run on already-collected
metrics dicts produced by ``scripts/evaluate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _band(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "F"


@dataclass
class GraderResult:
    name: str
    score: float
    band: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Episode-level graders
# ---------------------------------------------------------------------------

class JobCompletionGrader:
    name = "JobCompletion"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        total = max(1.0, m.get("total_jobs", 1.0))
        completed = m.get("completed_jobs", 0.0)
        score = max(0.0, min(1.0, completed / total))
        return GraderResult(self.name, score, _band(score))


class CriticalJobGrader:
    name = "CriticalJob"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        total = max(1.0, m.get("total_critical", 1.0))
        completed = m.get("completed_critical", 0.0)
        score = max(0.0, min(1.0, completed / total))
        return GraderResult(self.name, score, _band(score))


class DeadlineGrader:
    name = "Deadline"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        total = max(1.0, m.get("total_jobs", 1.0))
        misses = m.get("deadline_misses", 0.0)
        score = max(0.0, 1.0 - misses / total)
        return GraderResult(self.name, score, _band(score))


class ClusterHealthGrader:
    name = "ClusterHealth"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        score = max(0.0, min(1.0, m.get("avg_cluster_health", 0.0)))
        return GraderResult(self.name, score, _band(score))


class ThermalSafetyGrader:
    name = "ThermalSafety"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        avg_temp = m.get("avg_temperature", 60.0)
        # Below 70 → 1.0; at 95 → 0.0.
        score = max(0.0, min(1.0, (95.0 - avg_temp) / 25.0))
        return GraderResult(self.name, score, _band(score))


class EnergyEfficiencyGrader:
    name = "EnergyEfficiency"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        score = max(0.0, min(1.0, m.get("avg_energy_remaining", 0.0)))
        return GraderResult(self.name, score, _band(score))


class RecoveryGrader:
    name = "Recovery"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        outages = m.get("outage_count", 0.0)
        recovered = m.get("recoveries", 0.0)
        if outages == 0:
            return GraderResult(self.name, 1.0, "A", detail="no outages")
        score = max(0.0, min(1.0, recovered / outages))
        return GraderResult(self.name, score, _band(score))


class CascadingFailureGrader:
    name = "CascadingFailure"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        cascades = m.get("cascade_count", 0.0)
        # 0 cascades → 1.0; >=3 cascades → 0.0.
        score = max(0.0, min(1.0, 1.0 - cascades / 3.0))
        return GraderResult(self.name, score, _band(score))


class ActionValidityGrader:
    name = "ActionValidity"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        rate = m.get("invalid_action_rate", 0.0)
        score = max(0.0, min(1.0, 1.0 - rate))
        return GraderResult(self.name, score, _band(score))


class GuardrailViolationGrader:
    name = "GuardrailViolation"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        rate = m.get("guardrail_violation_rate", 0.0)
        score = max(0.0, min(1.0, 1.0 - rate))
        return GraderResult(self.name, score, _band(score))


class RewardHackingGrader:
    """A composite — high reward + low job completion is suspicious."""

    name = "RewardHacking"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        reward = m.get("avg_reward", 0.0)
        completion = m.get("completion_rate", 0.0)
        if reward > 0.5 and completion < 0.3:
            score = 0.0
        elif reward > 0.3 and completion < 0.4:
            score = 0.5
        else:
            score = max(0.0, min(1.0, completion + 0.2))
        return GraderResult(self.name, score, _band(score))


class ChaosSurvivalGrader:
    name = "ChaosSurvival"

    def grade(self, m: Dict[str, float]) -> GraderResult:
        score = m.get("chaos_survival_score", 0.0)
        # Survival is summed; bring into [0,1]: assume good=1.5+, bad<=0.0.
        normalized = max(0.0, min(1.0, (score + 1.0) / 3.0))
        return GraderResult(self.name, normalized, _band(normalized))


# ---------------------------------------------------------------------------
# Registry & orchestration
# ---------------------------------------------------------------------------

ALL_GRADERS = [
    JobCompletionGrader(),
    CriticalJobGrader(),
    DeadlineGrader(),
    ClusterHealthGrader(),
    ThermalSafetyGrader(),
    EnergyEfficiencyGrader(),
    RecoveryGrader(),
    CascadingFailureGrader(),
    ActionValidityGrader(),
    GuardrailViolationGrader(),
    RewardHackingGrader(),
    ChaosSurvivalGrader(),
]


def grade_metrics(metrics: Dict[str, float]) -> Tuple[List[GraderResult], float, str]:
    results = [g.grade(metrics) for g in ALL_GRADERS]
    overall = sum(r.score for r in results) / max(1, len(results))
    return results, overall, _band(overall)


def chaos_survival_score(
    *,
    completed_critical_under_chaos: float,
    cluster_health_after_chaos: float,
    cascade_penalties: float,
    guardrail_violations: float,
) -> float:
    return completed_critical_under_chaos + cluster_health_after_chaos - cascade_penalties - guardrail_violations
