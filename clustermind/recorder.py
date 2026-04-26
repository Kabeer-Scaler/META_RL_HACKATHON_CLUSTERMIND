"""Flight Recorder — keeps the last 100 steps for replay & failure forensics.

The recorder is intentionally agent-agnostic: it stores observations the
*agent saw* alongside the *true* state transitions for post-hoc analysis. The
``explain_failure_chain`` helper produces the human-readable narrative
(PRD §22 style) that the README and Gradio replay both render.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from clustermind.models import FlightRecord


class FlightRecorder:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.records: Deque[FlightRecord] = deque(maxlen=capacity)
        self.episode_id: Optional[str] = None
        self.failure_steps: List[int] = []
        self.cascade_steps: List[int] = []
        self.guardrail_steps: List[int] = []
        self.energy_exhaustion_step: Optional[int] = None
        self.deadline_miss_steps: List[int] = []

    def reset(self, episode_id: Optional[str] = None) -> None:
        self.records.clear()
        self.episode_id = episode_id
        self.failure_steps.clear()
        self.cascade_steps.clear()
        self.guardrail_steps.clear()
        self.energy_exhaustion_step = None
        self.deadline_miss_steps.clear()

    def record(self, rec: FlightRecord) -> None:
        self.records.append(rec)
        if rec.failure_events:
            self.failure_steps.append(rec.step)
        if rec.cascade_events:
            self.cascade_steps.append(rec.step)
        if rec.guardrail_flags:
            self.guardrail_steps.append(rec.step)
        if any("deadline_miss" in e for e in rec.event_log):
            self.deadline_miss_steps.append(rec.step)
        if any("energy_exhausted" in e for e in rec.event_log):
            self.energy_exhaustion_step = rec.step

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def latest(self, n: int = 5) -> List[FlightRecord]:
        return list(self.records)[-n:]

    def to_json(self) -> str:
        return json.dumps(
            {
                "episode_id": self.episode_id,
                "records": [rec.model_dump() for rec in self.records],
                "summary": {
                    "failure_steps": self.failure_steps,
                    "cascade_steps": self.cascade_steps,
                    "guardrail_steps": self.guardrail_steps,
                    "energy_exhaustion_step": self.energy_exhaustion_step,
                    "deadline_miss_steps": self.deadline_miss_steps,
                },
            },
            indent=2,
        )

    def export(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def explain_failure_chain(self) -> str:
        """Return a human-readable narrative for the latest failure window.

        Picks a 6-step window around the first node failure (or last cascade).
        Falls back to the last 6 steps if no failure occurred.
        """

        if not self.records:
            return "No transitions recorded."

        anchor: Optional[int] = None
        if self.failure_steps:
            anchor = self.failure_steps[0]
        elif self.cascade_steps:
            anchor = self.cascade_steps[0]
        elif self.deadline_miss_steps:
            anchor = self.deadline_miss_steps[0]

        if anchor is None:
            window = list(self.records)[-6:]
        else:
            window = [r for r in self.records if anchor - 3 <= r.step <= anchor + 3]
            if not window:
                window = list(self.records)[-6:]

        lines = []
        for rec in window:
            primary = rec.primary_action.get("action_type", "?")
            chaos = rec.chaos_action or "—"
            event_summary = "; ".join(rec.event_log) if rec.event_log else "no events"
            failures = ", ".join(rec.failure_events) if rec.failure_events else ""
            cascades = ", ".join(rec.cascade_events) if rec.cascade_events else ""
            guardrails = ", ".join(rec.guardrail_flags) if rec.guardrail_flags else ""

            extras = []
            if failures:
                extras.append(f"failures: {failures}")
            if cascades:
                extras.append(f"cascade: {cascades}")
            if guardrails:
                extras.append(f"guardrails: {guardrails}")
            tail = f" — {' | '.join(extras)}" if extras else ""

            lines.append(
                f"Step {rec.step:>2}: agent={primary}  chaos={chaos}  "
                f"events={event_summary}{tail}"
            )
        return "\n".join(lines)
