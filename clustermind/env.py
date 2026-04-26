"""OpenEnv wrapper.

Exposes :class:`ClusterMindChaosEnv` with the four-method OpenEnv API
(`reset` / `step` / `state` / `close`). The env owns one
:class:`ClusterSimulator` and forwards calls to it. We keep the API tolerant
of both Pydantic ``ClusterMindAction`` instances *and* dict payloads so the
FastAPI bridge from ``openenv.core.env_server.create_fastapi_app`` works
without a custom adapter.
"""

from __future__ import annotations

import copy
import os
import uuid
from typing import Any, Dict, Optional, Tuple

from clustermind.models import (
    ActionType,
    ClusterMindAction,
    ClusterMindObservation,
    ClusterMindState,
    IntensityLevel,
)
from clustermind.scenarios import SCENARIO_NAMES
from clustermind.simulator import ClusterSimulator, DEFAULT_MAX_STEPS


class ClusterMindChaosEnv:
    """OpenEnv-compatible environment for the ClusterMind benchmark.

    Parameters
    ----------
    default_scenario : str
        Used by ``reset()`` when no override is supplied.
    default_curriculum_level : int
        Used by ``reset()`` when no override is supplied.
    default_max_steps : int
        Episode length cap (PRD §6 requires 20).
    """

    def __init__(
        self,
        default_scenario: str = "demand_spike",
        default_curriculum_level: int = 1,
        default_max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.simulator = ClusterSimulator()
        self.default_scenario = default_scenario
        self.default_curriculum_level = default_curriculum_level
        self.default_max_steps = default_max_steps
        self._closed = False
        self._last_observation: Optional[ClusterMindObservation] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ClusterMindObservation:
        opts = dict(options or {})
        scenario = opts.get("scenario", self.default_scenario)
        if scenario not in SCENARIO_NAMES:
            scenario = self.default_scenario
        curriculum = int(opts.get("curriculum_level", self.default_curriculum_level))
        max_steps = int(opts.get("max_steps", self.default_max_steps))
        if seed is None:
            seed = opts.get("seed")
        episode_id = opts.get("episode_id") or f"ep_{uuid.uuid4().hex[:8]}"
        obs = self.simulator.reset(
            scenario=scenario,
            curriculum_level=curriculum,
            seed=seed,
            max_steps=max_steps,
            episode_id=episode_id,
        )
        self._last_observation = obs
        return obs

    def step(self, action: Any) -> Tuple[ClusterMindObservation, float, bool, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment has been closed.")
        action_obj = self._coerce_action(action)
        obs, reward, done, info = self.simulator.step(action_obj)
        self._last_observation = obs
        return obs, reward, done, info

    @property
    def state(self) -> ClusterMindState:
        return self.simulator.state

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_action(self, action: Any) -> ClusterMindAction:
        if isinstance(action, ClusterMindAction):
            return action
        if isinstance(action, dict):
            try:
                return ClusterMindAction.model_validate(action)
            except Exception:
                # Fall through to defensive parse
                return ClusterMindAction(action_type=ActionType.NO_OP, metadata={"parse_error": True})
        if isinstance(action, str):
            try:
                return ClusterMindAction(action_type=ActionType(action))
            except Exception:
                return ClusterMindAction(action_type=ActionType.NO_OP, metadata={"parse_error": True})
        # Unknown type
        return ClusterMindAction(action_type=ActionType.NO_OP, metadata={"parse_error": True})

    @property
    def last_observation(self) -> Optional[ClusterMindObservation]:
        return self._last_observation

    @property
    def episode_id(self) -> Optional[str]:
        return self.simulator.episode_id

    @property
    def recorder(self):
        return self.simulator.recorder

    @property
    def cluster_health_history(self):
        return list(self.simulator.cluster_health_history)


# ---------------------------------------------------------------------------
# Compatibility alias
# ---------------------------------------------------------------------------

ClusterMindEnv = ClusterMindChaosEnv
