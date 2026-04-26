"""Load a trained ClusterMind agent for demo inference.

Two modes are supported, mirroring the two training paths in
``scripts/train_trl.py``:

  * **policy** — load ``results/adapters/policy_net.pt`` (the small torch MLP
    over engineered features) and step it through one episode.
  * **llm** — load the LoRA adapter at ``results/adapters/clustermind_lora``
    on top of the configured base model and step it.

Used by the Gradio Space when judges click the "trained agent" toggle, and
also exposes a ``run_demo`` function callable from notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from clustermind import ClusterMindChaosEnv
from clustermind.agents import (
    HeuristicFallbackBackend,
    LLMJsonAgent,
    TransformersBackend,
)
from clustermind.baselines import ThermalAwareHeuristicAgent


def load_llm_agent(base_model: str, adapter_path: str, label: str = "RL-LoRA") -> LLMJsonAgent:
    backend = TransformersBackend(model_name=base_model, adapter_path=adapter_path)
    return LLMJsonAgent(
        backend=backend,
        fallback_baseline=ThermalAwareHeuristicAgent(),
        label=label,
    )


def load_policy_net_agent(adapter_path: str):
    """Return an agent-shaped object that steps a torch policy net."""

    import torch

    from scripts.train_trl import (  # type: ignore
        ABSTRACT_ACTIONS, FEATURE_NAMES, _PolicyNet, _ground_abstract_action, featurize,
    )
    from clustermind.baselines import BaselineAgent
    from clustermind.models import ClusterMindAction, ActionType

    class _PolicyAgent(BaselineAgent):
        name = "TrainedPolicyNet"

        def __init__(self):
            self.net = _PolicyNet(len(FEATURE_NAMES), len(ABSTRACT_ACTIONS))
            state = torch.load(adapter_path, map_location="cpu")
            self.net.model.load_state_dict(state)
            self.net.model.eval()
            import random as _rng
            self._rng = _rng.Random(0)

        def reset(self, seed: Optional[int] = None) -> None:
            if seed is not None:
                import random as _rng
                self._rng = _rng.Random(seed)

        def act(self, obs):
            feats = featurize(obs)
            idx = self.net.greedy(feats)
            return _ground_abstract_action(obs, ABSTRACT_ACTIONS[idx], self._rng)

    return _PolicyAgent()


def run_demo(
    *,
    mode: str = "auto",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    llm_adapter: str = "results/adapters/clustermind_lora",
    policy_adapter: str = "results/adapters/policy_net.pt",
    scenario: str = "triple_crisis",
    curriculum_level: int = 4,
    seed: int = 7,
    max_steps: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    if mode == "auto":
        if os.path.isdir(llm_adapter):
            mode = "llm"
        elif os.path.isfile(policy_adapter):
            mode = "policy"
        else:
            mode = "heuristic"
    if mode == "llm":
        agent = load_llm_agent(base_model, llm_adapter)
    elif mode == "policy":
        agent = load_policy_net_agent(policy_adapter)
    else:
        agent = ThermalAwareHeuristicAgent()
    agent.reset(seed=seed)
    env = ClusterMindChaosEnv()
    obs = env.reset(seed=seed, options={
        "scenario": scenario, "curriculum_level": curriculum_level, "max_steps": max_steps,
    })
    total_reward = 0.0
    transcript = []
    while not obs.done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if verbose:
            chaos = info.get("chaos_action") or "-"
            print(f"step {obs.step:>2}: {action.action_type.value:18s} chaos={chaos:25s} "
                  f"reward={reward:+.3f} health={obs.cluster_health:.2f}")
        transcript.append({
            "step": obs.step,
            "action": action.action_type.value,
            "reward": reward,
            "chaos": info.get("chaos_action"),
            "guardrails": info.get("guardrail_flags", []),
        })
    snap = info["metrics_snapshot"]
    return {
        "mode": mode,
        "total_reward": total_reward,
        "scenario": scenario,
        "curriculum_level": curriculum_level,
        "metrics": snap,
        "failure_chain": env.recorder.explain_failure_chain(),
        "transcript": transcript,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="auto", choices=["auto", "llm", "policy", "heuristic"])
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--llm-adapter", default="results/adapters/clustermind_lora")
    parser.add_argument("--policy-adapter", default="results/adapters/policy_net.pt")
    parser.add_argument("--scenario", default="triple_crisis")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    out = run_demo(
        mode=args.mode,
        base_model=args.base_model,
        llm_adapter=args.llm_adapter,
        policy_adapter=args.policy_adapter,
        scenario=args.scenario,
        curriculum_level=args.level,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"\nmode={out['mode']}  total_reward={out['total_reward']:.2f}")
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
