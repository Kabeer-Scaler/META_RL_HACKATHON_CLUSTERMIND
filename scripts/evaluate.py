"""Unified evaluator for baselines + LLM agents.

Usage examples:
    # baselines only
    python scripts/evaluate.py --episodes 10 --output results/evaluation_metrics.json

    # add a base LLM (echo backend, for plumbing tests)
    python scripts/evaluate.py --episodes 5 --include-llm echo

    # add SFT and RL adapters from the training pipeline
    python scripts/evaluate.py --episodes 5 \\
        --include-llm transformers \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --sft-adapter results/adapters/sft_lora \\
        --rl-adapter results/adapters/clustermind_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustermind import ClusterMindChaosEnv
from clustermind.agents import (
    EchoBackend,
    HeuristicFallbackBackend,
    LLMJsonAgent,
    OpenAICompatBackend,
    TransformersBackend,
)
from clustermind.baselines import ALL_BASELINES, BaselineAgent, RandomAgent, ThermalAwareHeuristicAgent
from clustermind.graders import grade_metrics
from clustermind.scenarios import SCENARIO_NAMES
from scripts.run_baselines import aggregate, run_episode  # type: ignore  # noqa: E402


def collect_for_agent(agent_factory, agent_name: str, scenarios: List[str], levels: List[int],
                      episodes: int, max_steps: int, seed_base: int = 5000) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    per_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for scenario in scenarios:
        for level in levels:
            if scenario == "chaos_arena" and level < 5:
                continue
            for ep in range(episodes):
                seed = seed_base + 71 * ep + 13 * level + hash(scenario) % 9967
                env = ClusterMindChaosEnv()
                agent = agent_factory()
                rec = run_episode(env, agent, scenario, level, seed, max_steps)
                rec["agent"] = agent_name
                records.append(rec)
                per_scenario.setdefault(f"{scenario}@L{level}", []).append(rec)
                env.close()
    summary = aggregate(records)
    grader_results, overall_score, overall_band = grade_metrics({
        "completed_jobs": summary.get("avg_completed_jobs", 0.0),
        "total_jobs": summary.get("avg_completed_jobs", 0.0) + summary.get("avg_failed_jobs", 0.0) + 1.0,
        "completed_critical": summary.get("avg_completed_critical", 0.0),
        "total_critical": summary.get("avg_total_critical", 1.0),
        "deadline_misses": summary.get("avg_deadline_misses", 0.0),
        "avg_cluster_health": summary.get("avg_avg_cluster_health", 0.0),
        "avg_temperature": summary.get("avg_avg_temperature", 60.0),
        "avg_energy_remaining": summary.get("avg_energy_remaining", 0.0),
        "outage_count": summary.get("avg_outage_count", 0.0),
        "recoveries": 0.0,  # not tracked separately in baseline pass
        "cascade_count": summary.get("avg_cascade_count", 0.0),
        "invalid_action_rate": summary.get("avg_invalid_action_rate", 0.0),
        "guardrail_violation_rate": summary.get("avg_guardrail_violation_rate", 0.0),
        "avg_reward": summary.get("avg_reward", 0.0),
        "completion_rate": summary.get("critical_completion_rate", 0.0),
        "chaos_survival_score": summary.get("avg_reward", 0.0) * summary.get("critical_completion_rate", 0.0),
    })
    return {
        "summary": summary,
        "by_scenario": {k: aggregate(v) for k, v in per_scenario.items()},
        "episodes": records,
        "graders": [r.__dict__ for r in grader_results],
        "overall_score": overall_score,
        "overall_band": overall_band,
    }


def build_llm_agent(kind: str, base_model: Optional[str], adapter: Optional[str], label: str):
    fallback = ThermalAwareHeuristicAgent()
    if kind == "echo":
        backend = EchoBackend()
    elif kind == "heuristic":
        backend = HeuristicFallbackBackend(ThermalAwareHeuristicAgent())
    elif kind == "transformers":
        if base_model is None:
            raise ValueError("--base-model is required for --include-llm transformers")
        backend = TransformersBackend(model_name=base_model, adapter_path=adapter)
    elif kind == "openai-compat":
        if base_model is None:
            raise ValueError("--base-model (model id) required for --include-llm openai-compat")
        backend = OpenAICompatBackend(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=base_model,
        )
    else:
        raise ValueError(f"Unknown llm kind {kind}")
    return LLMJsonAgent(backend=backend, fallback_baseline=fallback, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--scenarios", type=str, nargs="+", default=SCENARIO_NAMES)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "results", "evaluation_metrics.json"))
    parser.add_argument("--include-llm", choices=["none", "echo", "heuristic", "transformers", "openai-compat"], default="none")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--sft-adapter", type=str, default=None)
    parser.add_argument("--rl-adapter", type=str, default=None)
    parser.add_argument("--baselines", type=str, nargs="+", default=list(ALL_BASELINES.keys()))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    payload: Dict[str, Any] = {
        "schema": "clustermind.evaluation.v1",
        "scenarios": args.scenarios,
        "levels": args.levels,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "agents": {},
    }
    started = time.time()

    for name in args.baselines:
        if name not in ALL_BASELINES:
            print(f"skip unknown baseline {name}")
            continue
        cls = ALL_BASELINES[name]
        factory = (lambda c=cls: c(seed=int(time.time() * 1000) % 100000)) if cls is RandomAgent else cls
        result = collect_for_agent(factory, name, args.scenarios, args.levels, args.episodes, args.max_steps)
        payload["agents"][name] = result
        s = result["summary"]
        print(
            f"{name:32s} reward={s.get('avg_reward', 0):6.2f} "
            f"crit={s.get('critical_completion_rate', 0)*100:5.1f}% "
            f"outage={s.get('avg_outage_count', 0):.2f} "
            f"cascade={s.get('avg_cascade_count', 0):.2f} "
            f"grade={result['overall_band']}"
        )

    if args.include_llm != "none":
        # Always evaluate base LLM. Add SFT/RL only if adapters were given.
        try:
            base_agent = build_llm_agent(args.include_llm, args.base_model, None, label="BaseLLM")
            payload["agents"]["BaseLLM"] = collect_for_agent(lambda: base_agent, "BaseLLM",
                                                            args.scenarios, args.levels, args.episodes, args.max_steps)
            print(f"BaseLLM    reward={payload['agents']['BaseLLM']['summary'].get('avg_reward', 0):.2f}")
        except Exception as e:
            print(f"BaseLLM evaluation failed: {e}")

        if args.sft_adapter:
            try:
                sft_agent = build_llm_agent(args.include_llm, args.base_model, args.sft_adapter, label="SFT-LoRA")
                payload["agents"]["SFT-LoRA"] = collect_for_agent(lambda: sft_agent, "SFT-LoRA",
                                                                  args.scenarios, args.levels, args.episodes, args.max_steps)
                print(f"SFT-LoRA   reward={payload['agents']['SFT-LoRA']['summary'].get('avg_reward', 0):.2f}")
            except Exception as e:
                print(f"SFT-LoRA evaluation failed: {e}")

        if args.rl_adapter:
            try:
                rl_agent = build_llm_agent(args.include_llm, args.base_model, args.rl_adapter, label="RL-LoRA")
                payload["agents"]["RL-LoRA"] = collect_for_agent(lambda: rl_agent, "RL-LoRA",
                                                                  args.scenarios, args.levels, args.episodes, args.max_steps)
                print(f"RL-LoRA    reward={payload['agents']['RL-LoRA']['summary'].get('avg_reward', 0):.2f}")
            except Exception as e:
                print(f"RL-LoRA evaluation failed: {e}")

    payload["elapsed_seconds"] = time.time() - started
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
