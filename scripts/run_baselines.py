"""Run all baselines across all scenarios and save baseline_metrics.json.

Usage:
    python scripts/run_baselines.py --episodes 10 --output results/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustermind import ClusterMindChaosEnv
from clustermind.baselines import ALL_BASELINES, RandomAgent
from clustermind.scenarios import SCENARIO_NAMES


def run_episode(env: ClusterMindChaosEnv, agent, scenario: str, level: int, seed: int, max_steps: int) -> Dict[str, Any]:
    agent.reset(seed=seed)
    obs = env.reset(seed=seed, options={"scenario": scenario, "curriculum_level": level, "max_steps": max_steps})
    total_reward = 0.0
    steps = 0
    invalid = 0
    guardrail_violations = 0
    chaos_events = 0
    health_history: List[float] = [obs.cluster_health]
    while not obs.done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("invalid_action_reason"):
            invalid += 1
        guardrail_violations += len(info.get("guardrail_flags", []))
        if info.get("chaos_event"):
            chaos_events += 1
        health_history.append(obs.cluster_health)
    snap = info["metrics_snapshot"] if "metrics_snapshot" in info else {}
    completed_critical = snap.get("completed_critical", 0)
    total_critical = max(1, snap.get("total_critical", 1))
    avg_temp = sum(n.temperature for n in obs.nodes) / max(1, len(obs.nodes))
    avg_health = sum(health_history) / max(1, len(health_history))
    return {
        "scenario": scenario,
        "curriculum_level": level,
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "outage_count": snap.get("outage_count", 0),
        "cascade_count": snap.get("cascade_count", 0),
        "completed_critical": completed_critical,
        "total_critical": total_critical,
        "completed_jobs": snap.get("completed_jobs", 0),
        "failed_jobs": snap.get("failed_jobs", 0),
        "deadline_misses": snap.get("deadline_misses", 0),
        "energy_remaining": obs.energy_remaining,
        "avg_cluster_health": avg_health,
        "avg_temperature": avg_temp,
        "invalid_action_count": invalid,
        "invalid_action_rate": invalid / max(1, steps),
        "guardrail_violations": guardrail_violations,
        "guardrail_violation_rate": guardrail_violations / max(1, steps),
        "chaos_events": chaos_events,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    sum_keys = [
        "reward", "outage_count", "cascade_count", "completed_critical", "total_critical",
        "completed_jobs", "failed_jobs", "deadline_misses", "avg_cluster_health",
        "avg_temperature", "invalid_action_count", "invalid_action_rate",
        "guardrail_violations", "guardrail_violation_rate", "chaos_events",
        "energy_remaining", "steps",
    ]
    out = {f"avg_{k}": sum(r.get(k, 0.0) for r in records) / n for k in sum_keys}
    crit_done = sum(r["completed_critical"] for r in records)
    crit_total = sum(r["total_critical"] for r in records)
    out["critical_completion_rate"] = crit_done / max(1, crit_total)
    out["episodes"] = n
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="episodes per (agent, scenario, level)")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--scenarios", type=str, nargs="+", default=SCENARIO_NAMES)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--mode", choices=["quick", "full"], default=None,
                        help="quick = 5 episodes, full = 20 episodes (overrides --episodes)")
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "results", "baseline_metrics.json"))
    args = parser.parse_args()
    if args.mode == "quick":
        args.episodes = 5
    elif args.mode == "full":
        args.episodes = 20

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    payload: Dict[str, Any] = {
        "schema": "clustermind.baselines.v1",
        "max_steps": args.max_steps,
        "agents": {},
    }
    rng_base = 1000
    started = time.time()

    for agent_name, cls in ALL_BASELINES.items():
        agent_records: List[Dict[str, Any]] = []
        per_scenario: Dict[str, List[Dict[str, Any]]] = {}
        for scenario in args.scenarios:
            for level in args.levels:
                # Skip combinations that don't make sense (chaos_arena requires L5).
                if scenario == "chaos_arena" and level < 5:
                    continue
                for ep in range(args.episodes):
                    seed = rng_base + 71 * ep + 13 * level + hash(scenario) % 9967
                    env = ClusterMindChaosEnv()
                    agent = cls(seed=seed) if cls is RandomAgent else cls()
                    rec = run_episode(env, agent, scenario, level, seed, args.max_steps)
                    rec["agent"] = agent_name
                    agent_records.append(rec)
                    per_scenario.setdefault(f"{scenario}@L{level}", []).append(rec)
                    env.close()
        payload["agents"][agent_name] = {
            "summary": aggregate(agent_records),
            "by_scenario": {k: aggregate(v) for k, v in per_scenario.items()},
            "episodes": agent_records,
        }
        s = payload["agents"][agent_name]["summary"]
        print(
            f"{agent_name:32s} reward={s['avg_reward']:6.2f} "
            f"crit={s['critical_completion_rate']*100:5.1f}% "
            f"outage={s['avg_outage_count']:.2f} cascade={s['avg_cascade_count']:.2f} "
            f"gv_rate={s['avg_guardrail_violation_rate']:.2f} inv_rate={s['avg_invalid_action_rate']:.2f}"
        )

    payload["elapsed_seconds"] = time.time() - started
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
