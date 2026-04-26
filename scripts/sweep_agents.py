"""Full agent x scenario sweep with side-by-side reporting.

Runs every baseline on every scenario across multiple seeds, prints a
human-readable table per scenario, then a per-agent profile, and finally
captures one Flight Recorder failure-chain narrative per (worst-agent,
scenario) pair so the storytelling is concrete.

Read-only: this script does not modify env code. It just calls reset/step.

Usage:
    python scripts/sweep_agents.py --episodes 6 --levels 3 4 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustermind import ClusterMindChaosEnv
from clustermind.baselines import ALL_BASELINES, RandomAgent
from clustermind.scenarios import SCENARIO_NAMES


SCENARIO_PURPOSE = {
    "demand_spike": "priority scheduling under burst load",
    "cooling_failure": "thermal control with degraded zone",
    "hidden_degradation": "partial observability via inspection",
    "cascading_failure": "long-horizon failure containment",
    "energy_squeeze": "energy discipline under tight budget",
    "vip_job_arrival": "preserve capacity for late critical job",
    "triple_crisis": "combined demand + cooling + hidden",
    "chaos_arena": "robustness under adversarial chaos",
}


def run_one(agent_cls, scenario: str, level: int, seed: int, max_steps: int = 20) -> Dict[str, Any]:
    agent = agent_cls(seed=seed) if agent_cls is RandomAgent else agent_cls()
    agent.reset(seed=seed)
    env = ClusterMindChaosEnv()
    obs = env.reset(seed=seed, options={
        "scenario": scenario, "curriculum_level": level, "max_steps": max_steps,
    })
    total_reward = 0.0
    steps = 0
    invalid = 0
    guardrail_violations = 0
    chaos_events = 0
    action_counts: Dict[str, int] = defaultdict(int)
    while not obs.done:
        action = agent.act(obs)
        action_counts[action.action_type.value] += 1
        obs, r, done, info = env.step(action)
        total_reward += r
        steps += 1
        if info.get("invalid_action_reason"):
            invalid += 1
        guardrail_violations += len(info.get("guardrail_flags", []))
        if info.get("chaos_event"):
            chaos_events += 1
    snap = info["metrics_snapshot"]
    crit_done = snap["completed_critical"]
    crit_total = max(1, snap["total_critical"])
    failure_chain = env.recorder.explain_failure_chain()
    env.close()
    return {
        "reward": total_reward,
        "steps": steps,
        "outage_count": snap["outage_count"],
        "cascade_count": snap["cascade_count"],
        "critical_completion": crit_done / crit_total,
        "completed_critical": crit_done,
        "total_critical": crit_total,
        "completed_jobs": snap["completed_jobs"],
        "failed_jobs": snap["failed_jobs"],
        "deadline_misses": snap["deadline_misses"],
        "cluster_health": snap["cluster_health"],
        "avg_temperature": snap["average_temperature"],
        "energy_remaining": snap["energy_remaining"],
        "invalid_action_count": invalid,
        "guardrail_violations": guardrail_violations,
        "chaos_events": chaos_events,
        "action_counts": dict(action_counts),
        "failure_chain": failure_chain,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(records))
    keys = ["reward", "outage_count", "cascade_count", "critical_completion",
            "completed_jobs", "failed_jobs", "deadline_misses", "cluster_health",
            "avg_temperature", "energy_remaining", "invalid_action_count",
            "guardrail_violations", "chaos_events"]
    out = {k: sum(r[k] for r in records) / n for k in keys}
    out["episodes"] = len(records)
    return out


def print_per_scenario_tables(results: Dict[Tuple[str, str], Dict[str, float]],
                              scenarios: List[str], agents: List[str], level: int):
    print()
    print("=" * 92)
    print(f"PER-SCENARIO RESULTS  (curriculum level {level}, averaged across all seeds)")
    print("=" * 92)
    for scenario in scenarios:
        print(f"\n--- {scenario}  ({SCENARIO_PURPOSE.get(scenario, '')}) ---")
        header = f"{'agent':30s} {'reward':>7s} {'crit%':>6s} {'outage':>6s} {'cascade':>7s} {'gv':>4s} {'health':>6s} {'energy':>6s}"
        print(header)
        print("-" * len(header))
        rows = []
        for agent in agents:
            m = results.get((agent, scenario))
            if m is None:
                continue
            rows.append((agent, m))
        rows.sort(key=lambda x: -x[1]["reward"])
        for agent, m in rows:
            print(f"{agent:30s} {m['reward']:7.2f} "
                  f"{m['critical_completion']*100:5.1f}% "
                  f"{m['outage_count']:6.2f} {m['cascade_count']:7.2f} "
                  f"{m['guardrail_violations']:4.1f} "
                  f"{m['cluster_health']:6.2f} {m['energy_remaining']:6.2f}")


def print_per_agent_profile(results: Dict[Tuple[str, str], Dict[str, float]],
                            scenarios: List[str], agents: List[str]):
    print()
    print("=" * 92)
    print("PER-AGENT PROFILE  (across all scenarios, all seeds)")
    print("=" * 92)
    for agent in agents:
        all_records = [results[(agent, s)] for s in scenarios if (agent, s) in results]
        if not all_records:
            continue
        n = len(all_records)
        avg_reward = sum(r["reward"] for r in all_records) / n
        avg_crit = sum(r["critical_completion"] for r in all_records) / n
        avg_outage = sum(r["outage_count"] for r in all_records) / n
        avg_cascade = sum(r["cascade_count"] for r in all_records) / n
        avg_gv = sum(r["guardrail_violations"] for r in all_records) / n
        # Best/worst scenario for this agent.
        rb = max(all_records, key=lambda r: r["reward"])
        wb = min(all_records, key=lambda r: r["reward"])
        rb_name = scenarios[all_records.index(rb)]
        wb_name = scenarios[all_records.index(wb)]
        print(f"\n{agent}")
        print(f"  avg reward         = {avg_reward:6.2f}")
        print(f"  avg critical-done  = {avg_crit*100:5.1f}%")
        print(f"  avg outages        = {avg_outage:6.2f}")
        print(f"  avg cascades       = {avg_cascade:6.2f}")
        print(f"  avg guardrails     = {avg_gv:6.2f}")
        print(f"  best scenario      = {rb_name}  (reward {rb['reward']:.2f})")
        print(f"  worst scenario     = {wb_name}  (reward {wb['reward']:.2f})")


def print_action_mix(action_counts_by_agent: Dict[str, Dict[str, int]], agents: List[str]):
    print()
    print("=" * 92)
    print("ACTION MIX (how often each agent uses each verb, fraction of total actions)")
    print("=" * 92)
    verbs = ["ALLOCATE_JOB", "DELAY_JOB", "THROTTLE_NODE", "INCREASE_COOLING",
             "RUN_MAINTENANCE", "MIGRATE_JOB", "INSPECT_NODE", "SHUTDOWN_NODE", "NO_OP"]
    header = f"{'agent':30s} " + "  ".join(f"{v[:8]:>8s}" for v in verbs)
    print(header)
    print("-" * len(header))
    for agent in agents:
        counts = action_counts_by_agent.get(agent, {})
        total = sum(counts.values()) or 1
        cells = []
        for v in verbs:
            frac = counts.get(v, 0) / total
            cells.append(f"{frac*100:7.1f}%")
        print(f"{agent:30s} " + "  ".join(cells))


def print_one_failure_chain(results_raw: Dict[Tuple[str, str, int], Dict[str, Any]],
                            scenario: str, agent: str):
    matches = [(seed, r) for (a, s, seed), r in results_raw.items()
               if a == agent and s == scenario]
    if not matches:
        return
    # Pick the worst one (lowest reward) — that's the most interesting story.
    seed, rec = min(matches, key=lambda x: x[1]["reward"])
    print(f"\n--- {agent} on {scenario}, seed {seed} (reward {rec['reward']:.2f}) ---")
    print(f"outcome: {rec['completed_critical']}/{rec['total_critical']} critical jobs done, "
          f"{rec['outage_count']} outages, {rec['cascade_count']} cascades, "
          f"health={rec['cluster_health']:.2f}, energy={rec['energy_remaining']:.2f}")
    chain = rec["failure_chain"]
    if chain:
        print("flight recorder window:")
        for line in chain.splitlines():
            print(f"  {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6, help="seeds per (agent, scenario)")
    parser.add_argument("--levels", type=int, nargs="+", default=[4])
    parser.add_argument("--scenarios", type=str, nargs="+", default=SCENARIO_NAMES)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-json", type=str,
                        default=os.path.join(ROOT, "results", "agent_sweep.json"))
    args = parser.parse_args()

    agents = list(ALL_BASELINES.keys())
    seeds = list(range(101, 101 + args.episodes))

    print(f"Running sweep: {len(agents)} agents x {len(args.scenarios)} scenarios "
          f"x {len(args.levels)} levels x {args.episodes} seeds = "
          f"{len(agents) * len(args.scenarios) * len(args.levels) * args.episodes} episodes")
    print(f"agents: {agents}")
    print(f"scenarios: {args.scenarios}")
    print(f"levels: {args.levels}")

    # Aggregated results: (agent, scenario) -> averaged metrics
    agg_per_level: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {lvl: {} for lvl in args.levels}
    raw: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    action_mix: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in agents}

    for level in args.levels:
        for agent_name in agents:
            cls = ALL_BASELINES[agent_name]
            for scenario in args.scenarios:
                # chaos_arena requires level 5
                effective_level = max(level, 5) if scenario == "chaos_arena" else level
                records = []
                for seed in seeds:
                    rec = run_one(cls, scenario, effective_level, seed, args.max_steps)
                    records.append(rec)
                    raw[(agent_name, scenario, seed)] = rec
                    for verb, cnt in rec["action_counts"].items():
                        action_mix[agent_name][verb] += cnt
                agg_per_level[level][(agent_name, scenario)] = aggregate(records)

    # Print outputs ----------------------------------------------------------
    for level in args.levels:
        print_per_scenario_tables(agg_per_level[level], args.scenarios, agents, level)
        print_per_agent_profile(agg_per_level[level], args.scenarios, agents)
        print_action_mix({a: dict(action_mix[a]) for a in agents}, agents)

    # One concrete failure-chain narrative per (agent, scenario): pick a few
    # of the most illustrative ones.
    print()
    print("=" * 92)
    print("REPRESENTATIVE FAILURE-CHAIN NARRATIVES")
    print("=" * 92)
    illustrative = [
        ("GreedyThroughputAgent", "triple_crisis"),
        ("ConservativeAutoscalerAgent", "demand_spike"),
        ("RandomAgent", "cascading_failure"),
        ("ThermalAwareHeuristicAgent", "hidden_degradation"),
        ("BackfillAgent", "energy_squeeze"),
    ]
    for agent_name, scenario in illustrative:
        print_one_failure_chain(raw, scenario, agent_name)

    # Save full data
    serialisable = {
        "schema": "clustermind.agent_sweep.v1",
        "episodes_per_cell": args.episodes,
        "seeds": seeds,
        "agents": agents,
        "scenarios": args.scenarios,
        "levels": args.levels,
        "results": {
            f"L{lvl}|{agent}|{scenario}": metrics
            for lvl, m in agg_per_level.items()
            for (agent, scenario), metrics in m.items()
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[saved] {args.output_json}")


if __name__ == "__main__":
    main()
