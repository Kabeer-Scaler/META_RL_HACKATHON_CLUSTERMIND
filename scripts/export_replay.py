"""Export a Flight Recorder replay JSON for one chosen (agent, scenario, seed).

Usage:
    python scripts/export_replay.py --agent GreedyThroughputAgent \\
                                    --scenario triple_crisis --level 4 --seed 7

Writes:
    results/replays/<agent>__<scenario>__L<level>__seed<seed>.json
    results/replays/<agent>__<scenario>__L<level>__seed<seed>__failure_chain.txt
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustermind import ClusterMindChaosEnv
from clustermind.baselines import ALL_BASELINES, RandomAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="GreedyThroughputAgent",
                        choices=list(ALL_BASELINES))
    parser.add_argument("--scenario", type=str, default="triple_crisis")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "results", "replays"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cls = ALL_BASELINES[args.agent]
    agent = cls(seed=args.seed) if cls is RandomAgent else cls()
    agent.reset(seed=args.seed)

    env = ClusterMindChaosEnv()
    obs = env.reset(seed=args.seed, options={
        "scenario": args.scenario, "curriculum_level": args.level, "max_steps": args.max_steps,
    })
    steps = 0
    while not obs.done:
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)
        steps += 1

    base = f"{args.agent}__{args.scenario}__L{args.level}__seed{args.seed}"
    json_path = os.path.join(args.output_dir, f"{base}.json")
    txt_path = os.path.join(args.output_dir, f"{base}__failure_chain.txt")
    env.recorder.export(json_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(env.recorder.explain_failure_chain())
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    print(f"\n--- failure chain ---")
    print(env.recorder.explain_failure_chain())


if __name__ == "__main__":
    main()
