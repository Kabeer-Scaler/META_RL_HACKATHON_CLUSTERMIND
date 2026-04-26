"""Generate the eight required plots from real logs.

Reads:
    results/baseline_metrics.json
    results/evaluation_metrics.json   (optional)
    results/training_logs.jsonl       (optional)
    results/trained_results.json      (optional)

Writes:
    results/reward_curve.png
    results/loss_curve.png
    results/outage_comparison.png
    results/cascade_count_comparison.png
    results/critical_job_completion.png
    results/guardrail_violations.png
    results/chaos_survival_score.png
    results/cluster_health_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _agent_summary(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, blob in payload.get("agents", {}).items():
        s = blob.get("summary", {})
        out[name] = {
            "avg_reward": s.get("avg_reward", 0.0),
            "avg_outage_count": s.get("avg_outage_count", 0.0),
            "avg_cascade_count": s.get("avg_cascade_count", 0.0),
            "avg_guardrail_violations": s.get("avg_guardrail_violations", 0.0),
            "critical_completion_rate": s.get("critical_completion_rate", 0.0),
            "avg_cluster_health": s.get("avg_avg_cluster_health", 0.0),
            "chaos_survival": s.get("avg_reward", 0.0) * s.get("critical_completion_rate", 0.0),
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reward_curve(training_logs: List[Dict[str, Any]], out_path: str) -> bool:
    rl_entries = [e for e in training_logs if e.get("phase") == "rl" and isinstance(e.get("reward"), (int, float))]
    if not rl_entries:
        rl_entries = [e for e in training_logs if isinstance(e.get("reward"), (int, float))]
    if not rl_entries:
        return _placeholder_plot(
            out_path,
            title="Reward Curve",
            xlabel="Training episode",
            ylabel="Episode reward",
            note="No training_logs.jsonl found — run scripts/train_trl.py to populate.",
        )
    rewards = [float(e["reward"]) for e in rl_entries]
    episodes = [e.get("episode", e.get("step", i)) for i, e in enumerate(rl_entries)]
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, rewards, label="episode reward", linewidth=1.4, color="#2980b9")
    if len(rewards) >= 5:
        window = max(3, len(rewards) // 8)
        smoothed = [sum(rewards[max(0, i - window):i + 1]) / max(1, min(i + 1, window))
                    for i in range(len(rewards))]
        plt.plot(episodes, smoothed, label=f"moving avg (window={window})",
                 linewidth=2.0, color="#c0392b")
    plt.title("Reward Curve (live env rollouts)")
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_loss_curve(training_logs: List[Dict[str, Any]], out_path: str) -> bool:
    sft = [e for e in training_logs if e.get("phase") == "sft" and isinstance(e.get("loss"), (int, float))]
    rl = [e for e in training_logs if e.get("phase") == "rl" and isinstance(e.get("loss"), (int, float))]
    if not sft and not rl:
        # Try generic.
        rl = [e for e in training_logs if isinstance(e.get("loss"), (int, float))]
    if not sft and not rl:
        return _placeholder_plot(
            out_path, title="Loss Curve", xlabel="Update step", ylabel="Loss",
            note="No training_logs.jsonl found — run scripts/train_trl.py to populate.",
        )
    plt.figure(figsize=(7, 4))
    if sft:
        sft_steps = [e.get("epoch", e.get("step", i)) for i, e in enumerate(sft)]
        sft_losses = [float(e["loss"]) for e in sft]
        plt.plot(sft_steps, sft_losses, label="SFT loss (warm-start)",
                 color="#8e44ad", linewidth=1.6, marker="o")
    if rl:
        rl_x = [e.get("episode", i) + (max(sft_steps) if sft else 0) for i, e in enumerate(rl)]
        rl_losses = [float(e["loss"]) for e in rl]
        plt.plot(rl_x, rl_losses, label="RL policy loss",
                 color="#c0392b", linewidth=1.4)
    plt.title("Loss Curve (SFT warm-start + REINFORCE)")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _bar_compare(metrics: Dict[str, Dict[str, float]], key: str, title: str, ylabel: str, out_path: str, color: str = "#2980b9") -> bool:
    if not metrics:
        return _placeholder_plot(out_path, title=title, xlabel="Agent", ylabel=ylabel,
                                 note="No baseline_metrics.json found.")
    names = list(metrics.keys())
    values = [metrics[n].get(key, 0.0) for n in names]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, values, color=color)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_outage_comparison(metrics, out_path):
    return _bar_compare(metrics, "avg_outage_count", "Outage Count by Agent",
                        "Average outages per episode", out_path, color="#c0392b")


def plot_cascade_count_comparison(metrics, out_path):
    return _bar_compare(metrics, "avg_cascade_count", "Cascade Count by Agent",
                        "Average cascades per episode", out_path, color="#e67e22")


def plot_critical_job_completion(metrics, out_path):
    return _bar_compare(metrics, "critical_completion_rate", "Critical Job Completion by Agent",
                        "Critical completion rate", out_path, color="#27ae60")


def plot_guardrail_violations(metrics, out_path):
    return _bar_compare(metrics, "avg_guardrail_violations", "Guardrail Violations by Agent",
                        "Average violations per episode", out_path, color="#8e44ad")


def plot_chaos_survival(metrics, out_path):
    return _bar_compare(metrics, "chaos_survival", "Chaos Survival Score by Agent",
                        "reward × critical-completion (proxy)", out_path, color="#16a085")


def plot_cluster_health_curve(training_logs: List[Dict[str, Any]],
                              metrics: Dict[str, Dict[str, float]],
                              out_path: str) -> bool:
    """Combine an evaluation-time per-agent average health bar with a training
    "rolling cluster health" line if training_logs contain it.
    """

    rolling = [float(e["avg_cluster_health"]) for e in training_logs
               if isinstance(e.get("avg_cluster_health"), (int, float))]
    if rolling:
        steps = [e.get("episode", e.get("step", i)) for i, e in enumerate(training_logs)
                 if isinstance(e.get("avg_cluster_health"), (int, float))]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, rolling, color="#2c3e50", linewidth=1.6, label="rolling cluster health")
        plt.title("Cluster Health Over Training")
        plt.xlabel("Training step")
        plt.ylabel("Average cluster health")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True
    return _bar_compare(metrics, "avg_cluster_health", "Average Cluster Health by Agent",
                        "Average cluster health", out_path, color="#2c3e50")


def _placeholder_plot(out_path: str, title: str, xlabel: str, ylabel: str, note: str) -> bool:
    plt.figure(figsize=(7, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.5, 0.5, note, ha="center", va="center", transform=plt.gca().transAxes, fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=os.path.join(RESULTS, "baseline_metrics.json"))
    parser.add_argument("--evaluation", type=str, default=os.path.join(RESULTS, "evaluation_metrics.json"))
    parser.add_argument("--training-logs", type=str, default=os.path.join(RESULTS, "training_logs.jsonl"))
    parser.add_argument("--output-dir", type=str, default=RESULTS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = _load_json(args.baseline) or {}
    evaluation = _load_json(args.evaluation)
    training_logs = _load_jsonl(args.training_logs)

    # Prefer evaluation_metrics for comparison plots (it includes BaseLLM/SFT/RL),
    # fall back to baseline if no evaluation file.
    metrics_source = evaluation or baseline
    metrics_summary = _agent_summary(metrics_source) if metrics_source else {}

    # Always also pull baseline summary into a bar chart slot if eval is missing.
    if not metrics_summary and baseline:
        metrics_summary = _agent_summary(baseline)

    plots = [
        ("reward_curve.png", lambda p: plot_reward_curve(training_logs, p)),
        ("loss_curve.png", lambda p: plot_loss_curve(training_logs, p)),
        ("outage_comparison.png", lambda p: plot_outage_comparison(metrics_summary, p)),
        ("cascade_count_comparison.png", lambda p: plot_cascade_count_comparison(metrics_summary, p)),
        ("critical_job_completion.png", lambda p: plot_critical_job_completion(metrics_summary, p)),
        ("guardrail_violations.png", lambda p: plot_guardrail_violations(metrics_summary, p)),
        ("chaos_survival_score.png", lambda p: plot_chaos_survival(metrics_summary, p)),
        ("cluster_health_curve.png", lambda p: plot_cluster_health_curve(training_logs, metrics_summary, p)),
    ]
    for name, fn in plots:
        path = os.path.join(args.output_dir, name)
        ok = fn(path)
        print(f"{'wrote' if ok else 'skipped'}  {path}")


if __name__ == "__main__":
    main()
