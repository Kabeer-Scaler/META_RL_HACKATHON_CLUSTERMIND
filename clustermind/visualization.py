"""Rendering helpers for the Gradio dashboard.

The functions here take a live :class:`ClusterMindObservation` plus optional
metrics and return matplotlib Figures + structured panels. They are kept
simulator-agnostic so the same helpers can render baseline rollouts, trained
LLM rollouts, or recorded replay frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from clustermind.models import (
    ClusterMindObservation,
    NodeStatus,
    NodeView,
    ZoneView,
)


# Visual encoding per PRD §25.
STATUS_COLOURS = {
    NodeStatus.HEALTHY: "#27ae60",
    NodeStatus.WARNING: "#f1c40f",
    NodeStatus.OVERHEATED: "#e67e22",
    NodeStatus.FAILED: "#c0392b",
    NodeStatus.MAINTENANCE: "#3498db",
    NodeStatus.SHUTDOWN: "#7f8c8d",
}


def _zone_layout(nodes: List[NodeView], zones: List[ZoneView]) -> Dict[str, Tuple[float, float]]:
    layout: Dict[str, Tuple[float, float]] = {}
    zone_index = {z.zone_id: i for i, z in enumerate(zones)}
    n_zones = max(1, len(zones))
    grouped: Dict[str, List[NodeView]] = {z.zone_id: [] for z in zones}
    for n in nodes:
        grouped.setdefault(n.zone_id, []).append(n)
    for zid, group in grouped.items():
        col = zone_index.get(zid, 0)
        for j, node in enumerate(group):
            x = col * 4.5 + (j % 5) * 0.9 + 0.4
            y = 1.0 + (j // 5) * 1.2
            layout[node.node_id] = (x, y)
    return layout


def render_cluster_graph(obs: ClusterMindObservation, title: str = "Cluster") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(-0.5, 4.5 * max(1, len(obs.cooling_zones)) + 0.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis("off")
    ax.set_title(f"{title}  (step {obs.step}/{obs.max_steps}, scenario={obs.scenario}, L{obs.curriculum_level})")

    layout = _zone_layout(obs.nodes, obs.cooling_zones)

    # Draw zone backgrounds + cooling indicators.
    for zone_idx, zone in enumerate(obs.cooling_zones):
        x = zone_idx * 4.5
        rect = patches.Rectangle((x, 0.2), 4.0, 3.4,
                                 facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=1.2)
        ax.add_patch(rect)
        cool_label = f"{zone.zone_id}\ncool×{zone.cooling_efficiency:.2f}\nintensity={zone.intensity.value}"
        ax.text(x + 2.0, 0.05, cool_label, ha="center", va="top", fontsize=8, color="#34495e")

    # Draw nodes.
    for node in obs.nodes:
        x, y = layout.get(node.node_id, (0.0, 0.0))
        colour = STATUS_COLOURS.get(node.status, "#7f8c8d")
        size = 0.25 + 0.35 * node.utilization
        circle = patches.Circle((x, y), size, facecolor=colour, edgecolor="#2c3e50", linewidth=1.0, alpha=0.85)
        ax.add_patch(circle)
        # Heat halo when temperature is high.
        if node.temperature >= 80:
            halo = patches.Circle((x, y), size + 0.08, facecolor="none",
                                  edgecolor="#e74c3c", linewidth=1.5,
                                  linestyle=":")
            ax.add_patch(halo)
        ax.text(x, y - size - 0.18, f"{node.node_id}\n{node.temperature:.0f}°C",
                ha="center", va="top", fontsize=7.5, color="#2c3e50")

    # Legend.
    legend_y = 3.7
    for i, (status, colour) in enumerate(STATUS_COLOURS.items()):
        cx = 0.4 + i * 0.9
        ax.add_patch(patches.Circle((cx, legend_y), 0.10, facecolor=colour, edgecolor="#2c3e50"))
        ax.text(cx, legend_y - 0.2, status.value, ha="center", va="top", fontsize=7)

    return fig


def render_metrics_panel(obs: ClusterMindObservation, last_reward: Optional[float], reward_breakdown: Optional[Dict[str, Any]]) -> str:
    rb = reward_breakdown or {}
    lines = [
        f"Step:           {obs.step}/{obs.max_steps}",
        f"Scenario:       {obs.scenario} (L{obs.curriculum_level})",
        f"Cluster health: {obs.cluster_health:.3f}",
        f"Energy left:    {obs.energy_remaining:.3f}",
        f"Avg temperature: {obs.average_temperature:.1f}°C",
        f"Active outages:  {obs.active_outages}",
        f"Cascade count:   {obs.cascade_count}",
        f"Queue pressure:  {obs.queue_pressure:.2f}",
        "",
        f"Last reward:     {last_reward if last_reward is None else f'{last_reward:+.3f}'}",
    ]
    if rb:
        lines.append("Reward breakdown (last step):")
        for key in [
            "critical_job_completion", "normal_job_completion", "deadline_score",
            "cluster_health_score", "thermal_safety_score", "recovery_score",
            "energy_efficiency_score", "useful_inspection_or_maintenance",
            "outage_penalty", "cascade_penalty", "missed_critical_deadline_penalty",
            "guardrail_violation_penalty", "invalid_action_penalty", "no_progress_penalty",
            "total",
        ]:
            if key in rb:
                lines.append(f"  {key:34s}: {rb[key]:+.3f}")
    return "\n".join(lines)


def render_jobs_table(obs: ClusterMindObservation, limit: int = 10) -> List[List[Any]]:
    rows = []
    for job in obs.jobs[:limit]:
        rows.append([
            job.job_id,
            job.priority.value,
            job.status.value,
            job.gpu_required,
            job.deadline_remaining,
            f"{job.progress_pct*100:.0f}%",
            job.assigned_node or "-",
            job.reward_value,
        ])
    return rows


def render_alerts(obs: ClusterMindObservation) -> str:
    parts = []
    if obs.alerts:
        parts.append("Alerts:\n  " + "\n  ".join(obs.alerts))
    if obs.guardrail_warnings:
        parts.append("Guardrails:\n  " + "\n  ".join(obs.guardrail_warnings))
    if not parts:
        return "All quiet."
    return "\n\n".join(parts)


def render_event_log(events: List[str], chaos_event: Optional[Dict[str, Any]]) -> str:
    out = []
    if chaos_event:
        out.append(f"[CHAOS] {chaos_event.get('action')} → {chaos_event.get('detail', '')}")
    if events:
        out.extend(f"• {e}" for e in events)
    return "\n".join(out) if out else "No events this step."
