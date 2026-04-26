"""Hugging Face Space / Gradio entry point.

A simulator dashboard that wraps a real :class:`ClusterMindChaosEnv`. Every
button click triggers a real ``env.reset()`` or ``env.step()`` — there is no
canned demo data anywhere in this file.

Tabs:
    1. Live Simulation — pick a scenario/agent, step or play through it.
    2. Baseline Comparison — run all baselines for one scenario and rank them.
    3. Flight Recorder Replay — read a saved replay JSON and render frame N.

The visuals come from ``clustermind.visualization`` so the same layout serves
both this Space and any local notebook usage.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from clustermind import ClusterMindChaosEnv
from clustermind.baselines import ALL_BASELINES, BaselineAgent, RandomAgent
from clustermind.models import (
    ActionType,
    ClusterMindAction,
    IntensityLevel,
    JobStatus,
    NodeStatus,
)
from clustermind.recorder import FlightRecorder
from clustermind.scenarios import SCENARIO_NAMES
from clustermind.visualization import (
    render_alerts,
    render_cluster_graph,
    render_event_log,
    render_jobs_table,
    render_metrics_panel,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Live session
# ---------------------------------------------------------------------------

class LiveSession:
    def __init__(self):
        self.env: Optional[ClusterMindChaosEnv] = None
        self.agent: Optional[BaselineAgent] = None
        self.last_obs = None
        self.last_reward: Optional[float] = None
        self.last_breakdown: Optional[Dict[str, Any]] = None
        self.last_event_log: List[str] = []
        self.last_chaos_event: Optional[Dict[str, Any]] = None
        self.cumulative_reward: float = 0.0

    def start(self, agent_name: str, scenario: str, curriculum: int, seed: int, max_steps: int) -> Tuple[Any, str, str, str, str, List[List[Any]]]:
        self.env = ClusterMindChaosEnv()
        if agent_name == "Manual":
            self.agent = None
        else:
            cls = ALL_BASELINES[agent_name]
            self.agent = cls(seed=seed) if cls is RandomAgent else cls()
            self.agent.reset(seed=seed)
        self.last_obs = self.env.reset(seed=seed, options={
            "scenario": scenario, "curriculum_level": curriculum, "max_steps": max_steps,
        })
        self.last_reward = None
        self.last_breakdown = None
        self.last_event_log = []
        self.last_chaos_event = None
        self.cumulative_reward = 0.0
        return self._render()

    def step_once(self, action_override: Optional[ClusterMindAction] = None) -> Tuple[Any, str, str, str, str, List[List[Any]]]:
        if self.env is None or self.last_obs is None or self.last_obs.done:
            return self._render()
        if action_override is not None:
            action = action_override
        elif self.agent is not None:
            action = self.agent.act(self.last_obs)
        else:
            action = ClusterMindAction(action_type=ActionType.NO_OP)
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        self.last_reward = reward
        self.last_breakdown = info.get("reward_breakdown")
        self.last_event_log = info.get("event_log", [])
        self.last_chaos_event = info.get("chaos_event")
        self.cumulative_reward += reward
        return self._render()

    def step_n(self, n: int):
        for _ in range(n):
            self.step_once()
            if self.last_obs is None or self.last_obs.done:
                break
        return self._render()

    def _render(self):
        if self.env is None or self.last_obs is None:
            return None, "Press Reset to start.", "", "", "", []
        fig = render_cluster_graph(self.last_obs, title=f"ClusterMind — cumulative reward {self.cumulative_reward:+.2f}")
        metrics = render_metrics_panel(self.last_obs, self.last_reward, self.last_breakdown)
        alerts = render_alerts(self.last_obs)
        events = render_event_log(self.last_event_log, self.last_chaos_event)
        rows = render_jobs_table(self.last_obs)
        return fig, metrics, alerts, events, self._failure_chain(), rows

    def _failure_chain(self) -> str:
        if self.env is None:
            return ""
        return self.env.recorder.explain_failure_chain()


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def run_baseline_comparison(scenario: str, curriculum: int, seed: int, max_steps: int) -> List[List[Any]]:
    rows = []
    for name, cls in ALL_BASELINES.items():
        env = ClusterMindChaosEnv()
        agent = cls(seed=seed) if cls is RandomAgent else cls()
        agent.reset(seed=seed)
        obs = env.reset(seed=seed, options={
            "scenario": scenario, "curriculum_level": curriculum, "max_steps": max_steps,
        })
        total = 0.0
        invalid = 0
        gv = 0
        while not obs.done:
            action = agent.act(obs)
            obs, r, done, info = env.step(action)
            total += r
            if info.get("invalid_action_reason"):
                invalid += 1
            gv += len(info.get("guardrail_flags", []))
        snap = info["metrics_snapshot"]
        rows.append([
            name,
            f"{total:+.2f}",
            snap["outage_count"],
            snap["cascade_count"],
            f"{snap['completed_critical']}/{snap['total_critical']}",
            f"{snap['cluster_health']:.2f}",
            invalid,
            gv,
        ])
        env.close()
    return rows


# ---------------------------------------------------------------------------
# Flight Recorder replay
# ---------------------------------------------------------------------------

def list_replay_files() -> List[str]:
    folder = os.path.join(ROOT, "results", "replays")
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".json"))


def load_replay(file_name: str) -> Tuple[str, str]:
    folder = os.path.join(ROOT, "results", "replays")
    path = os.path.join(folder, file_name)
    if not file_name or not os.path.isfile(path):
        return "Pick a replay file.", ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {})
    summary_text = (
        f"episode_id: {data.get('episode_id')}\n"
        f"records: {len(data.get('records', []))}\n"
        f"failure_steps: {summary.get('failure_steps')}\n"
        f"cascade_steps: {summary.get('cascade_steps')}\n"
        f"guardrail_steps: {summary.get('guardrail_steps')}\n"
        f"deadline_miss_steps: {summary.get('deadline_miss_steps')}\n"
        f"energy_exhaustion_step: {summary.get('energy_exhaustion_step')}\n"
    )
    # Build a narrative directly from the records (the recorder helper would
    # need the live recorder object; we replay the storyboard from JSON).
    lines = []
    for rec in data.get("records", []):
        primary = rec.get("primary_action", {}).get("action_type", "?")
        chaos = rec.get("chaos_action") or "-"
        events = "; ".join(rec.get("event_log", [])) or "no events"
        lines.append(f"Step {rec.get('step', 0):>2}: agent={primary}  chaos={chaos}  events={events}")
    return summary_text, "\n".join(lines)


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    session = LiveSession()
    agent_choices = ["Manual"] + list(ALL_BASELINES.keys())

    with gr.Blocks(title="ClusterMind Chaos Arena") as demo:
        gr.Markdown(
            "# ClusterMind Chaos Arena\n"
            "Guarded adversarial OpenEnv benchmark for AI infrastructure control. "
            "All visuals are produced from real `env.reset()` / `env.step()` calls."
        )
        with gr.Tabs():
            # --------------------- Live tab ---------------------
            with gr.Tab("Live Simulation"):
                with gr.Row():
                    scenario_in = gr.Dropdown(SCENARIO_NAMES, value="triple_crisis", label="Scenario")
                    curriculum_in = gr.Slider(1, 5, value=4, step=1, label="Curriculum level")
                    agent_in = gr.Dropdown(agent_choices, value="ThermalAwareHeuristicAgent", label="Agent")
                    seed_in = gr.Number(value=7, label="Seed", precision=0)
                    max_steps_in = gr.Slider(8, 30, value=20, step=1, label="Max steps")
                with gr.Row():
                    reset_btn = gr.Button("Reset / Start", variant="primary")
                    step_btn = gr.Button("Step")
                    play_btn = gr.Button("Run to end")
                fig_out = gr.Plot(label="Cluster")
                with gr.Row():
                    metrics_out = gr.Code(label="Metrics + reward breakdown", language=None)
                    alerts_out = gr.Code(label="Alerts & guardrails", language=None)
                events_out = gr.Code(label="Event log", language=None)
                chain_out = gr.Code(label="Flight Recorder narrative", language=None)
                jobs_out = gr.Dataframe(
                    headers=["job_id", "priority", "status", "gpus", "deadline", "progress", "node", "value"],
                    label="Jobs",
                )

                def _start(agent_name, scenario, curriculum, seed, max_steps):
                    return session.start(agent_name, scenario, int(curriculum), int(seed), int(max_steps))

                def _step():
                    return session.step_once()

                def _play():
                    return session.step_n(40)

                reset_btn.click(_start,
                                inputs=[agent_in, scenario_in, curriculum_in, seed_in, max_steps_in],
                                outputs=[fig_out, metrics_out, alerts_out, events_out, chain_out, jobs_out])
                step_btn.click(_step, inputs=None,
                               outputs=[fig_out, metrics_out, alerts_out, events_out, chain_out, jobs_out])
                play_btn.click(_play, inputs=None,
                               outputs=[fig_out, metrics_out, alerts_out, events_out, chain_out, jobs_out])

            # --------------------- Comparison tab ---------------------
            with gr.Tab("Baseline Comparison"):
                with gr.Row():
                    cmp_scenario = gr.Dropdown(SCENARIO_NAMES, value="triple_crisis", label="Scenario")
                    cmp_curriculum = gr.Slider(1, 5, value=4, step=1, label="Curriculum level")
                    cmp_seed = gr.Number(value=7, label="Seed", precision=0)
                    cmp_steps = gr.Slider(8, 30, value=20, step=1, label="Max steps")
                cmp_btn = gr.Button("Run all baselines", variant="primary")
                cmp_table = gr.Dataframe(
                    headers=["agent", "reward", "outages", "cascades", "critical_done", "health", "invalid", "guardrails"],
                    label="Per-agent metrics on this scenario",
                )
                cmp_btn.click(
                    lambda s, c, se, ms: run_baseline_comparison(s, int(c), int(se), int(ms)),
                    inputs=[cmp_scenario, cmp_curriculum, cmp_seed, cmp_steps],
                    outputs=cmp_table,
                )

            # --------------------- Replay tab ---------------------
            with gr.Tab("Flight Recorder Replay"):
                replay_files = list_replay_files()
                replay_dropdown = gr.Dropdown(
                    replay_files, value=(replay_files[0] if replay_files else None), label="Replay file"
                )
                refresh_btn = gr.Button("Refresh list")
                load_btn = gr.Button("Load")
                replay_summary = gr.Code(label="Summary", language=None)
                replay_text = gr.Code(label="Narrative", language=None)

                refresh_btn.click(lambda: gr.update(choices=list_replay_files()), outputs=replay_dropdown)
                load_btn.click(load_replay, inputs=replay_dropdown, outputs=[replay_summary, replay_text])

            # --------------------- Results tab ---------------------
            with gr.Tab("Training Results"):
                gr.Markdown("Plots are produced by `scripts/generate_plots.py` from real logs.")
                with gr.Row():
                    for name in [
                        "reward_curve.png", "loss_curve.png",
                        "outage_comparison.png", "cascade_count_comparison.png",
                    ]:
                        path = os.path.join(ROOT, "results", name)
                        if os.path.isfile(path):
                            gr.Image(value=path, label=name, height=240)
                with gr.Row():
                    for name in [
                        "critical_job_completion.png", "guardrail_violations.png",
                        "chaos_survival_score.png", "cluster_health_curve.png",
                    ]:
                        path = os.path.join(ROOT, "results", name)
                        if os.path.isfile(path):
                            gr.Image(value=path, label=name, height=240)

    return demo


def main():
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
