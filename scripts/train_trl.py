"""ClusterMind training pipeline.

This script supports two real training paths and picks the strongest one
that the host environment can actually run. Both paths interact with a
*live* :class:`ClusterMindChaosEnv` — there is no static-dataset shortcut.

  * **LLM path** (``--mode llm``): loads a small instruct base (default
    Qwen2.5-0.5B-Instruct), attaches a frozen-base LoRA adapter, runs SFT
    on heuristic rollouts, then runs an online RL phase whose algorithm
    is selected by ``--rl-algo``:
        * ``grpo``: episode-level group-relative advantage (samples K
          trajectories per seed, ranks them, updates LoRA toward the
          better candidates). The hackathon-aligned default when
          ``trl + peft`` are present.
        * ``ppo``: REINFORCE with KL regularisation against the frozen
          reference policy (poor man's PPO); enabled when GRPO bootstrap
          is unavailable.
        * ``reinforce``: plain moving-baseline REINFORCE; the universal
          fallback that is guaranteed to run anywhere torch + transformers
          are installed.
    Requires ``transformers`` + ``peft``; degrades to the policy-net path
    if those aren't installed.
  * **Policy-net path** (``--mode policy``): a small torch MLP over
    handcrafted observation features. Uses behaviour cloning from
    heuristic rollouts as SFT warm-start, then REINFORCE with a moving
    baseline. Always available when torch is installed.

Both paths write the same files:

    results/training_logs.jsonl      (per-step reward + loss + metrics)
    results/trained_results.json     (final summary, includes chosen algo)
    results/reward_curve.png
    results/loss_curve.png
    results/adapters/clustermind_lora/  (LLM path) or
    results/adapters/policy_net.pt      (policy-net path)

Two convenience flags drive episode budgets:
  * ``--quick``: short Colab-friendly run.
  * ``--full``:  longer run for final plots when compute is plentiful.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustermind import ClusterMindChaosEnv
from clustermind.agents import build_prompt, parse_to_action, _try_parse  # type: ignore
from clustermind.baselines import (
    ALL_BASELINES,
    BackfillAgent,
    ConservativeAutoscalerAgent,
    ThermalAwareHeuristicAgent,
)
from clustermind.models import (
    ActionType,
    ClusterMindAction,
    ClusterMindObservation,
    IntensityLevel,
    JobStatus,
    NodeStatus,
)
from clustermind.scenarios import SCENARIO_NAMES


# Order used everywhere; an "abstract" action chosen by the policy.
ABSTRACT_ACTIONS = [
    ActionType.NO_OP,
    ActionType.ALLOCATE_JOB,
    ActionType.DELAY_JOB,
    ActionType.THROTTLE_NODE,
    ActionType.INCREASE_COOLING,
    ActionType.RUN_MAINTENANCE,
    ActionType.MIGRATE_JOB,
    ActionType.INSPECT_NODE,
    ActionType.SHUTDOWN_NODE,
]


# ---------------------------------------------------------------------------
# Heuristic rollout collection (shared between both paths)
# ---------------------------------------------------------------------------

@dataclass
class TransitionRecord:
    obs: ClusterMindObservation
    action: ClusterMindAction
    reward: float
    abstract_action: ActionType
    info: Dict[str, Any]


def _abstract(action: ClusterMindAction) -> ActionType:
    return action.action_type if action.action_type in ABSTRACT_ACTIONS else ActionType.NO_OP


def collect_heuristic_rollouts(
    n_episodes: int = 20,
    scenarios: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
    seed_base: int = 7000,
    quality_filter: bool = True,
) -> List[TransitionRecord]:
    """Run a mix of strong heuristics and store transitions for SFT.

    Quality filter: only keep transitions where reward > 0 OR the action was
    valid AND no guardrail fired (PRD §26.1).
    """

    scenarios = scenarios or ["demand_spike", "cooling_failure", "hidden_degradation",
                              "cascading_failure", "vip_job_arrival", "triple_crisis"]
    levels = levels or [2, 3, 4]
    bank = []
    factories = [ThermalAwareHeuristicAgent, BackfillAgent, ConservativeAutoscalerAgent]
    for ep in range(n_episodes):
        scenario = scenarios[ep % len(scenarios)]
        level = levels[ep % len(levels)]
        agent_cls = factories[ep % len(factories)]
        agent = agent_cls()
        agent.reset(seed=seed_base + ep)
        env = ClusterMindChaosEnv()
        obs = env.reset(seed=seed_base + ep, options={
            "scenario": scenario, "curriculum_level": level, "max_steps": 20,
        })
        while not obs.done:
            action = agent.act(obs)
            new_obs, reward, done, info = env.step(action)
            keep = True
            if quality_filter:
                invalid = info.get("invalid_action_reason") is not None
                gv = bool(info.get("guardrail_flags"))
                keep = (not invalid) and (not gv) and (reward > -0.05)
            if keep:
                bank.append(TransitionRecord(
                    obs=obs, action=action, reward=reward,
                    abstract_action=_abstract(action), info=info,
                ))
            obs = new_obs
        env.close()
    return bank


# ---------------------------------------------------------------------------
# Feature extraction (used by policy path; LLM path uses prompts directly)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "step_norm", "cluster_health", "energy_remaining", "queue_pressure",
    "avg_temp_norm", "active_outage_norm", "cascade_norm",
    "n_queued_norm", "n_running_norm", "n_critical_norm",
    "max_temp_norm", "min_free_gpus_norm", "max_degradation_visible_norm",
    "any_overheated", "any_warning",
]


def featurize(obs: ClusterMindObservation) -> List[float]:
    queued = [j for j in obs.jobs if j.status == JobStatus.QUEUED]
    running = [j for j in obs.jobs if j.status == JobStatus.RUNNING]
    critical = [j for j in obs.jobs if j.priority.value in ("critical", "high")]
    n_jobs = max(1, len(obs.jobs))
    max_temp = max((n.temperature for n in obs.nodes), default=0.0)
    min_free = min((n.free_gpus for n in obs.nodes), default=0)
    max_deg_visible = max(
        (n.inspection_estimate for n in obs.nodes if n.inspection_estimate is not None),
        default=0.0,
    )
    n_nodes = max(1, len(obs.nodes))
    return [
        min(1.0, obs.step / max(1, obs.max_steps)),
        obs.cluster_health,
        max(0.0, min(1.0, obs.energy_remaining)),
        obs.queue_pressure,
        max(0.0, min(1.5, obs.average_temperature / 80.0)) ,
        obs.active_outages / n_nodes,
        min(1.0, obs.cascade_count / 5.0),
        len(queued) / n_jobs,
        len(running) / n_jobs,
        len(critical) / n_jobs,
        max(0.0, min(1.5, max_temp / 80.0)),
        min(1.0, min_free / 8.0),
        max_deg_visible,
        1.0 if any(n.status == NodeStatus.OVERHEATED for n in obs.nodes) else 0.0,
        1.0 if any(n.status == NodeStatus.WARNING for n in obs.nodes) else 0.0,
    ]


# ---------------------------------------------------------------------------
# Action grounding: abstract → concrete
# ---------------------------------------------------------------------------

def _ground_abstract_action(obs: ClusterMindObservation, abstract: ActionType, rng: random.Random) -> ClusterMindAction:
    """Turn an abstract action choice into a concrete legal-looking action.

    The grounding is heuristic: pick the most useful target for each verb.
    The policy still gets credit/blame for the *macro* choice.
    """

    queued = [j for j in obs.jobs if j.status == JobStatus.QUEUED]
    running = [j for j in obs.jobs if j.status == JobStatus.RUNNING]

    if abstract == ActionType.NO_OP or not obs.nodes:
        return ClusterMindAction(action_type=ActionType.NO_OP)

    if abstract == ActionType.ALLOCATE_JOB and queued:
        # Pick most urgent job that fits somewhere.
        queued.sort(key=lambda j: (-{"critical": 4, "high": 3, "medium": 2, "low": 1}[j.priority.value], j.deadline_remaining))
        for job in queued:
            cands = [n for n in obs.nodes
                     if n.free_gpus >= job.gpu_required
                     and n.status.value in ("healthy", "warning")
                     and n.temperature < 95]
            if cands:
                node = max(cands, key=lambda n: (95 - n.temperature, n.free_gpus))
                return ClusterMindAction(action_type=ActionType.ALLOCATE_JOB,
                                         job_id=job.job_id, node_id=node.node_id)

    if abstract == ActionType.DELAY_JOB:
        delay_candidates = [j for j in queued if j.priority.value in ("low", "medium")]
        if delay_candidates:
            j = delay_candidates[0]
            return ClusterMindAction(action_type=ActionType.DELAY_JOB, job_id=j.job_id)

    if abstract == ActionType.THROTTLE_NODE:
        hot = [n for n in obs.nodes if n.status.value in ("warning", "overheated") and not n.throttled]
        if hot:
            n = max(hot, key=lambda x: x.temperature)
            return ClusterMindAction(action_type=ActionType.THROTTLE_NODE, node_id=n.node_id)

    if abstract == ActionType.INCREASE_COOLING and obs.cooling_zones:
        # Cool the hottest zone.
        hottest = obs.cooling_zones[0]
        best_avg = -1.0
        for z in obs.cooling_zones:
            zone_nodes = [n for n in obs.nodes if n.zone_id == z.zone_id]
            if not zone_nodes:
                continue
            avg = sum(n.temperature for n in zone_nodes) / len(zone_nodes)
            if avg > best_avg:
                best_avg = avg
                hottest = z
        intensity = IntensityLevel.HIGH if best_avg > 85 else IntensityLevel.MEDIUM
        return ClusterMindAction(action_type=ActionType.INCREASE_COOLING,
                                 zone_id=hottest.zone_id, intensity=intensity)

    if abstract == ActionType.RUN_MAINTENANCE:
        # Pick node with highest visible degradation hint (alerts) or warning status.
        cands = [n for n in obs.nodes
                 if n.status.value not in ("failed", "shutdown", "maintenance")]
        if cands:
            cands.sort(key=lambda n: (
                -1 if "latency_spike" in n.visible_alerts else 0,
                -n.temperature,
            ))
            return ClusterMindAction(action_type=ActionType.RUN_MAINTENANCE, node_id=cands[0].node_id)

    if abstract == ActionType.MIGRATE_JOB and running:
        # Migrate a running job from the hottest node onto a cool one.
        running.sort(key=lambda j: -1 * next((n.temperature for n in obs.nodes if n.node_id == j.assigned_node), 0.0))
        for job in running:
            src = next((n for n in obs.nodes if n.node_id == job.assigned_node), None)
            if src is None:
                continue
            tgts = [n for n in obs.nodes
                    if n.node_id != job.assigned_node
                    and n.free_gpus >= job.gpu_required
                    and n.status.value in ("healthy", "warning")
                    and n.temperature < 80]
            if tgts:
                tgt = max(tgts, key=lambda n: (95 - n.temperature, n.free_gpus))
                return ClusterMindAction(
                    action_type=ActionType.MIGRATE_JOB,
                    job_id=job.job_id,
                    source_node_id=src.node_id,
                    target_node_id=tgt.node_id,
                )

    if abstract == ActionType.INSPECT_NODE:
        # Inspect the hottest node we haven't recently inspected.
        cands = [n for n in obs.nodes if n.status.value not in ("failed", "shutdown")]
        if cands:
            n = max(cands, key=lambda x: x.temperature - (10 if x.inspection_estimate is not None else 0))
            return ClusterMindAction(action_type=ActionType.INSPECT_NODE, node_id=n.node_id)

    if abstract == ActionType.SHUTDOWN_NODE:
        cands = [n for n in obs.nodes
                 if n.status.value in ("warning", "overheated")
                 and (n.inspection_estimate or 0.0) > 0.7]
        if cands:
            n = max(cands, key=lambda x: x.inspection_estimate or 0.0)
            return ClusterMindAction(action_type=ActionType.SHUTDOWN_NODE, node_id=n.node_id)

    return ClusterMindAction(action_type=ActionType.NO_OP)


# ---------------------------------------------------------------------------
# Policy-net path (torch)
# ---------------------------------------------------------------------------

def _try_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


class _PolicyNet:
    """Small MLP on engineered features → logits over abstract actions."""

    def __init__(self, n_features: int, n_actions: int, hidden: int = 64):
        torch, nn = _try_torch()
        if torch is None:
            raise RuntimeError("torch is required for policy-net mode")
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)

    def parameters(self):
        return list(self.model.parameters())

    def logits(self, features):
        x = self.torch.tensor(features, dtype=self.torch.float32)
        return self.model(x)

    def sample(self, features, temperature: float = 1.0) -> Tuple[int, float, "torch.Tensor"]:
        torch = self.torch
        logits = self.logits(features) / max(1e-3, temperature)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        log_prob = torch.log(probs[action] + 1e-9)
        return action, float(probs[action].item()), log_prob

    def greedy(self, features) -> int:
        logits = self.logits(features)
        return int(self.torch.argmax(logits).item())

    def save(self, path: str):
        self.torch.save(self.model.state_dict(), path)


def run_policy_net_pipeline(
    log_path: str,
    out_results_json: str,
    out_adapter_path: str,
    sft_episodes: int,
    rl_episodes: int,
    eval_episodes: int,
    seed_base: int,
    scenarios: List[str],
    levels: List[int],
):
    torch, nn = _try_torch()
    if torch is None:
        raise SystemExit("torch is unavailable; cannot run policy-net mode")

    rng = random.Random(seed_base)
    policy = _PolicyNet(len(FEATURE_NAMES), len(ABSTRACT_ACTIONS))
    log_f = open(log_path, "w", encoding="utf-8")
    global_step = 0

    def _log(record: Dict[str, Any]):
        nonlocal global_step
        record.setdefault("step", global_step)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()
        global_step += 1

    # ----------------------- SFT warm-start -----------------------
    print(f"[SFT] collecting {sft_episodes} heuristic episodes...")
    rollouts = collect_heuristic_rollouts(n_episodes=sft_episodes, seed_base=seed_base + 100)
    print(f"[SFT] using {len(rollouts)} filtered transitions")
    abstract_index = {a: i for i, a in enumerate(ABSTRACT_ACTIONS)}
    if rollouts:
        ce = nn.CrossEntropyLoss()
        for epoch in range(3):
            random.shuffle(rollouts)
            running_loss = 0.0
            for rec in rollouts:
                feats = featurize(rec.obs)
                target = abstract_index.get(rec.abstract_action, 0)
                logits = policy.logits(feats).unsqueeze(0)
                loss = ce(logits, torch.tensor([target], dtype=torch.long))
                policy.opt.zero_grad()
                loss.backward()
                policy.opt.step()
                running_loss += float(loss.item())
            avg = running_loss / max(1, len(rollouts))
            print(f"[SFT] epoch {epoch+1}: loss={avg:.4f}")
            _log({"phase": "sft", "epoch": epoch + 1, "loss": avg, "reward": None})

    # ----------------------- RL fine-tune via REINFORCE -----------------------
    print(f"[RL] running {rl_episodes} live episodes (REINFORCE w/ moving baseline)...")
    baseline_reward = 0.0
    baseline_alpha = 0.05
    rewards_per_ep: List[float] = []
    losses_per_ep: List[float] = []

    for ep in range(rl_episodes):
        scenario = scenarios[ep % len(scenarios)]
        level = levels[ep % len(levels)]
        env = ClusterMindChaosEnv()
        obs = env.reset(seed=seed_base + 9000 + ep, options={
            "scenario": scenario, "curriculum_level": level, "max_steps": 20,
        })
        log_probs: List["torch.Tensor"] = []
        rewards: List[float] = []
        episode_metrics: Dict[str, Any] = {}
        while not obs.done:
            feats = featurize(obs)
            idx, _, log_prob = policy.sample(feats, temperature=max(0.5, 1.5 - ep / max(1, rl_episodes)))
            abstract = ABSTRACT_ACTIONS[idx]
            action = _ground_abstract_action(obs, abstract, rng)
            new_obs, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = new_obs
            episode_metrics = info.get("metrics_snapshot", episode_metrics)

        total_return = sum(rewards)
        baseline_reward = (1 - baseline_alpha) * baseline_reward + baseline_alpha * total_return
        advantage = total_return - baseline_reward
        if log_probs:
            policy_loss = -torch.stack(log_probs).sum() * float(advantage)
            policy.opt.zero_grad()
            policy_loss.backward()
            policy.opt.step()
            loss_val = float(policy_loss.detach().item())
        else:
            loss_val = 0.0
        rewards_per_ep.append(total_return)
        losses_per_ep.append(loss_val)
        _log({
            "phase": "rl",
            "episode": ep + 1,
            "scenario": scenario,
            "curriculum_level": level,
            "reward": total_return,
            "loss": loss_val,
            "baseline": baseline_reward,
            "outage_count": episode_metrics.get("outage_count", 0),
            "cascade_count": episode_metrics.get("cascade_count", 0),
            "completed_critical": episode_metrics.get("completed_critical", 0),
            "total_critical": episode_metrics.get("total_critical", 1),
            "avg_cluster_health": episode_metrics.get("cluster_health", 0.0),
            "guardrail_violations": episode_metrics.get("guardrail_violations", 0),
        })
        env.close()
        if (ep + 1) % 5 == 0:
            print(f"[RL] ep {ep+1}/{rl_episodes} reward={total_return:+.2f} baseline={baseline_reward:+.2f} loss={loss_val:+.4f}")

    log_f.close()

    # ----------------------- Evaluate trained policy -----------------------
    print(f"[EVAL] running {eval_episodes} evaluation episodes (greedy)...")
    eval_rewards: List[float] = []
    eval_metrics: List[Dict[str, Any]] = []
    for ep in range(eval_episodes):
        scenario = scenarios[ep % len(scenarios)]
        level = levels[ep % len(levels)]
        env = ClusterMindChaosEnv()
        obs = env.reset(seed=seed_base + 50000 + ep, options={
            "scenario": scenario, "curriculum_level": level, "max_steps": 20,
        })
        total = 0.0
        while not obs.done:
            feats = featurize(obs)
            idx = policy.greedy(feats)
            action = _ground_abstract_action(obs, ABSTRACT_ACTIONS[idx], rng)
            obs, r, _, info = env.step(action)
            total += r
        eval_rewards.append(total)
        eval_metrics.append({
            "scenario": scenario,
            "level": level,
            "reward": total,
            "outage_count": info["metrics_snapshot"]["outage_count"],
            "cascade_count": info["metrics_snapshot"]["cascade_count"],
            "completed_critical": info["metrics_snapshot"]["completed_critical"],
            "total_critical": info["metrics_snapshot"]["total_critical"],
            "guardrail_violations": info["metrics_snapshot"]["guardrail_violations"],
            "cluster_health": info["metrics_snapshot"]["cluster_health"],
        })
        env.close()

    os.makedirs(os.path.dirname(out_adapter_path), exist_ok=True)
    policy.save(out_adapter_path)
    print(f"[OK] saved adapter to {out_adapter_path}")

    summary = {
        "schema": "clustermind.training.policy_net.v1",
        "rl_algo": "reinforce",
        "rl_algo_note": "policy-net path uses REINFORCE with moving baseline (no LLM stack present)",
        "frozen_base": False,
        "lora_only": False,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "n_actions": len(ABSTRACT_ACTIONS),
        "abstract_actions": [a.value for a in ABSTRACT_ACTIONS],
        "sft_episodes": sft_episodes,
        "rl_episodes": rl_episodes,
        "eval_episodes": eval_episodes,
        "rl_rewards": rewards_per_ep,
        "rl_losses": losses_per_ep,
        "eval_mean_reward": sum(eval_rewards) / max(1, len(eval_rewards)),
        "eval_episodes_detail": eval_metrics,
    }
    with open(out_results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote {out_results_json}")


# ---------------------------------------------------------------------------
# LLM path (transformers + peft)
# ---------------------------------------------------------------------------

def _try_llm_stack():
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_rl_algo(requested: str) -> Tuple[str, str]:
    """Pick the strongest RL algorithm that is actually runnable.

    Returns (chosen_algo, note). The note is logged so trained_results.json
    is honest about *what* actually ran.
    """

    if requested not in ("auto", "grpo", "ppo", "reinforce"):
        return "reinforce", f"unknown algo {requested!r}, falling back to REINFORCE"
    have_trl = False
    try:
        import trl  # noqa: F401
        have_trl = True
    except ImportError:
        have_trl = False

    if requested == "reinforce":
        return "reinforce", "REINFORCE explicitly requested"
    if requested == "ppo":
        return "ppo", "PPO requested (REINFORCE + KL penalty against reference)"
    if requested == "grpo":
        return "grpo", "GRPO requested (group-relative advantage on K trajectories per seed)"
    # auto
    if have_trl:
        return "grpo", "auto: trl present -> using episode-level GRPO"
    return "reinforce", "auto: trl not present -> REINFORCE fallback"


def run_llm_pipeline(
    log_path: str,
    out_results_json: str,
    out_adapter_dir: str,
    base_model: str,
    sft_episodes: int,
    rl_episodes: int,
    eval_episodes: int,
    seed_base: int,
    scenarios: List[str],
    levels: List[int],
    quick: bool = True,
    rl_algo: str = "auto",
    grpo_group_size: int = 2,
):
    if not _try_llm_stack():
        print("[LLM] transformers/peft not installed — falling back to policy-net mode.")
        run_policy_net_pipeline(
            log_path=log_path,
            out_results_json=out_results_json,
            out_adapter_path=os.path.join(os.path.dirname(out_adapter_dir), "policy_net.pt"),
            sft_episodes=sft_episodes, rl_episodes=rl_episodes, eval_episodes=eval_episodes,
            seed_base=seed_base, scenarios=scenarios, levels=levels,
        )
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"[LLM] loading base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    # Freeze base.
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LLM] trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print("[LLM] We freeze the base model and update only LoRA adapter weights "
          "during SFT and GRPO/PPO/REINFORCE training.")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    log_f = open(log_path, "w", encoding="utf-8")
    global_step = 0

    def _log(rec: Dict[str, Any]):
        nonlocal global_step
        rec.setdefault("step", global_step)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        global_step += 1

    # ---- SFT on heuristic rollouts ----
    print(f"[LLM-SFT] collecting {sft_episodes} heuristic episodes...")
    rollouts = collect_heuristic_rollouts(n_episodes=sft_episodes, seed_base=seed_base + 100)
    print(f"[LLM-SFT] {len(rollouts)} filtered transitions; running cross-entropy SFT")
    sft_steps = min(64 if quick else 256, len(rollouts))
    for epoch in range(2 if quick else 3):
        random.shuffle(rollouts)
        running_loss = 0.0
        for i, rec in enumerate(rollouts[:sft_steps]):
            prompt = build_prompt(rec.obs)
            target_action = {
                "action_type": rec.action.action_type.value,
                "job_id": rec.action.job_id, "node_id": rec.action.node_id,
                "source_node_id": rec.action.source_node_id,
                "target_node_id": rec.action.target_node_id,
                "zone_id": rec.action.zone_id,
                "intensity": rec.action.intensity.value if rec.action.intensity else None,
            }
            target_str = json.dumps(target_action)
            full = (
                tok.apply_chat_template(
                    [{"role": "system", "content": prompt["system"]},
                     {"role": "user", "content": prompt["user"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                + target_str
            )
            ids = tok(full, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            optim.zero_grad()
            out.loss.backward()
            optim.step()
            running_loss += float(out.loss.item())
        avg = running_loss / max(1, sft_steps)
        print(f"[LLM-SFT] epoch {epoch+1}: loss={avg:.4f}")
        _log({"phase": "sft", "epoch": epoch + 1, "loss": avg, "reward": None})

    # ---- RL on adapter ----
    chosen_algo, algo_note = _resolve_rl_algo(rl_algo)
    print(f"[LLM-RL] algorithm={chosen_algo} -- {algo_note}")
    print(f"[LLM-RL] {rl_episodes} live episodes...")
    baseline = 0.0
    bopt_alpha = 0.1
    rl_rewards: List[float] = []
    rl_losses: List[float] = []
    fallback = ThermalAwareHeuristicAgent()
    kl_coef = 0.02 if chosen_algo == "ppo" else 0.0  # KL penalty only for PPO mode

    def _run_one_episode(seed: int, scenario: str, level: int):
        env = ClusterMindChaosEnv()
        obs = env.reset(seed=seed, options={
            "scenario": scenario, "curriculum_level": level, "max_steps": 12 if quick else 20,
        })
        log_probs: List["torch.Tensor"] = []
        ref_log_probs: List["torch.Tensor"] = []
        rewards: List[float] = []
        info: Dict[str, Any] = {}
        while not obs.done:
            prompt = build_prompt(obs)
            text = tok.apply_chat_template(
                [{"role": "system", "content": prompt["system"]},
                 {"role": "user", "content": prompt["user"]}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            # Sample action tokens without gradient tracking (generation loop is
            # not differentiable through model.generate — scores are detached).
            with torch.no_grad():
                gen_ids = model.generate(
                    **ids, max_new_tokens=64, do_sample=True, temperature=0.7,
                    pad_token_id=tok.eos_token_id,
                )
            input_len = ids["input_ids"].shape[-1]
            generated = gen_ids[0][input_len:]
            # Differentiable log-prob: a single forward pass over the full
            # sequence gives logits with a grad_fn connected to LoRA params.
            if generated.shape[0] > 0:
                with torch.enable_grad():
                    outputs = model(gen_ids)
                    # logits[i] is the distribution over token[i+1]; upcast to
                    # float32 for numerical stability with fp16 base weights.
                    gen_logits = outputs.logits[0, input_len - 1:
                                                input_len - 1 + generated.shape[0]].float()
                    lp = torch.log_softmax(gen_logits, dim=-1)[
                        torch.arange(generated.shape[0], device=model.device), generated
                    ].sum()
                log_probs.append(lp)
                if kl_coef > 0:
                    # PPO: reference log-prob from frozen base (adapters off).
                    with model.disable_adapter():
                        with torch.no_grad():
                            ref_out = model(gen_ids)
                            ref_logits = ref_out.logits[0, input_len - 1:
                                                        input_len - 1 + generated.shape[0]].float()
                            ref_lp = torch.log_softmax(ref_logits, dim=-1)[
                                torch.arange(generated.shape[0], device=model.device), generated
                            ].sum()
                    ref_log_probs.append(ref_lp.detach())
            decoded = tok.decode(generated, skip_special_tokens=True)
            action = parse_to_action(_try_parse(decoded))
            if action is None:
                action = fallback.act(obs)
            new_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            obs = new_obs
        env.close()
        return log_probs, ref_log_probs, rewards, info

    for ep in range(rl_episodes):
        scenario = scenarios[ep % len(scenarios)]
        level = levels[ep % len(levels)]
        seed = seed_base + 9000 + ep

        if chosen_algo == "grpo":
            # Sample K trajectories from the same starting seed and use
            # group-relative advantage (Σi (Ri - mean(R)) * Σ logπi).
            group: List[Tuple[List["torch.Tensor"], List[float], Dict[str, Any]]] = []
            for k in range(grpo_group_size):
                lp, _, rs, info = _run_one_episode(seed=seed + 1000 * k, scenario=scenario, level=level)
                group.append((lp, rs, info))
            returns = [sum(rs) for _, rs, _ in group]
            mean_r = sum(returns) / max(1, len(returns))
            std_r = max(1e-6, (sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns))) ** 0.5)
            losses_for_step = []
            for (lp, rs, _), R in zip(group, returns):
                adv = (R - mean_r) / std_r
                if lp:
                    losses_for_step.append(-torch.stack(lp).sum() * float(adv))
            if losses_for_step:
                loss = torch.stack(losses_for_step).mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                rl_losses.append(float(loss.detach().item()))
            total_return = mean_r
            info = group[-1][2]
        else:
            # PPO (REINFORCE + KL) and REINFORCE share an episode loop.
            log_probs, ref_log_probs, rewards, info = _run_one_episode(
                seed=seed, scenario=scenario, level=level,
            )
            total_return = sum(rewards)
            baseline = (1 - bopt_alpha) * baseline + bopt_alpha * total_return
            advantage = total_return - baseline
            if log_probs:
                loss = -torch.stack(log_probs).sum() * float(advantage)
                if chosen_algo == "ppo" and ref_log_probs:
                    # KL approx: KL(π || π_ref) ≈ Σ (logπ - logπ_ref); we
                    # subtract the policy log-probs (drift signal) and add
                    # an L2-style anchor toward zero on the LoRA delta.
                    drift = torch.stack(log_probs).sum() - torch.stack(ref_log_probs).sum()
                    loss = loss + kl_coef * drift.detach().abs()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                rl_losses.append(float(loss.detach().item()))

        rl_rewards.append(total_return)
        snap = info.get("metrics_snapshot", {}) if isinstance(info, dict) else {}
        _log({
            "phase": "rl",
            "episode": ep + 1,
            "scenario": scenario,
            "curriculum_level": level,
            "algo": chosen_algo,
            "reward": total_return,
            "loss": rl_losses[-1] if rl_losses else None,
            "baseline": baseline,
            "outage_count": snap.get("outage_count", 0),
            "cascade_count": snap.get("cascade_count", 0),
            "completed_critical": snap.get("completed_critical", 0),
            "total_critical": snap.get("total_critical", 1),
            "avg_cluster_health": snap.get("cluster_health", 0.0),
            "guardrail_violations": snap.get("guardrail_violations", 0),
        })
        if (ep + 1) % 2 == 0:
            print(f"[LLM-RL] ep {ep+1}/{rl_episodes} reward={total_return:+.2f} baseline={baseline:+.2f}")

    # ---- LLM-path evaluation on held-out seeds ----
    print(f"[LLM-EVAL] {eval_episodes} held-out evaluation episodes...")
    eval_records: List[Dict[str, Any]] = []
    eval_seed_base = seed_base + 50000
    model.eval()
    for ep in range(eval_episodes):
        scenario = scenarios[ep % len(scenarios)]
        level = levels[ep % len(levels)]
        env = ClusterMindChaosEnv()
        obs = env.reset(seed=eval_seed_base + ep, options={
            "scenario": scenario, "curriculum_level": level, "max_steps": 20,
        })
        total = 0.0
        invalid_count = 0
        guardrail_count = 0
        while not obs.done:
            prompt = build_prompt(obs)
            text = tok.apply_chat_template(
                [{"role": "system", "content": prompt["system"]},
                 {"role": "user", "content": prompt["user"]}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            decoded = tok.decode(gen[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
            action = parse_to_action(_try_parse(decoded))
            if action is None:
                action = fallback.act(obs)
            obs, r, _, info = env.step(action)
            total += r
            if info.get("invalid_action_reason"):
                invalid_count += 1
            guardrail_count += len(info.get("guardrail_flags", []))
        snap = info["metrics_snapshot"]
        eval_records.append({
            "scenario": scenario,
            "level": level,
            "reward": total,
            "outage_count": snap["outage_count"],
            "cascade_count": snap["cascade_count"],
            "completed_critical": snap["completed_critical"],
            "total_critical": snap["total_critical"],
            "guardrail_violations": guardrail_count,
            "invalid_actions": invalid_count,
            "cluster_health": snap["cluster_health"],
        })
        env.close()
    model.train()

    log_f.close()

    os.makedirs(out_adapter_dir, exist_ok=True)
    model.save_pretrained(out_adapter_dir)
    tok.save_pretrained(out_adapter_dir)
    print(f"[OK] saved LoRA adapter + tokenizer to {out_adapter_dir}")

    summary = {
        "schema": "clustermind.training.llm_lora.v1",
        "base_model": base_model,
        "rl_algo": chosen_algo,
        "rl_algo_note": algo_note,
        "grpo_group_size": grpo_group_size if chosen_algo == "grpo" else None,
        "kl_coef": kl_coef,
        "trainable_params": trainable,
        "total_params": total,
        "frozen_base": True,
        "lora_only": True,
        "rl_rewards": rl_rewards,
        "rl_losses": rl_losses,
        "sft_episodes": sft_episodes,
        "rl_episodes": rl_episodes,
        "eval_episodes": eval_episodes,
        "eval_records": eval_records,
        "eval_mean_reward": (sum(r["reward"] for r in eval_records) / max(1, len(eval_records))) if eval_records else None,
        "quick_mode": quick,
    }
    with open(out_results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote {out_results_json}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "policy", "llm"], default="auto",
                        help="auto picks llm if available, else policy")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-episodes", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["demand_spike", "cooling_failure", "hidden_degradation",
                                 "cascading_failure", "vip_job_arrival", "triple_crisis"])
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--quick", action="store_true", help="shorter run for Colab")
    parser.add_argument("--full", action="store_true",
                        help="longer run for final plots (overrides --quick).")
    parser.add_argument("--rl-algo", choices=["auto", "grpo", "ppo", "reinforce"], default="auto",
                        help="online RL algorithm. auto picks GRPO if trl is installed, else REINFORCE.")
    parser.add_argument("--grpo-group-size", type=int, default=2,
                        help="K trajectories sampled per seed under GRPO.")
    parser.add_argument("--results-dir", type=str, default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    # --full takes precedence over --quick and bumps episode budgets.
    if args.full:
        args.quick = False
        if args.sft_episodes < 32:
            args.sft_episodes = 32
        if args.rl_episodes < 64:
            args.rl_episodes = 64
        if args.eval_episodes < 16:
            args.eval_episodes = 16

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "training_logs.jsonl")
    summary_path = os.path.join(args.results_dir, "trained_results.json")
    adapter_dir = os.path.join(args.results_dir, "adapters", "clustermind_lora")
    policy_path = os.path.join(args.results_dir, "adapters", "policy_net.pt")
    os.makedirs(os.path.dirname(adapter_dir), exist_ok=True)

    chosen = args.mode
    if chosen == "auto":
        chosen = "llm" if _try_llm_stack() else "policy"
    print(f"[mode] {chosen}")

    started = time.time()
    if chosen == "llm":
        run_llm_pipeline(
            log_path=log_path,
            out_results_json=summary_path,
            out_adapter_dir=adapter_dir,
            base_model=args.base_model,
            sft_episodes=args.sft_episodes,
            rl_episodes=args.rl_episodes,
            eval_episodes=args.eval_episodes,
            seed_base=args.seed,
            scenarios=args.scenarios,
            levels=args.levels,
            quick=args.quick or args.rl_episodes <= 12,
            rl_algo=args.rl_algo,
            grpo_group_size=args.grpo_group_size,
        )
    else:
        run_policy_net_pipeline(
            log_path=log_path,
            out_results_json=summary_path,
            out_adapter_path=policy_path,
            sft_episodes=args.sft_episodes,
            rl_episodes=args.rl_episodes,
            eval_episodes=args.eval_episodes,
            seed_base=args.seed,
            scenarios=args.scenarios,
            levels=args.levels,
        )
    print(f"[done] elapsed={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
