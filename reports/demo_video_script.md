# ClusterMind Chaos Arena — 2-minute demo script

**Target length:** 110–120 seconds. Read out loud at ~145 wpm.
**Tone:** technical but accessible. The judge has 60 seconds before they tab away.

---

## 0:00 – 0:15 — Hook (15 s)

> "What happens when an AI agent greedily manages a fragile AI cluster?
> ClusterMind Chaos Arena is an OpenEnv RL benchmark where an LLM has to keep
> a partially-observable compute cluster alive — through cooling failures,
> hidden hardware degradation, cascading outages, and an adversarial chaos agent."

**On screen:** title card → cluster-graph thumbnail → 8-plot collage (≤2 s).

---

## 0:15 – 0:35 — Random baseline (20 s)

> "A random policy bleeds the energy budget, shuts down healthy nodes, inspects
> nodes that already failed, and trips eight reward-hacking guardrails per
> episode. Critical-job completion: twenty-two percent."

**On screen:** Live Simulation tab in Gradio. Scenario `cascading_failure`,
agent `RandomAgent`, seed 106. Click *Run to end*. Show:
- the cluster turning red,
- the alerts panel filling up,
- final metrics (1/4 critical jobs done, health 0.10).

---

## 0:35 – 1:00 — Greedy collapse on triple_crisis (25 s)

> "Greedy throughput looks great until it doesn't. On the triple-crisis scenario,
> it allocates straight into a hot zone with hidden degradation — and the
> Flight Recorder shows the cascade chain step by step:
> node fails, neighbours absorb load, linked failure within three steps,
> two more nodes go down. By the end, the cluster is at forty-percent health
> even though most critical jobs technically completed."

**On screen:** Live Simulation, scenario `triple_crisis`, level 4, agent
`GreedyThroughputAgent`, seed 7. Run to end. Then the Flight Recorder narrative
panel — read out:
```
Step 1: ALLOCATE_JOB → gpu_2; node_failed:gpu_0
Step 2: ALLOCATE_JOB → gpu_7; node_failed:gpu_3 (linked cascade)
```

---

## 1:00 – 1:25 — ThermalAware survives chaos_arena (25 s)

> "Now ThermalAware on chaos_arena. The chaos agent injects bounded faults —
> demand spikes, cooling drops, latency alerts — but the policy fires
> INCREASE_COOLING when the hottest node in a zone crosses seventy-eight
> degrees. Reward: eleven-point-seven, two-and-a-half points above Greedy.
> This is the headline benchmark finding: thermal management actually wins
> under perturbation."

**On screen:** Switch to scenario `chaos_arena`, level 5, agent
`ThermalAwareHeuristicAgent`, same seed. Run to end. Then switch to
*Baseline Comparison* tab and show all 5 agents on `chaos_arena`.

---

## 1:25 – 1:45 — Training (20 s)

> "Training is real and runs on the live environment. We freeze the base model
> — Qwen-two-point-five-zero-point-five-billion-instruct — and update only
> a LoRA adapter. SFT warm-starts from heuristic rollouts; then GRPO samples
> two trajectories per seed and updates toward the better one. The reward
> curve and loss curve come straight from the training logs — no fake plots,
> no scripted success."

**On screen:** Training Results tab. Show `reward_curve.png` and
`loss_curve.png` side by side. Then a Colab terminal scrolling through:
```
[LLM-RL] algorithm=grpo
[LLM-RL] ep 24/24 reward=+10.5 baseline=+7.8
```

---

## 1:45 – 2:00 — Close (15 s)

> "ClusterMind Chaos Arena: nine action verbs, twelve guardrails, eight
> scenarios, frozen-base LoRA training, full Flight-Recorder observability.
> Open-source, OpenEnv-compliant, ready for Colab today.
> Code, plots, and notebook in the README."

**On screen:** README hero shot showing the 8 plots collage and the HF Space
URL banner.

---

## Fallback line (use if total length is under 105 s)

> "Honest disclosure: the local CPU run produces real reward curves but is the
> policy-net plumbing path. The Colab notebook runs the full LoRA-on-Qwen
> pipeline; that's where the headline numbers come from."

---

## Recording tips

- Record screen at 1080p, 30 fps, lossless if possible.
- Mute Gradio's tab audio; voice over narration only.
- Cut between live demo and plots every 6–8 s — judges glaze on static frames.
- Show *cumulative reward* counter visible in the metrics panel during all live segments.
- End on the README so the judge knows where to click next.
