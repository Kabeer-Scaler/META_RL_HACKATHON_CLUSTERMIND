# ClusterMind demo — shot list

Use as a recording checklist. Numbers refer to scenes in
`reports/demo_video_script.md`.

| # | Source | Action | Duration |
|---|---|---|---|
| 1 | Static title card (slide / Figma) | "ClusterMind Chaos Arena" + 1-line tagline | 0:00–0:04 |
| 2 | Gradio Live tab | Reset → cluster graph appears | 0:04–0:09 |
| 3 | `results/` thumbnails | 8-plot collage (PNG mosaic) | 0:09–0:15 |
| 4 | Gradio Live tab — Random / cascading_failure | Run to end; pan over alerts panel + cluster going red | 0:15–0:30 |
| 5 | Gradio metrics panel | Final stats: 1/4 critical, health 0.10 | 0:30–0:35 |
| 6 | Gradio Live tab — Greedy / triple_crisis seed 7 | Run to end | 0:35–0:50 |
| 7 | Gradio Flight Recorder panel | Scroll narrative lines step 1–4 | 0:50–1:00 |
| 8 | Gradio Live tab — ThermalAware / chaos_arena seed 7 | Run to end; highlight INCREASE_COOLING action in event log | 1:00–1:18 |
| 9 | Gradio Baseline Comparison tab | Run all 5; highlight ThermalAware row leading | 1:18–1:25 |
| 10 | Code snippet overlay | `lora_config = LoraConfig(r=8, ...)` + `freeze base` | 1:25–1:32 |
| 11 | Training Results tab | `reward_curve.png` + `loss_curve.png` side-by-side | 1:32–1:42 |
| 12 | Colab terminal screenshot | `[LLM-RL] algorithm=grpo` + a few episode lines | 1:42–1:45 |
| 13 | README hero shot | 8 plots collage + HF Space banner | 1:45–2:00 |

## Asset prep

Before recording:

- [ ] Run a fresh Gradio session: `python app.py`. Verify port 7860 reachable.
- [ ] In Live tab, pre-select `triple_crisis` so the title card transitions cleanly.
- [ ] Pre-run `python scripts/export_replay.py --agent GreedyThroughputAgent --scenario triple_crisis --level 4 --seed 7` so the Flight Recorder file exists for the Replay tab.
- [ ] Confirm all 8 PNGs in `results/` are >50 KB and embed cleanly in README preview.
- [ ] Run `python scripts/audit_prd.py` and grab the "155/155 checks passed" line for an end-card overlay.

## Voice-over notes

- Speak conservatively — 145 wpm leaves room to breathe.
- Pause for 0.5 s on each metric reveal (lets the eye catch up).
- Pronounce: GRPO = "gee-arr-pee-oh", LoRA = "lora", PPO = "pee-pee-oh".
- Don't claim "production cluster operator" — say "OpenEnv RL benchmark".
- Don't claim trained agent beats every heuristic everywhere — say "ThermalAware wins on chaos_arena, trained agent should learn a mixed policy."

## Don't

- Don't show the local-CPU policy-net training schema as if it's the headline LoRA result.
- Don't speed up the Flight Recorder scroll past readable speed.
- Don't include the HF Space URL until it's actually live.
- Don't include any cooked numbers — every figure must come from `results/`.
