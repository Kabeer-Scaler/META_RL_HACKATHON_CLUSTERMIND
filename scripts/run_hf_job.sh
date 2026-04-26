#!/usr/bin/env bash
# Entry-point for Hugging Face Jobs runs.
# Clones the repo, installs training deps, runs SFT+RL training across all 8
# OpenEnv scenarios, runs per-scenario evaluation against the trained adapter,
# and pushes the LoRA adapter + training logs + per-scenario eval results to
# HF Hub.
#
# Required env vars (passed via `hf jobs run --secrets` and `-e`):
#   HF_TOKEN        — HF write token (Kaggle/HF secret)
#   HUB_MODEL_ID    — destination HF Hub repo, e.g. "user/clustermind-lora"
#
# Optional env-var overrides:
#   REPO_URL        — git URL of this repo (default: GitHub origin)
#   BRANCH          — git branch to train from (default: main)
#   BASE_MODEL      — base LLM (default: Qwen/Qwen2.5-0.5B-Instruct)
#   RL_ALGO         — auto | grpo | ppo | reinforce (default: auto)
#   GRPO_K          — group size for GRPO (default: 2)
#   SFT_EPISODES    — default 16
#   RL_EPISODES     — default 24
#   EVAL_EPISODES   — default 8 (one per scenario for the in-train eval pass)
#   SEED            — default 42
#   MODE_FLAG       — --quick | --full (default: --quick)
#   SCENARIOS       — space-separated list (default: all 8 OpenEnv scenarios)
#   LEVELS          — space-separated curriculum levels (default: "3 4 5";
#                     L>=5 is required for chaos_arena per evaluate.py:52)
#   POST_EVAL_EPISODES — episodes per scenario in the post-training eval pass
#                        across all 5 baselines + RL-LoRA (default: 3)
#   SKIP_POST_EVAL — set to "1" to skip the per-scenario evaluation step

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN env var is required (pass via --secrets HF_TOKEN)}"

# HUB_MODEL_ID can be:
#   "user/repo"   — used as-is
#   "repo"        — username is resolved from HF_TOKEN via whoami()
#   unset         — defaults to "clustermind-lora", username from token
HUB_REPO_NAME="${HUB_REPO_NAME:-${HUB_MODEL_ID:-clustermind-lora}}"
if [[ "${HUB_REPO_NAME}" != */* ]]; then
    echo "[hf-job] resolving HF username from token..."
    pip install --quiet --no-cache-dir "huggingface_hub>=0.25"
    HF_USERNAME="$(python - <<'PY'
import os
from huggingface_hub import whoami
print(whoami(token=os.environ["HF_TOKEN"])["name"])
PY
)"
    HUB_MODEL_ID="${HF_USERNAME}/${HUB_REPO_NAME}"
    echo "[hf-job] resolved HUB_MODEL_ID=${HUB_MODEL_ID}"
else
    HUB_MODEL_ID="${HUB_REPO_NAME}"
fi
export HUB_MODEL_ID

REPO_URL="${REPO_URL:-https://github.com/Kabeer-Scaler/META_RL_HACKATHON_CLUSTERMIND.git}"
BRANCH="${BRANCH:-main}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RL_ALGO="${RL_ALGO:-auto}"
GRPO_K="${GRPO_K:-2}"
SFT_EPISODES="${SFT_EPISODES:-16}"
RL_EPISODES="${RL_EPISODES:-24}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
SEED="${SEED:-42}"
MODE_FLAG="${MODE_FLAG:---quick}"

# All 8 OpenEnv scenarios per openenv.yaml. Override via SCENARIOS="a b c".
SCENARIOS_DEFAULT="demand_spike cooling_failure hidden_degradation cascading_failure energy_squeeze vip_job_arrival triple_crisis chaos_arena"
SCENARIOS="${SCENARIOS:-${SCENARIOS_DEFAULT}}"
LEVELS="${LEVELS:-3 4 5}"
POST_EVAL_EPISODES="${POST_EVAL_EPISODES:-3}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-0}"

echo "[hf-job] starting on $(date -u +%FT%TZ)"
echo "[hf-job] repo=${REPO_URL}@${BRANCH}  hub=${HUB_MODEL_ID}  base=${BASE_MODEL}"
echo "[hf-job] scenarios=${SCENARIOS}"
echo "[hf-job] levels=${LEVELS}"

# Some PyTorch/CUDA images are minimal; install git if missing.
if ! command -v git >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates
fi

WORKDIR="${WORKDIR:-/work}"
# If we're already inside a checkout of this repo, reuse it. Otherwise clone.
if [ -d "${WORKDIR}/.git" ] && [ -f "${WORKDIR}/scripts/train_trl.py" ]; then
    echo "[hf-job] reusing existing repo at ${WORKDIR}"
    cd "${WORKDIR}"
    git -C "${WORKDIR}" fetch --quiet origin "${BRANCH}" && \
        git -C "${WORKDIR}" checkout --quiet "origin/${BRANCH}" -- scripts/ clustermind/ requirements-train.txt 2>/dev/null || true
else
    echo "[hf-job] cloning ${REPO_URL}@${BRANCH} -> ${WORKDIR}"
    rm -rf "${WORKDIR}"
    git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
    cd "${WORKDIR}"
fi

echo "[hf-job] installing training deps..."
pip install --no-cache-dir -r requirements-train.txt

# Sanity-check GPU + bitsandbytes
python - <<'PY'
import torch
print(f"[hf-job] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[hf-job] device={torch.cuda.get_device_name(0)}  "
          f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
try:
    import bitsandbytes as bnb  # noqa: F401
    print("[hf-job] bitsandbytes import OK")
except Exception as e:
    print(f"[hf-job] bitsandbytes import FAILED: {e}")
PY

echo "[hf-job] launching training -> ${HUB_MODEL_ID}"
# Word-splitting on $SCENARIOS / $LEVELS is intentional — argparse needs them as N tokens.
python scripts/train_trl.py \
    --mode auto \
    --base-model "${BASE_MODEL}" \
    --rl-algo "${RL_ALGO}" \
    --grpo-group-size "${GRPO_K}" \
    --sft-episodes "${SFT_EPISODES}" \
    --rl-episodes "${RL_EPISODES}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}" \
    --scenarios ${SCENARIOS} \
    --levels ${LEVELS} \
    ${MODE_FLAG} \
    --hub-model-id "${HUB_MODEL_ID}"

ADAPTER_DIR="results/adapters/clustermind_lora"
EVAL_OUT="results/evaluation_metrics.json"
if [ "${SKIP_POST_EVAL}" = "1" ]; then
    echo "[hf-job] SKIP_POST_EVAL=1 — skipping per-scenario evaluation"
elif [ ! -d "${ADAPTER_DIR}" ]; then
    echo "[hf-job] no adapter at ${ADAPTER_DIR} — skipping per-scenario evaluation"
else
    echo "[hf-job] per-scenario evaluation: 5 baselines + RL-LoRA on ${SCENARIOS}"
    python scripts/evaluate.py \
        --episodes "${POST_EVAL_EPISODES}" \
        --scenarios ${SCENARIOS} \
        --levels ${LEVELS} \
        --include-llm transformers \
        --base-model "${BASE_MODEL}" \
        --rl-adapter "${ADAPTER_DIR}" \
        --output "${EVAL_OUT}"

    echo "[hf-job] uploading ${EVAL_OUT} -> ${HUB_MODEL_ID}"
    python - <<'PY'
import os
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="results/evaluation_metrics.json",
    path_in_repo="evaluation_metrics.json",
    repo_id=os.environ["HUB_MODEL_ID"],
    token=os.environ["HF_TOKEN"],
)
print(f"[hf-job] pushed evaluation_metrics.json -> https://huggingface.co/{os.environ['HUB_MODEL_ID']}/blob/main/evaluation_metrics.json")
PY

    echo "[hf-job] per-scenario reward summary (RL-LoRA):"
    python - <<'PY'
import json
with open("results/evaluation_metrics.json") as f:
    d = json.load(f)
rl = d["agents"].get("RL-LoRA")
if not rl:
    print("(RL-LoRA agent not present in evaluation_metrics.json)")
else:
    for k, s in sorted(rl["by_scenario"].items()):
        print(f"  {k:32s} reward={s.get('avg_reward', 0):+6.2f} "
              f"crit={s.get('critical_completion_rate', 0)*100:5.1f}% "
              f"outage={s.get('avg_outage_count', 0):.2f}")
PY
fi

echo "[hf-job] done. Artifacts live at https://huggingface.co/${HUB_MODEL_ID}/tree/main"
