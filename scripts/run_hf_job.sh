#!/usr/bin/env bash
# Entry-point for Hugging Face Jobs runs.
# Clones the repo, installs training deps, runs SFT+RL training, and pushes
# the trained LoRA adapter (+ model card + logs) to HF Hub.
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
#   EVAL_EPISODES   — default 6
#   SEED            — default 42
#   MODE_FLAG       — --quick | --full (default: --quick)

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN env var is required (pass via --secrets HF_TOKEN)}"
: "${HUB_MODEL_ID:?HUB_MODEL_ID env var is required (pass via -e HUB_MODEL_ID=user/repo)}"

REPO_URL="${REPO_URL:-https://github.com/Kabeer-Scaler/META_RL_HACKATHON_CLUSTERMIND.git}"
BRANCH="${BRANCH:-main}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RL_ALGO="${RL_ALGO:-auto}"
GRPO_K="${GRPO_K:-2}"
SFT_EPISODES="${SFT_EPISODES:-16}"
RL_EPISODES="${RL_EPISODES:-24}"
EVAL_EPISODES="${EVAL_EPISODES:-6}"
SEED="${SEED:-42}"
MODE_FLAG="${MODE_FLAG:---quick}"

echo "[hf-job] starting on $(date -u +%FT%TZ)"
echo "[hf-job] repo=${REPO_URL}@${BRANCH}  hub=${HUB_MODEL_ID}  base=${BASE_MODEL}"

# Some PyTorch/CUDA images are minimal; install git if missing.
if ! command -v git >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates
fi

WORKDIR="${WORKDIR:-/work}"
rm -rf "${WORKDIR}"
git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"

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
python scripts/train_trl.py \
    --mode auto \
    --base-model "${BASE_MODEL}" \
    --rl-algo "${RL_ALGO}" \
    --grpo-group-size "${GRPO_K}" \
    --sft-episodes "${SFT_EPISODES}" \
    --rl-episodes "${RL_EPISODES}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}" \
    ${MODE_FLAG} \
    --hub-model-id "${HUB_MODEL_ID}"

echo "[hf-job] done. Adapter live at https://huggingface.co/${HUB_MODEL_ID}"
