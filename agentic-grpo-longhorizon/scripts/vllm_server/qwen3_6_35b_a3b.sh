#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6 tau2 user simulator served from the project conda environment.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentrl-qwen35}"

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found; activate ${CONDA_ENV_NAME} first or set SKIP_CONDA_ACTIVATE=1" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi
MODEL_PATH="${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"

if [[ -z "${CUDA_DEVICES:-}" ]]; then
    echo "Set CUDA_DEVICES explicitly, for example CUDA_DEVICES=1,2." >&2
    exit 2
fi

IFS=',' read -r -a selected_gpus <<< "${CUDA_DEVICES}"
if [[ "${#selected_gpus[@]}" -ne "${TP_SIZE}" ]]; then
    echo "CUDA_DEVICES contains ${#selected_gpus[@]} GPU(s), but TP_SIZE=${TP_SIZE}." >&2
    exit 2
fi

python - <<'CHECK'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
try:
    installed = Version(version("vllm"))
except PackageNotFoundError as exc:
    raise SystemExit("vLLM is not installed in the active environment") from exc
if installed < Version("0.19.0"):
    raise SystemExit(f"Qwen3.6 requires vLLM >= 0.19.0; found {installed}")
print(f"vLLM compatibility check passed: {installed}")
CHECK

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --language-model-only \
    --trust-remote-code
