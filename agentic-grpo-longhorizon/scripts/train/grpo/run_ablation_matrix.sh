#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentrl}"

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found; activate ${CONDA_ENV_NAME} first or set SKIP_CONDA_ACTIVATE=1" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DS_SKIP_TRITON="${DS_SKIP_TRITON:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/run_ablation_matrix.py" "$@"
