#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentrl

export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p experiments/progpo_lata_tau2

python scripts/train/grpo/build_grpo_parquet.py \
    --train-task-split train \
    --val-task-split test \
    --output-train experiments/tau2/train.parquet \
    --output-val experiments/tau2/val.parquet

python -m verl.trainer.main_ppo \
    --config-path="$(pwd)/configs/train/grpo" \
    --config-name=progpo_lata_tau2 2>&1 \
    | tee experiments/progpo_lata_tau2/training.log
