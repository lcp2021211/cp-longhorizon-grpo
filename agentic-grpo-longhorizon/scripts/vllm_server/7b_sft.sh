#!/bin/bash
# 启动导出的 7B checkpoint vLLM server（SFT 或 RL 均可）
# served-model-name 保持稳定，让评测 policy 配置不用改。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH=${MODEL_PATH:-"${PROJECT_DIR}/experiments/sft_lora_merged"}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.82}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
CUDA_DEVICES=${CUDA_DEVICES:-0}   # GPU0，GPU1 留给 72B user sim

echo "Starting vLLM server (SFT-merged 7B)..."
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "TP:     $TP_SIZE"
echo "GPU:    $CUDA_DEVICES"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
