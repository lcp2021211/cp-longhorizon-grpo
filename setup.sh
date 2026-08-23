#!/bin/bash
# 项目环境搭建脚本，建议在全新 conda env 里跑
# 用法: bash setup.sh

set -e

ENV_NAME="agentrl"
# tau2-bench >=1.0 requires Python 3.12.
PYTHON_VERSION="3.12"

echo "=== [1/6] 创建 conda 环境: $ENV_NAME ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在，跳过创建步骤..."
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -c "import sys; assert sys.version_info[:2] == (3, 12), 'agentrl must use Python 3.12 for tau2-bench'"

echo "=== [2/6] 安装 PyTorch (CUDA 12.6) ==="
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== [3/6] 安装项目依赖 ==="
pip install -r requirements.txt

echo "=== [4/6] 安装 current tau2-bench ==="
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU2_DIR="${PROJECT_DIR}/tau2-bench"
# Tested current-main revision on 2026-08-18 (package version 1.0.1).
TAU2_COMMIT="c3398666e6559e3a063da3fc04b5acf7f941464e"
if [ ! -f "${TAU2_DIR}/src/tau2/gym/gym_agent.py" ]; then
    if [ -d "${TAU2_DIR}" ]; then
        echo "${TAU2_DIR} 是旧 tau-bench 快照，请移走后重新运行 setup.sh"
        exit 1
    fi
    git clone https://github.com/sierra-research/tau2-bench.git "${TAU2_DIR}"
    git -C "${TAU2_DIR}" checkout "${TAU2_COMMIT}"
fi
pip install -e "${TAU2_DIR}[gym]"

echo "=== [5/6] 安装本地 veRL ==="
pip install -e "${PROJECT_DIR}/verl"
cd "${PROJECT_DIR}/agentic-grpo-longhorizon"

echo "=== [6/6] 从 ModelScope 下载策略模型 ==="
# 国内网络,用 ModelScope 更快
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
"

if [ "${DOWNLOAD_USER_SIMULATOR:-0}" = "1" ]; then
    python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', cache_dir='./models')"
else
    echo "跳过 72B user simulator；需要时使用 DOWNLOAD_USER_SIMULATOR=1 bash setup.sh"
fi

echo "=== 搭建完成 ==="
echo "激活环境: conda activate $ENV_NAME"
