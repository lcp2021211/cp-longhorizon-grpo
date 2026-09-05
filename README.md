# Agentic GRPO Long-Horizon

## 简述

本项目面向长程工具调用 Agent 的强化学习训练，在 [tau2-bench](https://github.com/sierra-research/tau2-bench) airline 环境中，将 **SALT、ProGPO 与 LATA** 组合到 GRPO 训练流程中，以改善长轨迹中的 credit assignment。

当前主实验使用：

- **Policy**：Qwen3.5-9B，以 text-only LoRA 方式训练；
- **User simulator**：Qwen3.6-35B-A3B，通过 OpenAI-compatible vLLM 服务提供；
- **训练框架**：项目内适配 Qwen3.5 的 `verl_qwen35`，采用 FSDP2、异步 vLLM rollout 和多轮工具调用。

## 做了什么

项目在 trajectory-level GRPO 的基础上加入三个可独立启用的模块：

- **SALT**：在同一任务组同时包含成功与失败轨迹时，将轨迹级 advantage 细化到 assistant step；相同 transition 共享组内信号，分叉 transition 保留各自信号。
- **ProGPO**：在一组轨迹全部失败时，根据 first-visit progress 提供辅助排序，使训练仍能从不同失败路径中获得学习信号。
- **LATA**：按照真实 assistant turn 将 advantage 传递到 policy token，并使用 `1/sqrt(L)` 缩放长轨迹信号。

## 数据与流程

项目固定使用 tau2-bench airline 的官方划分：

| 数据 | 任务数 | 用途 |
|---|---:|---|
| `train` | 30 | 在线 rollout 与训练 |
| `test` | 20 | 训练完成后的统一评测 |

生成的 `train.parquet` 和 `val.parquet` 只保存任务 ID 与最小 prompt metadata，不保存离线轨迹。环境 reset 时会注入 airline policy 和模拟用户的首条消息，Agent 随后在线完成多轮回复与工具调用。

默认每个训练 step 从 4 个任务各采样 8 条轨迹，共得到 32 条 rollout。每个任务组独立计算 advantage，再合并为同一次 policy update：

```text
tau2 task
  -> Qwen3.6-35B-A3B 模拟用户
  -> Qwen3.5-9B Agent 在线回复与调用工具
  -> terminal outcome + progress + assistant-step trace
  -> GRPO + 可选 SALT / ProGPO / LATA
  -> LoRA policy update
```

训练产物按实验隔离在 `agentic-grpo-longhorizon/experiments/ablations/<variant>/`，包括运行清单、checkpoint、日志、TensorBoard 事件与 Hydra 配置。

## 启动指南

以下命令均从仓库根目录开始。训练入口默认使用名为 `agentrl-qwen35` 的 Conda 环境，并从仓库内的 `verl_qwen35` 加载 veRL；请先准备与本机 CUDA 环境匹配的 Python 3.12 训练环境、Qwen3.5-9B 权重和 Qwen3.6-35B-A3B 权重。

### 1. 启动 user simulator

在独立终端中使用两张 GPU 启动 Qwen3.6-35B-A3B：

```bash
cd agentic-grpo-longhorizon

MODEL_PATH=/absolute/path/to/Qwen3.6-35B-A3B \
CUDA_DEVICES=1,2 \
TP_SIZE=2 \
bash scripts/vllm_server/qwen3_6_35b_a3b.sh
```

确认服务可用：

```bash
curl -fsS http://localhost:8001/v1/models
```

也可以使用远程 OpenAI-compatible endpoint：

```bash
export TAU2_USER_MODEL='Qwen/Qwen3.6-35B-A3B'
export TAU2_USER_PROVIDER='openai'
export TAU2_USER_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='your-key'
```

### 2. 检查实验配置

在训练终端指定 Qwen3.5-9B policy，并先查看矩阵与执行计划：

```bash
cd agentic-grpo-longhorizon

export POLICY_MODEL_PATH=/absolute/path/to/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=0

bash scripts/train/grpo/run_ablation_matrix.sh --list

bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$POLICY_MODEL_PATH" \
  --dry-run
```

runner 会在首次正式运行时生成并校验官方 train/test parquet；同一矩阵中的实验复用相同数据。

### 3. 启动训练

运行完整方法：

```bash
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$POLICY_MODEL_PATH"
```

顺序运行全部八组消融：

```bash
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments all \
  --model-path "$POLICY_MODEL_PATH"
```

也可传入逗号或空格分隔的实验子集。常用选项包括 `--steps`、`--skip-data`、`--rebuild-data`、`--resume-mode auto|disable` 和 `--continue-on-error`。

查看 TensorBoard：

```bash
tensorboard \
  --logdir experiments/ablations/111_full/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `agentic-grpo-longhorizon/configs/train/grpo/agentic_ablation_tau2.yaml` | Qwen3.5-9B 主训练配置 |
| `agentic-grpo-longhorizon/configs/ablation/salt_progpo_lata_matrix.yaml` | `2^3` 消融矩阵 |
| `agentic-grpo-longhorizon/configs/interaction_config/tau2_airline_progpo.yaml` | tau2 环境与 Qwen3.6 user simulator 配置 |
| `agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh` | 单组、子集与全矩阵训练入口 |
| `agentic-grpo-longhorizon/scripts/vllm_server/qwen3_6_35b_a3b.sh` | Qwen3.6-35B-A3B 本地服务入口 |
| `agentic-grpo-longhorizon/src/envs/tau2_adapter.py` | tau2 Gym 适配 |
| `agentic-grpo-longhorizon/src/envs/progpo_progress.py` | ProGPO progress 计算 |
| `verl_qwen35/verl/experimental/agent_loop/salt_trace.py` | SALT transition trace |
| `verl_qwen35/verl/experimental/agent_loop/tool_agent_loop.py` | 多轮工具调用与 assistant span 采集 |
| `verl_qwen35/verl/trainer/ppo/core_algos.py` | SALT、ProGPO、LATA advantage 实现 |
| `verl_qwen35/verl/trainer/ppo/ray_trainer.py` | rollout metadata、训练与指标汇总 |

算法细节见 [SALT + ProGPO + LATA 实现说明](agentic-grpo-longhorizon/docs/salt_progpo_lata_tau2.md)。
