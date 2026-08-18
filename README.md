# Agentic GRPO Long-Horizon

在 current **tau2-bench airline** 上训练长程工具 Agent，并用 **SALT + ProGPO + LATA** 分别解决 mixed-outcome 的粗粒度 credit assignment、all-fail 的零梯度死锁，以及长轨迹中的信号稀释。

[English](README_EN.md) · [完整训练与消融指南](agentic-grpo-longhorizon/docs/training_and_ablation_guide.md) · [算法实现说明](agentic-grpo-longhorizon/docs/salt_progpo_lata_tau2.md)

> 状态：tau2 适配、三模块训练链路、`2^3` 消融、诊断指标和单元测试已经完成；仓库尚未声明新的 Qwen2.5-7B tau2 GPU 训练成绩。历史 50-task `tau-bench` 结果与当前 official split 不可直接比较。

## 项目做什么

标准 GRPO 只根据同一任务的多条轨迹最终奖励计算相对 advantage。长程 Agent 训练会遇到两个典型问题：

- 一组 8 条轨迹全部失败时，binary outcome 都是 0，GRPO 没有学习方向。
- 一组已有成功/失败差异时，每条轨迹仍只得到一个标量 advantage，无法区分共同的中性步骤与真正导致成功/失败的分叉步骤。

本项目把一次 group update 处理为：

```text
同一 tau2 task 的 8 条在线 rollout
  │
  ├─ outcome 有成功/失败差异
  │    GRPO trajectory advantage
  │      └─ SALT：相同 transition 合并为 step advantage
  │
  ├─ 8 条全部失败
  │    ProGPO：first-visit progress fallback
  │    （该分支不再经过 SALT）
  │
  └─ 全成功，或 progress 也无差异
       advantage = 0

step/trajectory advantage
  └─ LATA：真实 assistant-turn 权重 / sqrt(policy token 数)
       └─ PPO clipped policy update
```

三个模块互相独立：

| 模块 | 何时触发 | 作用 | 不做什么 |
|---|---|---|---|
| SALT | mixed-outcome group | 把 trajectory advantage 重分配到 assistant step | 不增加新的 reward |
| ProGPO | all-fail group | 用 progress 给不同失败轨迹一个弱排序信号 | 有 outcome 差异时不干预 |
| LATA | 可独立开关 | 按真实 turn 传到 token，并使用 `1/sqrt(L)` | 不改变 reward source |

SALT 的 transition key 为 `(s_{t-1}, a_t, s_t)`，其中 state 是最近 `h=3` 个 `(action, observation)`。完全相同的 transition occurrence 取其来源轨迹 advantage 的算术平均；唯一或分叉 transition 保留原 advantage。工具 action/observation 使用稳定 canonical key，自由文本采用保守 exact matching。

## tau2 数据与 Agent 流程

项目使用 setup 中固定 revision 的 current tau2 Gym API 和官方 airline split：

| 文件 | 行数 | 用途 |
|---|---:|---|
| `agentic-grpo-longhorizon/experiments/tau2/train.parquet` | 30 | official `train` task |
| `agentic-grpo-longhorizon/experiments/tau2/val.parquet` | 20 | official `test` task |

parquet 只保存 task ID 和最小 prompt metadata，不保存离线轨迹。环境 reset 会动态注入 domain policy 与首条模拟用户消息；训练用工具 schema 从 current tau2 生成并固化在 `agentic-grpo-longhorizon/configs/tool_config/tau_bench_airline_tools.yaml`。Agent 随后在线执行多轮回复/工具调用，直到 `done`、环境终止或达到 turn 上限。

默认 `train_batch_size=4`、`rollout.n=8`，所以一次采样产生 4 个独立 task group、共 32 条轨迹。advantage 只在各自 task 的 8 条轨迹内计算；四组随后可以合并成同一个 policy-gradient batch 更新共享模型。

## 公平的 `2^3` 消融

八组实验共用一个 estimator、一个训练配置、相同 rollout metadata 和相同优化器，只改变三个真布尔开关：

| ID | SALT | ProGPO | LATA |
|---|:---:|:---:|:---:|
| `000_vanilla` | 关 | 关 | 关 |
| `100_salt` | 开 | 关 | 关 |
| `010_progpo` | 关 | 开 | 关 |
| `001_lata` | 关 | 关 | 开 |
| `110_salt_progpo` | 开 | 开 | 关 |
| `101_salt_lata` | 开 | 关 | 开 |
| `011_progpo_lata` | 关 | 开 | 开 |
| `111_full` | 开 | 开 | 开 |

`000` 已用数值测试对齐 veRL 原始 trajectory-level GRPO，`111` 已对齐独立的完整 SALT + ProGPO + LATA estimator。这样不会用旧 token-position LATA 混入部分实验，保证全因子比较使用同一实现口径。

## 快速开始

### 1. 环境

目标环境是 Linux、Python 3.12、PyTorch 2.7 和 NVIDIA CUDA。训练 7B policy 并在本地部署 72B-AWQ 用户模拟器时，推荐把它们放在不同 GPU；使用远程 API 时只需本地训练资源。

```bash
git clone https://github.com/qiqihezh/agentic-grpo-longhorizon.git
cd agentic-grpo-longhorizon

# 使用远程 simulator 时直接运行：bash setup.sh
# 需要同时下载本地 72B-AWQ 时：
DOWNLOAD_USER_SIMULATOR=1 bash setup.sh
conda activate agentrl
```

`setup.sh` 会安装本地修改版 veRL 和固定 revision 的 `tau2-bench[gym]`。`flash-attn` 会在 PyTorch 之后使用 `--no-build-isolation` 安装。

### 2. 选择 policy warm start

训练配置默认指向：

```text
agentic-grpo-longhorizon/experiments/sft_lora_merged
```

该 checkpoint 不随 Git 提供，`setup.sh` 也不会生成它。请准备一个合并后的 SFT checkpoint，或直接使用下载的 Qwen2.5-7B-Instruct，并在所有消融组传入同一个绝对路径：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
test -f "$MODEL_PATH/config.json"
```

### 3. 启动用户模拟器

本地 OpenAI-compatible vLLM：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-72B-Instruct-AWQ \
CUDA_DEVICES=1 PORT=8001 \
bash agentic-grpo-longhorizon/scripts/vllm_server/72b.sh

curl -fsS http://localhost:8001/v1/models
```

也可以使用任意支持多轮 Chat Completions 的远程 OpenAI-compatible API：

```bash
export TAU2_USER_MODEL='provider-model-name'
export TAU2_USER_PROVIDER='openai'
export TAU2_USER_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='your-key'
```

API key 不应写进 YAML 或提交到 Git。完整矩阵会产生大量在线对话，请提前确认并发限制、超时与费用。

### 4. 检查并运行

先列出矩阵并做无副作用 dry-run：

```bash
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh --list

bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$MODEL_PATH" \
  --dry-run
```

只跑最终方案：

```bash
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$MODEL_PATH"
```

顺序跑完八组：

```bash
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments all \
  --model-path "$MODEL_PATH"
```

也可传空格或逗号分隔的子集，例如 `--experiments 000_vanilla,111_full`。常用参数：

| 参数 | 行为 |
|---|---|
| `--steps 2` | 极短链路 smoke run；不能作为实验结果 |
| `--skip-data` | 只使用已有 train/val parquet，缺失即报错 |
| `--rebuild-data` | 训练前重建一次共享 parquet |
| `--resume-mode auto` | 恢复未完成组，跳过已达到目标 step 的组 |
| `--resume-mode disable` | 仅允许空 checkpoint 目录，防止混写 |
| `--continue-on-error` | 某组失败后继续后续组，最终仍返回失败状态 |

runner 默认顺序执行，不使用会争抢单卡资源的 Hydra multirun。数据只构建一次，八组使用相同 parquet。

首次非 dry-run 时，每个 variant 会原子创建 `run_manifest.json`。恢复、跳过和每个训练进程启动前，runner 都会校验实验开关、本地 actor/ref 的实际模型文件、公共/tool/interaction 配置、关键实现代码、tau2 pinned commit/airline 数据/实际 Python import 来源、simulator 路由环境变量，以及 train/val parquet。身份不一致，或旧目录已有产物却缺少 manifest，都会安全拒绝。目标步数不属于运行身份，因此同一身份可用 `--steps` 延长训练；远程 Hugging Face ID 无法做本地内容 hash，正式实验应使用固定的本地绝对路径。

## 输出与诊断

```text
agentic-grpo-longhorizon/experiments/ablations/<variant>/
├── run_manifest.json
├── checkpoints/
├── logs/training_<timestamp>.log
└── hydra/<timestamp>/
```

重点检查：

- `salt/merge_rate`、`salt/num_merged_transitions`
- `salt/graph_invalid_spans`、`salt/uncovered_token_rate`
- `progpo/all_fail_group_rate`、`progpo/trigger_rate`
- `progpo/progress_degenerate_rate`、`progpo/lambda_effective`
- `lata/metadata_fallback_samples`、`lata/uncovered_token_rate`

如果 mixed-outcome 已出现而 `salt/merge_rate` 长期接近 0，说明 canonical-exact matcher 实际没有产生足够共享 transition，应先检查 trace/matcher，不能只凭开关为 true 宣称 SALT 生效。

默认关闭训练期 official-test validation（`val_before_train=false, test_freq=-1`），避免反复观察 test 后选模或调参。所有组完成固定 step budget 后，再以统一配置评测 official test 20 tasks；当前 evaluator 中应把 `pass_hat_1` 解释为单次采样平均成功率。checkpoint 导出与评测命令见[完整指南](agentic-grpo-longhorizon/docs/training_and_ablation_guide.md)。

## 测试

```bash
# matrix/runner（不需要 GPU）
python -m unittest discover \
  -s agentic-grpo-longhorizon/tests \
  -p 'test_*.py' -v

# 算法、trace 与 progress
PYTHONPATH="$PWD/verl:$PWD/agentic-grpo-longhorizon" \
python -m pytest -q \
  verl/tests/trainer/ppo/test_progpo_lata.py \
  verl/tests/experimental/agent_loop/test_salt_trace.py \
  agentic-grpo-longhorizon/src/envs/tests/test_progpo_progress.py
```

核心算法测试覆盖 8 种组合、000/111 数值等价、all-fail 分支、SALT transition 合并/分叉、真实 turn LATA、空 mask、无效 span 和 metadata fallback。

## 关键文件

| 文件 | 作用 |
|---|---|
| `agentic-grpo-longhorizon/configs/train/grpo/agentic_ablation_tau2.yaml` | 八组共享训练配置 |
| `agentic-grpo-longhorizon/configs/ablation/salt_progpo_lata_matrix.yaml` | `2^3` 矩阵 |
| `agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.py` | 校验、规划、恢复与顺序执行 |
| `agentic-grpo-longhorizon/src/envs/tau2_adapter.py` | current tau2 Gym adapter |
| `agentic-grpo-longhorizon/src/envs/progpo_progress.py` | first-visit progress |
| `verl/verl/experimental/agent_loop/salt_trace.py` | 稳定 canonical transition key |
| `verl/verl/experimental/agent_loop/tool_agent_loop.py` | assistant span/trace 采集 |
| `verl/verl/trainer/ppo/core_algos.py` | SALT、ProGPO、LATA 与组合 estimator |
| `verl/verl/trainer/ppo/ray_trainer.py` | rollout metadata 传递与 diagnostics |

## 实验边界

- 当前代码固定 tau2 revision，并使用 official train 30 / test 20；旧仓库的 50-task 人工切分属于另一个实验设置。
- 自由文本 transition 当前是 exact matching；SALT 论文在连续文本场景使用 semantic matching，但没有公开完整 matcher 规格。本实现不伪造模型或阈值。
- 本地/API user simulator 都会影响任务分布与可复现性。公平消融必须固定 simulator 模型、endpoint、温度和版本。
- 单元测试通过只说明算法和运行链路满足设计，不等价于完成 7B GPU 训练或获得 benchmark 提升。

历史 PRM-Lite、旧 LATA 和 Turn-Discount 材料保留在 [`docs/ablation`](agentic-grpo-longhorizon/docs/ablation/)，仅作项目演进记录。

## 参考

- [SALT: EACL 2026 Findings](https://aclanthology.org/2026.findings-eacl.247/)
- [ProGPO: Progress-conditioned Group Policy Optimization](https://arxiv.org/abs/2607.22724)
- [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- [volcengine/verl](https://github.com/volcengine/verl)
