# SALT + ProGPO + LATA：训练、消融与评测指南

本文对应当前项目的主实验路径：在 **tau2-bench airline** 上，用同一套 rollout 和 PPO 配置，对 **SALT、ProGPO、LATA** 做可组合的 `2^3` 全因子消融。

> 当前状态：算法、rollout metadata、tau2 适配、配置和单元测试已经实现；仓库尚未提供一次完整 Qwen2.5-7B GPU 训练后的新 tau2 成绩。文末的旧数值来自历史 `tau-bench` 50-task 实验，不能与当前 official 30/20 split 横向比较。

## 1. 最终算法

一次训练 update 默认从 4 个 tau2 task 各采样 8 条轨迹，共得到 4 个独立 rollout group、32 条轨迹。不同 task 的 advantage 可以放在同一个 batch 中更新，是因为每条轨迹的 advantage 都只在自己的 task group 内归一化；合并的只是最终 policy-gradient mini-batch，不会把不同任务的奖励相互比较。

```text
同一 task 的 8 条 rollout
  -> terminal outcome R + first-visit progress P + assistant step trace
  -> 按 task/uid 计算 group advantage
       outcome 有成功/失败差异:
         GRPO trajectory advantage
         -> 可选 SALT：重分配为 assistant-step advantage
       8 条全部失败:
         可选 ProGPO：用 progress 做弱 fallback 排序
       全成功，或 progress 也无差异:
         advantage = 0
  -> 可选 LATA：按真实 assistant turn 加权并除以 sqrt(policy token 数)
  -> PPO clipped update
```

### 1.1 SALT：mixed-outcome 组内的细粒度 credit assignment

SALT 只处理 outcome 已经有差异的 group。它先使用原始 GRPO 得到每条轨迹的标量 advantage，再把每个 assistant turn 表示成 transition：

```text
s_t = 最近 h=3 个 (action, observation)
k_t = (s_{t-1}, action_t, s_t)
```

同组轨迹中 canonical key 完全相同的 transition 会合并，其 step advantage 取这些 occurrence 所属轨迹 advantage 的均值；分叉或唯一 transition 保留原轨迹 advantage。SALT 不引入新的 reward model，也不修改 terminal outcome。

tau2 的工具 action 使用“工具名 + 排序后的 JSON 参数”匹配，observation 使用 canonical exact match。自由文本同样采用保守 exact match。因此需要监控 `salt/merge_rate`：若长期接近 0，SALT 代码虽已启用，但实际上没有产生有效的 step 重分配。

### 1.2 ProGPO：all-fail 组的 progress fallback

对轨迹 `i`：

```text
P_i = 首次访问的后续 observation 数 / agent action 数
```

若同组 8 条轨迹全失败、但 `P_i` 有方差，ProGPO 使用标准化 progress 形成弱 advantage。默认：

```text
lambda_eff = 0.3 * 当前 update 的 all-fail group 比例
A_i = lambda_eff * z_score(P_i)
```

只要组内出现成功轨迹，progress fallback 立即关闭，重新以真实 outcome 为准；如果 progress 也退化为常数，则该组 advantage 仍为 0。

### 1.3 LATA：把 step/trajectory 信号传到 token

LATA 使用真实 assistant turn，而不是 response buffer 的绝对 token 位置：

```text
w_k proportional to alpha^(N_turn - 1 - k), alpha=1.05
sum_k(w_k * policy_tokens_in_turn_k) / L = 1
A_token = A_step(k) * w_k / sqrt(L)
```

工具 observation、用户消息和 padding 始终被 mask。这样长工具返回不会改变 assistant turn 的相对权重，长轨迹的信号也不会被 `1/L` 过度稀释。

## 2. tau2 数据如何使用

`setup.sh` 会安装固定 revision 的新 `tau2-bench` Gym API。当前 airline official split 为：

| Split | task 数 | 用途 |
|---|---:|---|
| `train` | 30 | GRPO/SALT/ProGPO/LATA rollout 与训练 |
| `test` | 20 | 固定训练预算完成后的 held-out 最终评测 |

数据文件只保存 task ID 和最小 prompt metadata，不保存离线轨迹：

```text
train.parquet：30 rows，每 row 一个 official train task
val.parquet：20 rows，每 row 一个 official test task（veRL 仍加载，但训练期默认不执行）
rollout.n=8：运行时把一个 task 扩展成 8 条独立在线轨迹
```

环境 reset 后，tau2 的 domain policy 与首条模拟用户消息会动态注入 prompt。训练工具 schema 由 current tau2 生成后固化在 `configs/tool_config/tau_bench_airline_tools.yaml`，并由 ToolAgentLoop 加载。Agent 随后在 `AgentGymEnv` 中交替生成自然语言或工具调用，直到调用 `done`、环境终止或达到 turn 上限。

手动构建数据：

```bash
cd /absolute/path/to/agentic-grpo-longhorizon/agentic-grpo-longhorizon

python scripts/train/grpo/build_grpo_parquet.py \
  --train-task-split train \
  --val-task-split test \
  --output-train experiments/tau2/train.parquet \
  --output-val experiments/tau2/val.parquet
```

正常输出应显示 `Train: 30 rows` 和 `Val: 20 rows`。若仍出现旧版 50-task 口径，说明导入了仓库中的 legacy `tau-bench/`，而不是 setup 安装的 `tau2-bench/`。

## 3. 安装与路径前置

以下命令中的两个目录不要混淆：

```text
REPO_ROOT=/absolute/path/to/agentic-grpo-longhorizon
PROJECT_ROOT=$REPO_ROOT/agentic-grpo-longhorizon
```

- `tau-bench/`：历史代码快照，仅用于复核旧实验。
- `tau2-bench/`：`setup.sh` 安装的新 benchmark，才是当前训练路径使用的版本。
- `verl/`：本项目修改过的本地 veRL，必须 editable install，不能换成 PyPI 同名包。

推荐安装：

```bash
cd /absolute/path/to/agentic-grpo-longhorizon

# 同时下载本地 72B-AWQ user simulator；若使用远程 API，可不设此变量。
DOWNLOAD_USER_SIMULATOR=1 bash setup.sh
conda activate agentrl
```

安装后检查：

```bash
python -c "import tau2, verl, torch; print(tau2.__file__); print(verl.__file__); print(torch.__version__)"
python -c "from tau2.registry import registry; print(len(registry.get_tasks_loader('airline')(task_split_name='train')), len(registry.get_tasks_loader('airline')(task_split_name='test')))"
```

第二条应输出 `30 20`。

### 3.1 FlashAttention 安装失败

`flash-attn` 需要在 PyTorch 已安装后、关闭 build isolation 单独编译，因此它没有作为普通条目放在 `requirements.txt`。如果 `setup.sh` 的显式安装步骤失败，可在确认 CUDA 工具链后手动执行：

```bash
conda activate agentrl
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e './tau2-bench[gym]'
pip install -e ./verl
```

同时确认系统 CUDA toolkit 中存在 `ptxas`。脚本默认使用 `/usr/local/cuda-12.4`；机器路径不同则应在 runner 或启动脚本中改为实际 `CUDA_HOME`，不能仅根据 PyTorch wheel 的 `cu126` 名称猜测系统 toolkit 路径。

### 3.2 模型路径必须使用绝对路径

server 脚本会相对项目目录寻找 setup 下载的模型，但 ModelScope 的实际缓存层级仍可能随版本变化。建议先定位模型，并在复现实验时显式传绝对路径：

```bash
find /absolute/path/to/agentic-grpo-longhorizon -name config.json \
  -path '*Qwen2.5-7B-Instruct*' -print
```

## 4. 明确处理缺失的 warm start

主配置默认写的是：

```text
experiments/sft_lora_merged
```

该目录和模型权重不会随 Git 仓库提供。开始训练前必须选择以下一种方式，并通过 runner 的 `--model-path` 同时覆盖 actor 和 reference model。

### 方式 A：从用户自己的 SFT merged checkpoint 开始

```bash
MODEL_PATH=/absolute/path/to/sft_lora_merged
test -f "$MODEL_PATH/config.json"
```

### 方式 B：直接从 Qwen2.5-7B-Instruct 开始

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
test -f "$MODEL_PATH/config.json"
```

方式 B 是干净、可复现的零 SFT 起点，但不应与历史“从旧 SFT warm start 出发”的数值比较。所有 8 个消融组必须使用完全相同的 `MODEL_PATH`。

当前仓库保留的 SFT 数据采集配置属于旧 50-task 流程，不能直接当作新 tau2 official train/test 的无泄漏 warm start 指南。

## 5. 用户模拟器

训练和评测都要求一个 OpenAI Chat Completions 兼容的 user simulator。默认模型名为 `Qwen/Qwen2.5-72B-Instruct-AWQ`，endpoint 为 `http://localhost:8001/v1`。

### 5.1 本地 vLLM

在独立终端启动；训练进程使用 GPU 0 时，默认把 simulator 放在 GPU 1：

```bash
conda activate agentrl
cd /absolute/path/to/agentic-grpo-longhorizon/agentic-grpo-longhorizon

MODEL_PATH=/absolute/path/to/Qwen2.5-72B-Instruct-AWQ \
CUDA_DEVICES=1 \
PORT=8001 \
bash scripts/vllm_server/72b.sh
```

另一个终端检查服务：

```bash
curl -fsS http://localhost:8001/v1/models
```

默认本地资源布局是 2×80GB GPU：GPU 0 跑 7B policy/actor，GPU 1 跑 72B-AWQ simulator。若 simulator 改为远程 API，本地只需要为 policy 预留训练资源。

### 5.2 远程或第三方 OpenAI-compatible API

interaction 配置支持环境变量覆盖，不需要修改 YAML。密钥也只放环境变量，不提交到 Git：

```bash
export TAU2_USER_MODEL='API 实际暴露的模型名'
export TAU2_USER_PROVIDER='openai'
export TAU2_USER_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='your-key'
```

API 必须支持多轮 Chat Completions。一次默认 update 可能同时推进 32 条轨迹，应预先确认并发、速率限制、超时和费用。做公平消融时，8 组必须固定同一个 simulator 模型、温度和 provider；API 后端版本变化也应记录。

## 6. `2^3` 消融矩阵

矩阵定义在 `configs/ablation/salt_progpo_lata_matrix.yaml`：

| ID | SALT | ProGPO | LATA | 解释 |
|---|:---:|:---:|:---:|---|
| `000_vanilla` | 关 | 关 | 关 | trajectory-level vanilla GRPO |
| `100_salt` | 开 | 关 | 关 | 只细化 mixed-outcome step advantage |
| `010_progpo` | 关 | 开 | 关 | 只给 all-fail group progress fallback |
| `001_lata` | 关 | 关 | 开 | 只改变 turn/token 信号传输 |
| `110_salt_progpo` | 开 | 开 | 关 | 两种 credit source，不做 LATA |
| `101_salt_lata` | 开 | 关 | 开 | mixed-outcome step 信号 + LATA |
| `011_progpo_lata` | 关 | 开 | 开 | all-fail fallback + LATA |
| `111_full` | 开 | 开 | 开 | 最终完整方案 |

所有组统一使用 `configs/train/grpo/agentic_ablation_tau2.yaml` 和 estimator `grpo_agentic_ablation`。除三个布尔开关、实验名和输出目录外，task split、warm start、seed、group size、采样温度、KL、学习率和训练步数均应保持不变。

## 7. 运行训练

查看 runner 识别到的完整矩阵：

```bash
bash scripts/train/grpo/run_ablation_matrix.sh --list
```

先做配置 dry-run；它不应启动 GPU 训练：

```bash
cd /absolute/path/to/agentic-grpo-longhorizon/agentic-grpo-longhorizon

bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct \
  --dry-run
```

dry-run 不创建 parquet、输出目录或 `run_manifest.json`。若 parquet 已存在，它会校验完整运行身份；若 parquet 尚不存在，则只校验已有 manifest 的静态字段，并明确提示 dataset hash 会在真实构建数据后再校验。

runner 默认只在 parquet 缺失时构建一次，并在同一次矩阵执行中供所有 variant 复用。`--rebuild-data` 会在训练前重新生成，`--skip-data` 则要求两个文件已经存在，否则立即报错。

### 7.1 只跑最终方案

```bash
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct
```

先做极短 GPU smoke run 时，可以显式限制目标 step；这类结果只用于验证链路，不能进入实验表：

```bash
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct \
  --steps 2
```

### 7.2 只跑一组或若干组

```bash
# 单组 baseline
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 000_vanilla \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct

# 指定子集
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 000_vanilla,100_salt,010_progpo,001_lata,111_full \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct
```

### 7.3 顺序跑完 8 组

```bash
bash scripts/train/grpo/run_ablation_matrix.sh \
  --experiments all \
  --model-path /absolute/path/to/Qwen2.5-7B-Instruct
```

默认输出约定：

```text
experiments/ablations/<variant>/
  run_manifest.json
  checkpoints/
  logs/training_<timestamp>.log
  hydra/<timestamp>/
```

runner 默认顺序执行并在任一训练失败后停止；`--continue-on-error` 可记录失败并继续后续组。不要让多个进程同时写同一个 variant 目录。八组顺序训练约等于完整方案八倍的训练预算；只有资源和 simulator endpoint 相互隔离时才应并行。

### 7.4 本地日志与 SwanLab

默认配置只启用本地 `console`，不要求 SwanLab 登录：

```yaml
trainer:
  logger: [console]
```

需要 SwanLab 时，可以统一改为 `[console, swanlab]` 并提前完成登录。八组必须采用同样的日志设置；日志后端不应改变训练超参。

## 8. Resume 与重新开始

每个 variant 的运行身份保存在 `experiments/ablations/<variant>/run_manifest.json`。首次非 dry-run 会在 parquet 就绪后原子创建该文件；后续恢复、跳过，以及每个真实训练 subprocess 启动前都会先重新计算并校验：

- experiment ID 与 SALT/ProGPO/LATA 三个开关；
- actor/ref 的有效模型路径；本地模型会对目录中所有非缓存/非临时 regular files 建立排序后的 path、size 和 SHA256 清单；
- 公共训练 YAML、data/rollout tool config 与 interaction config；
- ProGPO、tau2 adapter/context/interaction/tools、SALT trace、ToolAgentLoop、`core_algos.py` 和 `ray_trainer.py` 等关键实现文件；
- tau2 checkout 路径、是否精确命中 setup 固定的 Git HEAD、tracked binary diff hash、airline domain content tree，以及当前 Python 实际导入的 `tau2` 是否来自该 checkout；
- `TAU2_USER_MODEL` / `TAU2_USER_PROVIDER` / `TAU2_USER_BASE_URL` 的实际值（不记录 API key）；
- train/val parquet 的 SHA256。

身份不一致会直接拒绝，而不会把旧 actor/optimizer checkpoint 与新的 reference model、配置或数据静默混用。已有 checkpoint、日志或 Hydra 产物但缺少 manifest 的 legacy 目录也会被拒绝；应恢复原 manifest，或把旧目录移走后创建新实验，不能手工伪造 manifest。

目标 step 故意不属于运行身份，所以同一身份可以用 `--steps` 从较小目标延长到较大目标。直接修改公共训练 YAML 会改变其 hash，不能作为延长 step 的方法。

checkpoint 策略仍由 `--resume-mode` 控制：

- 它读取 `checkpoints/latest_checkpointed_iteration.txt`；低于目标 step 时继续，高于或等于目标 step 时自动跳过该 variant。
- `--resume-mode disable` 只允许空 checkpoint 目录；发现已有内容会拒绝运行，避免覆盖旧实验。
- `auto` 或 `disable` 都不能绕过 manifest 身份校验。
- runner 没有 `resume_path` 参数。必须从一个指定 checkpoint 恢复时，直接运行 Hydra，并设置 `trainer.resume_mode=resume_path` 与 `trainer.resume_from_path=/absolute/path/to/global_step_x`；这是高级用法，不属于矩阵 runner CLI，也会绕过上述 safeguard，调用者需自行保证身份一致。
- 想从头开始时应选择新的 variant/output 目录。不要覆盖或删除仍需比较的旧实验。

manifest 是防误续训的最低安全线，不是完整实验追踪系统。远程 Hugging Face ID 只能记录 identifier，无法得到本地内容 hash；同一 simulator endpoint/model 名后的服务端权重或实现仍可能变化。manifest 也不记录 API key、所有进程环境变量或整个仓库的 Git revision，只 hash 上述关键实现文件。因此正式实验仍应使用固定本地模型路径，并额外记录仓库 commit、simulator 后端 revision 和完整运行环境。任何未覆盖条件发生变化，都应视为新实验。

## 9. 训练时重点监控

### 通用健康指标

- outcome/success 分布、`critic/score/min`、`critic/score/max`
- `actor/grad_norm`、KL、entropy、policy loss
- response policy token 数、assistant turn 数、工具调用数、异常率
- 是否频繁达到 max turns 或 max response length

### SALT

- `salt/merge_rate`
- `salt/num_merged_transitions`
- `salt/merged_tool_occurrences`、`salt/merged_text_occurrences`
- `salt/invalid_spans`、`salt/graph_invalid_spans`
- `salt/uncovered_token_rate`

`salt/merge_rate` 在进入 mixed-outcome 阶段后仍长期接近 0，通常表示 canonical exact matcher 没有真正合并 transition，而不是模型已经学好。

### ProGPO

- `progpo/all_fail_group_rate`
- `progpo/trigger_rate`
- `progpo/progress_degenerate_rate`
- `progpo/lambda_effective`
- tool/user 两类 progress 均值

理想动态是训练早期 all-fail 和 fallback 较高；出现成功轨迹后，主导权逐步交还 outcome。若 user progress 几乎恒为 1 且支配排序，应单独做 tool-only progress 实验，不能在主实验中静默修改定义。

### LATA

- `lata/metadata_fallback_samples`
- `lata/uncovered_token_rate`

这两个指标持续非零通常意味着 assistant token span 与 response mask 没有正确对齐。

## 10. 导出 checkpoint 并独立评测

公共配置默认 `val_before_train=false, test_freq=-1`，不会在训练中反复观察 official test。所有组完成预先约定的固定 step budget 后，先把 veRL FSDP checkpoint 合并成 Hugging Face 模型：

```bash
cd /absolute/path/to/agentic-grpo-longhorizon/agentic-grpo-longhorizon

python scripts/test/merge_fsdp_to_hf.py \
  --actor-dir experiments/ablations/111_full/checkpoints/global_step_300/actor \
  --output-dir experiments/ablations/111_full/hf_step_300
```

保持已配置的 user simulator 可用（本地方案为 8001），在另一个终端启动待评 policy：

```bash
conda activate agentrl
cd /absolute/path/to/agentic-grpo-longhorizon/agentic-grpo-longhorizon

MODEL_PATH="$PWD/experiments/ablations/111_full/hf_step_300" \
CUDA_DEVICES=0 \
PORT=8000 \
bash scripts/vllm_server/7b_sft.sh
```

为避免覆盖其他实验，先生成该 variant 专属的 eval 配置：

```bash
sed 's#experiments/eval_sft_airline#experiments/ablations/111_full/eval_step_300#' \
  configs/eval/eval_sft_airline.yaml \
  > /tmp/eval_111_full_step300.yaml
```

先做小规模连通性测试，再跑 official test 20 tasks × 4 samples：

```bash
python scripts/eval/eval_sft.py \
  --config /tmp/eval_111_full_step300.yaml \
  --tiny

python scripts/eval/eval_sft.py \
  --config /tmp/eval_111_full_step300.yaml
```

不要给当前 tau2 评测传历史 `experiments/sft_collect_airline/split.json`。它属于旧 50-task covered/uncovered/unseen 切分，不适用于 official test 20 tasks。

### 10.1 正确的 success-rate 口径

以 `eval_report.json` 为准：

- `pass_hat_1`：每个 task 的成功样本比例再跨 task 平均，即本项目应报告的单次采样平均成功率。
- `pass_at_1`：legacy 字段名；当前代码实际表示“一个 task 的 N 条样本中至少一条成功”的观测 task 占比，即 observed task-solved@N。默认 `N=4` 时就是 observed task-solved@4，不能当作 pass@1。
- `pass_hat_4` / `pass_hat_8`：代码使用“k 次采样至少一次成功”的无偏 pass@k 估计，不是“连续 k 次全部成功”或稳定性指标；只有 `N >= k` 才可报告。旧 JSON 在样本不足时写 `0.0`，这个值应标为 unavailable/N/A，不能解释为真实性能 0。

主表建议报告 `pass_hat_1`、轨迹异常率、平均 turns 和平均工具调用数，并同时写明 task 数、每 task 样本数、policy 温度、simulator 版本与 checkpoint。

## 11. 测试

在仓库根目录运行核心 CPU 测试：

```bash
cd /absolute/path/to/agentic-grpo-longhorizon

PYTHONPATH="$PWD/verl:$PWD/agentic-grpo-longhorizon" \
python -m pytest -q \
  verl/tests/trainer/ppo/test_progpo_lata.py \
  verl/tests/experimental/agent_loop/test_salt_trace.py \
  agentic-grpo-longhorizon/src/envs/tests/test_progpo_progress.py \
  agentic-grpo-longhorizon/tests/test_run_ablation_matrix.py
```

随后运行矩阵 dry-run，确认 8 个 variant 的三个开关、模型路径、输出目录和最终 Hydra 配置均正确。单元测试通过不等价于完成 GPU 训练或获得 benchmark 提升。

## 12. 常见故障

### `Connection refused: localhost:8001`

user simulator 未启动或端口不一致。先执行 `curl -fsS .../v1/models`，再启动训练。远程 API 还需检查证书、代理和 key 是否被 Ray worker 继承。

### `experiments/sft_lora_merged` 不存在

这是预期的缺失资产，不是 Git checkout 损坏。通过 `--model-path` 显式传入用户自己的 SFT merged checkpoint 或 base 7B。

### `ModuleNotFoundError: tau2`

运行 `pip install -e './tau2-bench[gym]'`，并确认是在 `agentrl` 环境中。不要把 legacy `tau-bench/` 加到 `PYTHONPATH` 代替它。

### task 数仍是 50

当前进程导入了旧 benchmark 或旧数据。打印 `tau2.__file__`、重新生成 parquet，并确认 official split 为 30/20。

### CUDA、Triton 或 `ptxas` 报错

检查 `CUDA_HOME` 和 `TRITON_PTXAS_PATH` 指向机器上真实存在的 toolkit。PyTorch wheel 的 CUDA runtime 版本不保证 `/usr/local/cuda-*` 路径存在。

### OOM

优先减小 `max_num_seqs`、`ppo_micro_batch_size_per_gpu` 或 vLLM `gpu_memory_utilization`。做消融时，任何内存相关改动必须同步应用到 8 组；不要只为某一组改变 batch、group size 或最大长度。

### SALT 没有效果

先看 `salt/merge_rate`、invalid span 和 uncovered token 指标。exact matching 下零合并是可能的；不能把“开关为 true”直接解释为算法实际生效。

### 训练意外接着旧 run

默认 `resume_mode=auto`。若报 `run identity mismatch`，把旧 variant 目录移走并使用新目录；若提示已有产物但缺少 `run_manifest.json`，恢复该 run 的原 manifest 或移走旧目录，不要伪造 manifest。新的实验条件应使用新的输出目录，并可设置 `resume_mode=disable`。

### 日志后端报错或等待登录

把 `trainer.logger` 设为 `[console]`。这不会改变算法，但应在所有对照组保持一致。

## 13. 新旧实验边界

| 项目 | Benchmark / split | 方法 | 状态 | 能否与当前方案比较 |
|---|---|---|---|---|
| 当前主实验 | pinned current `tau2-bench`，official train 30 / test 20 | SALT + ProGPO + LATA 与 `2^3` 消融 | 代码与测试完成；完整 7B GPU 结果待跑 | 同矩阵内可以 |
| 历史实验 | 旧 `tau-bench` airline 50 tasks，人工 40/10 与 covered/uncovered 划分 | PRM-Lite、旧 LATA、Turn-Discount 等 | 已有历史日志和 `0.240` 等报告值 | 不可以直接横比 |

旧任务、旧 split、旧 evaluator、warm start 和部分 advantage 实现均不同。历史数字只能说明旧项目曾观察到的现象，不能作为新 SALT + ProGPO + LATA 的成绩，也不能用来计算新方案相对提升。

当前方案只有在以下条件全部满足后才能在 README 中新增结果表：

1. 8 组使用同一 warm start、tau2 revision、30/20 parquet 和随机种子策略；
2. 每组完成约定训练 budget，并保留最终 Hydra config 与日志；
3. 训练期间不根据 official test 选模或调参，完成后使用同一 user simulator 和 official test 评测；
4. 主指标明确写为 `pass_hat_1`，同时报告样本数和误差/方差；
5. 若曾临时开启 official-test validation，必须披露，并且不能再把后续同 split 结果称为严格独立 held-out 评测。

## 14. 关键文件

下列 `configs/`、`scripts/`、`src/`、`docs/` 路径相对 `PROJECT_ROOT`；`verl/` 路径相对 `REPO_ROOT`。

- `configs/train/grpo/agentic_ablation_tau2.yaml`：8 组共享的训练配置。
- `configs/ablation/salt_progpo_lata_matrix.yaml`：全因子矩阵。
- `scripts/train/grpo/run_ablation_matrix.sh`：单组、子集和全矩阵入口。
- `src/envs/tau2_adapter.py`：current tau2 Gym 适配。
- `src/envs/progpo_progress.py`：first-visit progress。
- `verl/verl/experimental/agent_loop/salt_trace.py`：SALT canonical trace。
- `verl/verl/experimental/agent_loop/tool_agent_loop.py`：assistant token span 与 transition 采集。
- `verl/verl/trainer/ppo/core_algos.py`：组合 estimator。
- `verl/verl/trainer/ppo/ray_trainer.py`：metadata 传递和 diagnostics。
- `docs/salt_progpo_lata_tau2.md`：算法实现细节。

参考论文：[SALT, Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.247/)；[ProGPO: Progress-conditioned Group Policy Optimization for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2607.22724)。
