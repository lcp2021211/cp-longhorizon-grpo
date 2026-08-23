# ProGPO + LATA on tau2-bench

## 现在的项目是什么

这条训练路径让 Qwen 策略模型在 tau2-bench airline 中处理长程客服任务：理解用户需求，遵守航空政策，查询用户、预订和航班，必要时修改或取消预订，最后由 tau2 判定任务是否成功。训练仍以二值 outcome 为真实目标，但在同一 task 的整个 rollout group 都失败时，用 ProGPO 恢复一个弱的组内排序信号。

## 算法数据流

```text
official tau2 train task
  -> K 条独立 AgentGymEnv rollout
  -> outcome R 与 first-visit progress P
  -> 按 task/uid 分组
       outcome 有差异: 原样 GRPO advantage
       outcome 全为 0: 若 P 有差异，启用 ProGPO fallback
       P 也无差异: advantage = 0
  -> LATA 的 turn weight + 1/sqrt(L)
  -> PPO clipped policy update
```

对轨迹 `i`，初始 observation 先进入 visited set。每执行一个 agent action 后，若 tau2 返回的完整 observation 字符串首次出现，则计一次 novelty：

```text
P_i = C_i / T_i
```

`C_i` 是首次访问的后续 observation 数，`T_i` 是 agent action 数。不做文本归一化，保持 ProGPO 的 exact-string 定义。

在一次 update 中，全失败 group 占比为 `q_fail`，默认有效系数为：

```text
lambda_eff = 0.3 * q_fail
A_i = lambda_eff * (P_i - mean(P_group)) / (std_pop(P_group) + eps)
```

这个 fallback 只在 outcome 全为零且 progress 方差足够大时生效。一旦 group 里出现成功轨迹，立即回到原始 GRPO，不会让 progress 压过真实 outcome。

LATA 随后把轨迹级 advantage 传到 token：

```text
w_t ∝ alpha^(L-1-t),  mean(w)=1,  alpha=1.05
A_token(t) = A_i * w_t / sqrt(L)
```

## tau2-bench 适配

- 使用 `tau2.gym.gym_agent.AgentGymEnv`，而不再直接调用旧 `tau_bench.envs.get_env`。已针对 2026-08-18 的 current-main commit `c339866`（package 1.0.1）校验。
- 训练/验证数据默认来自官方 airline `train`/`test` split，当前分别为 30/20 个 task。
- reset 后把 tau2 的 domain policy 和首条模拟用户消息动态注入 veRL prompt。
- schema 从当前 tau2 registry 生成：14 个 airline 工具，再加 Gym `done`。
- tau2 的 orchestrator step 与旧项目 agent turn 含义不同；适配层用 `2 * max_turns + 2` 换算，避免过早截断。
- task ID 作为字符串传递，不再依赖旧版固定 `0..49` 索引。

## 关键文件

- `src/envs/tau2_adapter.py`：当前 Gym API、task split 和 schema 适配。
- `src/envs/progpo_progress.py`：first-visit progress 记录。
- `src/envs/tau_bench_interaction.py`：轨迹生命周期，outcome/progress 返回。
- `verl/verl/trainer/ppo/core_algos.py`：`grpo_progpo_lata` advantage estimator。
- `configs/train/grpo/progpo_lata_tau2.yaml`：训练超参。
- `scripts/train/grpo/run_progpo_lata_tau2.sh`：数据生成和训练入口。

## 运行与观测

```bash
bash setup.sh
conda activate agentrl
bash agentic-grpo-longhorizon/scripts/train/grpo/run_progpo_lata_tau2.sh
```

训练时重点看以下指标：

- `progpo/all_fail_group_rate`：当前 update 中全失败 group 占比。
- `progpo/trigger_rate`：progress fallback 真正触发的 group 占比。
- `progpo/progress_degenerate_rate`：全失败且 progress 也无法区分的比例。
- `progpo/lambda_effective`：随 `q_fail` 自动衰减的 fallback 强度。
- `progpo/progress_tool_score_mean` / `progpo/progress_user_score_mean`：工具观测和用户回复分别的 novelty，用于检查信号来源。

理想的训练动态是：早期 all-fail 高，fallback 提供弱信号；随着策略开始成功，all-fail 和 `lambda_effective` 下降，主导权自动交还 outcome GRPO。

tau2 是对话环境，自然语言用户回复比 ALFWorld 的离散观测更容易形成 exact-string novelty。当前实现保留论文的完整 observation 定义，并拆分记录 tool/user novelty。如果后者长期接近 1 且主导组内排序，应将“tool-only coverage”作为下一个明确消融，而不在主实验中悄悄改变 ProGPO 定义。

## 当前验证边界

已完成算法单测、Python 编译、当前 tau2 split/schema 集成校验和 parquet 入口校验。仓库中原有的 PRM-Lite 数字不能当作 ProGPO + tau2 的实验结果；新方法的收益需要在 7B 策略模型上完成实际 GPU 训练和独立 pass@k 评测后再报告。
