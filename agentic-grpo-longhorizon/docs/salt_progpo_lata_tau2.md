# SALT + ProGPO + LATA on tau2-bench

## 调整后的项目

策略模型仍在当前 tau2-bench airline Gym 环境中完成多轮客服任务，但训练信号现在分三层处理：

```text
同一 tau2 task 的 8 条 rollout
  -> 稀疏最终结果 R + first-visit progress P
  -> 按 task/uid 分组
       R 有成功/失败差异:
         GRPO trajectory advantage
         -> SALT trajectory graph
         -> 每个 assistant turn 的 step advantage
       全部失败:
         ProGPO progress fallback trajectory advantage
       全部成功，或 progress 也无差异:
         advantage = 0
  -> LATA token weight / sqrt(policy_token_count)
  -> PPO policy update
```

三部分分别解决不同问题：SALT 解决“有最终奖励差异，但整条轨迹仍只有一个粗粒度 advantage”；ProGPO 解决“8 条轨迹全失败，原始 GRPO 完全没有信号”；LATA 解决长回复中的梯度稀释。

## rollout 如何记录 SALT 轨迹

每轮 assistant generation 记录一个半开 token 区间 `[token_start, token_end)`，随后在 tau2 返回工具 observation 或模拟用户回复时补全：

```python
{
    "token_start": 120,
    "token_end": 168,
    "action_type": "tool",
    "action_key": "get_reservation_details + canonical JSON arguments",
    "observation_key": "canonical tau2 observation",
    "mergeable": True,
}
```

一个 assistant turn 即一个 SALT step；即使一次 generation 解析出多个工具调用，也只建立一个聚合 transition，避免同一 token span 被重复赋权。`done` 工具、terminal user response、长度截断也会关闭 span；没有真实 next observation 的截断 step 标记为不可合并，并安全回退到原轨迹 advantage。

## SALT 的精确计算

对 mixed-outcome group，先得到原始 GRPO 标量 `A_i`。状态使用最近 `h=3` 个 `(action, observation)` 对：

```text
s_t = [(a_{t-h+1}, o_{t-h+1}), ..., (a_t, o_t)]
k_t = (s_{t-1}, a_t, s_t)
```

只有 `k_t` 完全一致的 transition 才 merge。对一个重复 transition 的全部 occurrence：

```text
A'_step = mean(A_i of every occurrence)
```

unique/divergent transition 保留所属轨迹的 `A_i`。因此成功和失败轨迹共同执行的中性步骤会被软化，而只出现在成功轨迹中的恢复动作仍保留正 advantage。实现按 occurrence 而非按 trajectory 去重，与论文附录 Algorithm 1 一致。

SALT 只作用于 outcome 有差异的 group。全失败组的 ProGPO progress scalar 不再经过 SALT，避免把 outcome-only 方法未经验证地二次用于 progress 信号。

## tau2 的 transition 等价规则

- 工具 action：工具名 + 排序后的 canonical JSON 参数。
- 工具 observation：JSON key/Unicode/空白稳定化后的完整 observation。
- 对话 action/user observation：Unicode 与空白规范化后做保守 exact match。
- 初始 prompt：使用稳定 BLAKE2 摘要作为 root key，防止同一 task 下不同初始用户措辞被误合并。

SALT 论文对连续文本的 AppWorld 使用了 embedding 语义匹配，但没有公开 embedding 模型和阈值。本项目因此不伪造所谓“论文精确阈值”，而采用完全可复现的 canonical-exact 首版，并用 `salt/merge_rate` 监控它是否真正产生合并。

## LATA 传到 token

SALT step advantage 只覆盖对应 assistant token span；环境 observation 与 padding 始终为零。LATA 使用真实 assistant turn，而不是包含工具 observation 的 response buffer 绝对位置：

```text
w_k proportional to alpha^(N_turn-1-k), alpha=1.05
sum_k(w_k * tokens_in_turn_k) / L = 1
A_token = A_step(k) * w_k / sqrt(L)
```

这样工具返回再长也不会额外压低后续 assistant turn；`alpha=1.05` 最多按 turn 连乘，不会在上千 token 上指数爆炸。没有合法 SALT span 覆盖的 policy token 保留 trajectory scalar和中性 LATA 权重，而不是被置零。

## 入口与关键文件

- `configs/train/grpo/salt_progpo_lata_tau2.yaml`：新训练配置，`G=8`、`h=3`。
- `scripts/train/grpo/run_salt_progpo_lata_tau2.sh`：构建官方 train/test parquet 并启动训练。
- `verl/experimental/agent_loop/salt_trace.py`：跨 Ray worker 稳定的 action/observation canonical key。
- `verl/experimental/agent_loop/tool_agent_loop.py`：采集 root、transition 和 token span。
- `verl/trainer/ppo/core_algos.py`：SALT graph、ProGPO switch、LATA token advantage。
- `verl/trainer/ppo/ray_trainer.py`：把 object trace 传给 estimator 并上报指标。

运行：

```bash
bash agentic-grpo-longhorizon/scripts/train/grpo/run_salt_progpo_lata_tau2.sh
```

## 重点指标

- `salt/merge_rate`：mixed-outcome 组内参与重复 transition 的 occurrence 比例。
- `salt/num_merged_transitions`：本次 update 真正被合并的 transition key 数。
- `salt/merged_tool_occurrences` / `salt/merged_text_occurrences`：工具与自由文本各自的合并量。
- `salt/invalid_spans` / `salt/uncovered_token_rate`：trace 到 token 对齐是否健康。
- `salt/graph_invalid_spans`：因 span 非法而被禁止进入 SALT merge 的 step 数。
- `lata/metadata_fallback_samples` / `lata/uncovered_token_rate`：turn 权重是否退化为中性 fallback。
- `progpo/trigger_rate`：全失败组启用 progress fallback 的比例。
- `progpo/lambda_effective`：随 all-fail group 比例衰减的 fallback 强度。

健康检查：如果进入 mixed-outcome 阶段后，`salt/merge_rate` 连续多个 update 仍接近 0，那么当前 canonical-exact matcher 实际上没有提供细粒度修正；此时应先检查 trace，再把 embedding matcher 作为独立消融，而不是把零合并结果解释成 SALT 的效果。

当前提交提供算法、rollout metadata、训练配置、诊断和单元测试；它没有虚构新的训练收益。最终效果仍需完成 7B GPU 训练和独立 pass@k 评测后报告。

参考：SALT 论文与附录算法 <https://aclanthology.org/2026.findings-eacl.247/>。
