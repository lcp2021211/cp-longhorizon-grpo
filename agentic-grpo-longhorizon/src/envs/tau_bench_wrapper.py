"""tau2-bench Gym wrapper used by independent evaluation and SFT collection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import Counter

from src.envs.tau2_adapter import Tau2GymAdapter, get_tau2_task_ids


@dataclass
class TrajectoryStep:
    """一轮交互的完整记录"""
    turn_idx: int
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_name: Optional[str] = None


@dataclass
class TrajectoryResult:
    """一条完整轨迹的结果,用于评测和 RL 训练"""
    task_id: str
    success: bool           # outcome reward (0/1)
    reward: float           # τ-bench 原生 reward
    num_turns: int
    num_tool_calls: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)  # OpenAI 格式
    error: Optional[str] = None  # 如果 trajectory 异常中止
    # [污染标记] 截断发生时的 turn 索引；None 表示未被截断污染
    was_contaminated_from_turn: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "raw_messages": self.raw_messages,
            "error": self.error,
            "was_contaminated_from_turn": self.was_contaminated_from_turn,
        }


class TauBenchWrapper:
    """
    对 τ-bench 的薄封装,提供两个关键能力:
    1. run_single_task: 给定 policy 和 task_id,跑一条轨迹
    2. batch_eval: 批量评测,用于 baseline 测试
    
    policy 需要实现 __call__(messages: list[dict]) -> dict 接口,
    返回 OpenAI 格式的 assistant message (可能包含 tool_calls).
    """

    def __init__(
        self,
        env_name: str = "airline",        # "airline" | "retail"
        user_strategy: str = "llm",       # τ-bench 默认用 llm simulator
        user_model: str = "gpt-4o",       # 后面会改成本地 Qwen-72B
        user_provider: str = "openai",    # 或 "anthropic" / "local"
        user_base_url: Optional[str] = None,  # user simulator 请求的 vLLM 地址
        task_split: str = "base",
        task_index: Optional[int] = None,
    ):
        self.env_name = env_name
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_base_url = user_base_url
        self.task_split = task_split
        self.task_index = task_index

    def _make_env(self, task_idx: int):
        """Create one isolated current tau2 Gym trajectory."""
        return Tau2GymAdapter(
            domain=self.env_name,
            task_id=task_idx,
            task_split=self.task_split,
            task_id_mode="index",
            max_steps=62,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_base_url=self.user_base_url,
        )

    def get_num_tasks(self) -> int:
        return len(get_tau2_task_ids(self.env_name, self.task_split))

    def run_single_task(
        self,
        task_idx: int,
        policy,
        max_turns: int = 30,
    ) -> TrajectoryResult:
        env = self._make_env(task_idx)
        env.max_steps = 2 * max_turns + 2
        initial_observation, _ = env.reset()
        # 把环境可用工具注册给 policy，模型才知道能调哪些工具
        if hasattr(policy, "set_tools"):
            policy.set_tools(env.tool_schemas)

        # [Fix: policy 状态泄漏] 每个 task 开始前重置截断标记
        if hasattr(policy, "was_truncated"):
            policy.was_truncated = False

        # tau2 reset gives the first simulated user message and the authoritative
        # domain policy. Both must be visible before the first policy generation.
        if self.env_name == "retail":
            system_content = (
                "# Current Date Context\n"
                "The current date is 2024-05-15 (Wednesday). "
                "When users mention dates without specifying the year, "
                "always assume they refer to 2024. "
                "All product orders and exchanges should use 2024 dates unless explicitly stated otherwise."
            )
        else:
            system_content = (
                "# Current Date Context\n"
                "The current date is 2024-05-15 (Wednesday). "
                "When users mention dates without specifying the year, "
                "always assume they refer to 2024. "
                "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
            )
        messages = [{"role": "system", "content": system_content}]
        if env.policy:
            messages.append(
                {"role": "system", "content": "# tau2 Domain Policy\n" + env.policy}
            )
        messages.append({"role": "user", "content": initial_observation})
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        done = False
        error_msg = None
        turn_idx = 0
        tool_call_count = 0
        recent_tool_calls: list[tuple[str, str]] = []
        was_contaminated_from_turn: Optional[int] = None

        try:
            while not done and turn_idx < max_turns:
                assistant_msg = policy(messages)
                messages.append(assistant_msg)
                
                # [污染检测] policy 一旦截断，标记当前 turn 为污染起点
                if hasattr(policy, "was_truncated") and policy.was_truncated and was_contaminated_from_turn is None:
                    was_contaminated_from_turn = turn_idx
                
                steps.append(TrajectoryStep(
                    turn_idx=turn_idx,
                    role="assistant",
                    content=assistant_msg.get("content", "") or "",
                    tool_calls=assistant_msg.get("tool_calls"),
                ))

                # 判定是否有 tool_calls
                tcs = assistant_msg.get("tool_calls")
                if tcs:
                    # [Loop detection] 检测完全重复的工具调用
                    for tc in tcs:
                        call_sig = (tc["function"]["name"], tc["function"]["arguments"])
                        recent_tool_calls.append(call_sig)
                    if len(recent_tool_calls) > 20:
                        recent_tool_calls = recent_tool_calls[-20:]
                    call_counts = Counter(recent_tool_calls)
                    if any(c >= 3 for c in call_counts.values()):
                        error_msg = "Loop detected: same tool call repeated 3+ times"
                        break  # break while

                    for tc in tcs:
                        tool_call_count += 1
                        import json

                        tool_result = env.step_tool(
                            tc["function"]["name"],
                            json.loads(tc["function"]["arguments"]),
                        )
                        obs_content = tool_result.observation
                        
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{tool_call_count}"),
                            "name": tc["function"]["name"],
                            "content": str(obs_content),
                        }
                        messages.append(tool_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="tool",
                            content=tool_msg["content"], tool_name=tc["function"]["name"],
                        ))
                        total_reward += tool_result.reward
                        if tool_result.done:
                            done = True
                            break
                else:
                    user_obs = env.step_response(assistant_msg.get("content", "") or "")
                    if user_obs.done:
                        done = True
                        total_reward += user_obs.reward
                    else:
                        obs_str = user_obs.observation
                        user_msg = {"role": "user", "content": obs_str}
                        messages.append(user_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="user", content=obs_str,
                        ))
                turn_idx += 1
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        success = total_reward >= 1.0
        return TrajectoryResult(
            task_id=env.task_id, success=success, reward=total_reward,
            num_turns=turn_idx, num_tool_calls=tool_call_count,
            steps=steps, raw_messages=messages, error=error_msg,
            was_contaminated_from_turn=was_contaminated_from_turn,
        )
