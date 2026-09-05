"""Thin adapter around the current :mod:`tau2` Gymnasium environment.

The old project talked directly to ``tau_bench.envs.get_env``.  Current
tau2-bench exposes a Gymnasium-compatible ``AgentGymEnv`` and string task IDs.
This module keeps those version-specific details out of the veRL interaction and
evaluation code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from typing import Any, Literal, Optional


TaskIdMode = Literal["id", "index"]


@dataclass(frozen=True)
class Tau2StepResult:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def get_tau2_task_ids(domain: str, task_split: str = "base") -> list[str]:
    """Load task IDs from tau2's registered, official split."""
    try:
        from tau2.registry import registry
    except ImportError as exc:  # pragma: no cover - exercised in integration env
        raise ImportError(
            "tau2-bench is not installed. Install the current repository with "
            "`pip install -e './tau2-bench[gym]'`."
        ) from exc

    tasks = registry.get_tasks_loader(domain)(task_split_name=task_split)
    return [str(task.id) for task in tasks]


def resolve_tau2_task_id(
    domain: str,
    task_id: int | str,
    task_split: str,
    task_id_mode: TaskIdMode,
) -> str:
    """Resolve either an official task ID or an index within a split."""
    if task_id_mode == "id":
        resolved = str(task_id)
        all_ids = set(get_tau2_task_ids(domain, "base"))
        if resolved not in all_ids:
            raise ValueError(
                f"Unknown tau2 task id {resolved!r} for domain {domain!r}."
            )
        return resolved
    if task_id_mode != "index":
        raise ValueError("task_id_mode must be either 'id' or 'index'")

    split_ids = get_tau2_task_ids(domain, task_split)
    index = int(task_id)
    if not 0 <= index < len(split_ids):
        raise IndexError(
            f"Task index {index} is outside tau2 split {task_split!r} "
            f"with {len(split_ids)} tasks."
        )
    return split_ids[index]


def get_tau2_tool_schemas(domain: str, *, include_done: bool = True) -> list[dict]:
    """Return OpenAI schemas from the current tau2 domain registry."""
    from tau2.registry import registry

    environment = registry.get_env_constructor(domain)(solo_mode=False)
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    if include_done:
        from tau2.environment.tool import as_tool
        from tau2.gym.gym_agent import done

        schemas.append(as_tool(done).openai_schema)
    return schemas


class Tau2GymAdapter:
    """One trajectory backed by tau2 ``AgentGymEnv``."""

    def __init__(
        self,
        *,
        domain: str = "airline",
        task_id: int | str,
        task_split: str = "base",
        task_id_mode: TaskIdMode = "id",
        max_steps: int = 30,
        user_model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
        user_provider: str = "openai",
        user_base_url: Optional[str] = None,
        user_temperature: float = 0.0,
        user_llm_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.domain = domain
        self.task_split = task_split
        self.task_id = resolve_tau2_task_id(
            domain, task_id, task_split, task_id_mode
        )
        self.max_steps = int(max_steps)
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_base_url = user_base_url
        self.user_temperature = float(user_temperature)
        self.user_llm_args = deepcopy(user_llm_args or {})
        self._env = None
        self.initial_observation = ""
        self.info: dict[str, Any] = {}

    def _litellm_model_name(self) -> str:
        if self.user_provider == "openai" and not self.user_model.startswith(
            "openai/"
        ):
            return f"openai/{self.user_model}"
        return self.user_model

    def _make_env(self):
        try:
            from tau2.gym.gym_agent import AgentGymEnv
        except ImportError as exc:  # pragma: no cover - integration dependency
            raise ImportError(
                "Current tau2-bench with the gym extra is required. Install it "
                "with `pip install -e './tau2-bench[gym]'`."
            ) from exc

        user_llm_args: dict[str, Any] = deepcopy(self.user_llm_args)
        user_llm_args.setdefault("temperature", self.user_temperature)
        if self.user_base_url:
            user_llm_args["api_base"] = self.user_base_url
            user_llm_args["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")

        return AgentGymEnv(
            domain=self.domain,
            task_id=self.task_id,
            max_steps=self.max_steps,
            solo_mode=False,
            user_llm=self._litellm_model_name(),
            user_llm_args=user_llm_args,
            all_messages_as_observation=False,
        )

    def reset(self) -> tuple[str, dict[str, Any]]:
        self._env = self._make_env()
        observation, info = self._env.reset()
        self.initial_observation = str(observation)
        self.info = info
        return self.initial_observation, info

    @property
    def policy(self) -> str:
        return str(self.info.get("policy", ""))

    @property
    def tool_schemas(self) -> list[dict]:
        tools = self.info.get("tools", [])
        return [tool.openai_schema for tool in tools]

    def _step(self, action: str) -> Tau2StepResult:
        if self._env is None:
            raise RuntimeError("Tau2GymAdapter.reset() must be called before step().")
        observation, reward, terminated, truncated, info = self._env.step(action)
        self.info = info
        return Tau2StepResult(
            observation=str(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def step_tool(self, name: str, arguments: dict[str, Any]) -> Tau2StepResult:
        action = json.dumps(
            {"name": name, "arguments": arguments, "requestor": "assistant"},
            ensure_ascii=False,
        )
        return self._step(action)

    def step_response(self, content: str) -> Tau2StepResult:
        return self._step(content)

    def close(self) -> None:
        """Release a Gym trajectory that veRL truncated before tau2 terminated."""
        if self._env is None:
            return
        simulation_done = getattr(self._env, "_simulation_done", None)
        if simulation_done is not None and simulation_done.is_set():
            return
        try:
            self.step_tool("done", {})
        except Exception:
            # The orchestrator uses a daemon thread; close is best-effort during
            # rollout cancellation and must not mask the already computed score.
            pass
