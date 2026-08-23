"""Model-free progress accounting used by ProGPO.

The implementation follows arXiv:2607.22724 literally: observations are compared
as exact strings, the initial observation is inserted before the first action, and
the trajectory score is the fraction of actions whose next observation is a first
visit.  No lower-casing, whitespace normalization, or semantic encoder is used.
"""
from __future__ import annotations

from typing import Any


def initialize_progress_state(state: dict[str, Any], initial_observation: Any) -> None:
    """Insert the initial observation and reset per-trajectory counters."""
    observation = str(initial_observation)
    state["visited_observations"] = {observation}
    state["progress_action_count"] = 0
    state["progress_novel_count"] = 0
    state["progress_by_source"] = {}


def record_observation_transition(
    state: dict[str, Any],
    next_observation: Any,
    *,
    source: str,
) -> bool:
    """Record one executed environment action and return whether it was novel."""
    if "visited_observations" not in state:
        raise RuntimeError(
            "ProGPO progress state is uninitialized. Call initialize_progress_state "
            "with the tau2 reset observation before executing an action."
        )

    observation = str(next_observation)
    visited: set[str] = state["visited_observations"]
    is_novel = observation not in visited
    visited.add(observation)

    state["progress_action_count"] += 1
    if is_novel:
        state["progress_novel_count"] += 1

    source_stats = state["progress_by_source"].setdefault(
        source, {"actions": 0, "novel": 0}
    )
    source_stats["actions"] += 1
    source_stats["novel"] += int(is_novel)
    return is_novel


def compute_progress_score(state: dict[str, Any]) -> float:
    """Return first-visit observation coverage P=C/T for one trajectory."""
    action_count = int(state.get("progress_action_count", 0))
    if action_count == 0:
        return 0.0
    return float(state.get("progress_novel_count", 0)) / action_count
