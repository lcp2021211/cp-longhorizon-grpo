from src.envs.progpo_progress import (
    compute_progress_score,
    initialize_progress_state,
    record_observation_transition,
)


def test_first_visit_coverage_uses_exact_observation_strings():
    state = {}
    initialize_progress_state(state, "lobby")

    assert record_observation_transition(state, "reservation A", source="tool")
    assert not record_observation_transition(state, "reservation A", source="tool")
    assert not record_observation_transition(state, "lobby", source="user")
    assert record_observation_transition(state, "Reservation A", source="tool")

    # Exact-string comparison means capitalization remains a distinct observation.
    assert compute_progress_score(state) == 0.5
    assert state["progress_novel_count"] == 2
    assert state["progress_action_count"] == 4
    assert state["progress_by_source"] == {
        "tool": {"actions": 3, "novel": 2},
        "user": {"actions": 1, "novel": 0},
    }


def test_zero_action_trajectory_has_zero_progress():
    state = {}
    initialize_progress_state(state, "initial")
    assert compute_progress_score(state) == 0.0
