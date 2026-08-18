from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_progpo_lata_outcome_advantage,
    compute_grpo_salt_progpo_lata_outcome_advantage,
    compute_lata_turn_weights,
    compute_progpo_group_scores,
    compute_salt_step_scores,
)


def test_progpo_switch_is_conditional_and_batch_scaled():
    # Three groups: mixed outcome, all fail with progress contrast, all success.
    outcomes = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    progress = torch.tensor([0.9, 0.2, 0.1, 0.3, 0.2, 0.8])
    group_index = np.array(["mixed", "mixed", "fail", "fail", "success", "success"])

    advantages, diagnostics = compute_progpo_group_scores(
        outcomes,
        progress,
        group_index,
        outcome_rewards=outcomes,
    )

    # Mixed group is the unchanged existing GRPO branch (sample std).
    torch.testing.assert_close(
        advantages[:2], torch.tensor([-0.7071058, 0.7071058]), rtol=1e-5, atol=1e-5
    )
    # q_fail=1/3, so lambda_eff=0.3/3=0.1. Population-normalized
    # progress [0.1, 0.3] becomes [-1, +1].
    torch.testing.assert_close(
        advantages[2:4], torch.tensor([-0.1, 0.1]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(advantages[4:], torch.zeros(2))

    assert diagnostics["progpo/all_fail_group_rate"] == 1 / 3
    assert diagnostics["progpo/trigger_rate"] == 1 / 3
    assert diagnostics["progpo/lambda_effective"] == pytest.approx(0.1)
    assert diagnostics["progpo/branch_by_sample"].tolist() == [
        "outcome",
        "outcome",
        "progress_fallback",
        "progress_fallback",
        "outcome",
        "outcome",
    ]


def test_progpo_discards_progress_degenerate_all_fail_group():
    outcomes = torch.zeros(4)
    progress = torch.full((4,), 0.25)
    groups = np.array([0, 0, 0, 0])

    advantages, diagnostics = compute_progpo_group_scores(
        outcomes, progress, groups, outcome_rewards=outcomes
    )

    torch.testing.assert_close(advantages, torch.zeros(4))
    assert diagnostics["progpo/num_all_fail_groups"] == 1
    assert diagnostics["progpo/num_triggered_groups"] == 0
    assert diagnostics["progpo/progress_degenerate_rate"] == 1.0


def test_progpo_lata_masks_padding_and_preserves_fallback_order():
    rewards = torch.zeros((2, 4))
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    progress = np.array([0.25, 0.75])
    outcomes = np.array([0.0, 0.0])
    groups = np.array(["task", "task"])

    advantages, _ = compute_grpo_progpo_lata_outcome_advantage(
        rewards,
        mask,
        groups,
        progress_scores=progress,
        outcome_rewards=outcomes,
    )

    assert torch.all(advantages[0, :4] < 0)
    assert torch.all(advantages[1, :2] > 0)
    assert torch.equal(advantages[1, 2:], torch.zeros(2))


def _step(start, end, action, observation, action_type="tool", mergeable=True):
    return {
        "token_start": start,
        "token_end": end,
        "action_type": action_type,
        "action_key": action,
        "observation_key": observation,
        "mergeable": mergeable,
    }


def test_salt_averages_shared_transition_and_preserves_divergent_steps():
    trajectory_advantages = torch.tensor([-1.0, 1.0])
    groups = np.array(["task", "task"])
    outcomes = np.array([0.0, 1.0])
    traces = np.empty(2, dtype=object)
    traces[0] = [
        _step(0, 2, "lookup", "reservation"),
        _step(2, 4, "wrong", "error"),
    ]
    traces[1] = [
        _step(0, 2, "lookup", "reservation"),
        _step(2, 4, "recover", "success"),
    ]

    scores, eligible, diagnostics = compute_salt_step_scores(
        trajectory_advantages,
        groups,
        traces,
        root_keys=np.array(["same-prompt", "same-prompt"], dtype=object),
        outcome_rewards=outcomes,
        history_length=3,
    )

    assert eligible.tolist() == [True, True]
    assert scores[0][0].item() == pytest.approx(0.0)
    assert scores[1][0].item() == pytest.approx(0.0)
    assert scores[0][1].item() == pytest.approx(-1.0)
    assert scores[1][1].item() == pytest.approx(1.0)
    assert diagnostics["salt/num_merged_transitions"] == 1
    assert diagnostics["salt/num_merged_occurrences"] == 2


def test_salt_same_action_with_different_resulting_state_does_not_merge():
    traces = np.empty(2, dtype=object)
    traces[0] = [_step(0, 1, "lookup", "state-a")]
    traces[1] = [_step(0, 1, "lookup", "state-b")]

    scores, _, diagnostics = compute_salt_step_scores(
        torch.tensor([-1.0, 1.0]),
        np.array([0, 0]),
        traces,
        outcome_rewards=np.array([0.0, 1.0]),
    )

    assert scores[0][0].item() == -1.0
    assert scores[1][0].item() == 1.0
    assert diagnostics["salt/num_merged_transitions"] == 0


def test_salt_history_prevents_false_merge_of_same_current_pair():
    traces = np.empty(2, dtype=object)
    traces[0] = [
        _step(0, 1, "path-a", "state-a"),
        _step(1, 2, "lookup", "reservation"),
    ]
    traces[1] = [
        _step(0, 1, "path-b", "state-b"),
        _step(1, 2, "lookup", "reservation"),
    ]

    scores, _, diagnostics = compute_salt_step_scores(
        torch.tensor([-1.0, 1.0]),
        np.array([0, 0]),
        traces,
        outcome_rewards=np.array([0.0, 1.0]),
        history_length=3,
    )

    assert [value.item() for value in scores[0]] == [-1.0, -1.0]
    assert [value.item() for value in scores[1]] == [1.0, 1.0]
    assert diagnostics["salt/num_merged_transitions"] == 0


def test_salt_invalid_span_cannot_change_other_trajectory():
    traces = np.empty(2, dtype=object)
    traces[0] = [_step(0, 3, "lookup", "reservation")]
    traces[1] = [_step(0, 2, "lookup", "reservation")]

    scores, _, diagnostics = compute_salt_step_scores(
        torch.tensor([-1.0, 1.0]),
        np.array([0, 0]),
        traces,
        outcome_rewards=np.array([0.0, 1.0]),
        response_mask=torch.ones((2, 2), dtype=torch.long),
    )

    assert scores[0][0].item() == -1.0
    assert scores[1][0].item() == 1.0
    assert diagnostics["salt/graph_invalid_spans"] == 1
    assert diagnostics["salt/num_merged_transitions"] == 0


def test_salt_progpo_lata_assigns_shared_and_unique_turn_tokens():
    rewards = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    )
    mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1]])
    traces = np.empty(2, dtype=object)
    traces[0] = [
        _step(0, 2, "lookup", "reservation"),
        _step(4, 6, "wrong", "error"),
    ]
    traces[1] = [
        _step(0, 2, "lookup", "reservation"),
        _step(4, 6, "recover", "success"),
    ]
    config = SimpleNamespace(
        turn_discount=SimpleNamespace(alpha=1.0),
        progpo=SimpleNamespace(
            tau_reward=1e-3,
            tau_progress=1e-4,
            lambda_aux=0.3,
            lambda_fixed=False,
        ),
        salt=SimpleNamespace(history_length=3),
    )

    advantages, _ = compute_grpo_salt_progpo_lata_outcome_advantage(
        rewards,
        mask,
        np.array(["task", "task"]),
        progress_scores=np.array([0.2, 0.8]),
        outcome_rewards=np.array([0.0, 1.0]),
        salt_steps=traces,
        salt_root_keys=np.array(["same-prompt", "same-prompt"], dtype=object),
        config=config,
    )

    torch.testing.assert_close(advantages[:, :2], torch.zeros((2, 2)))
    assert torch.all(advantages[0, 4:6] < 0)
    assert torch.all(advantages[1, 4:6] > 0)
    assert torch.equal(advantages[:, 2:4], torch.zeros((2, 2)))


def test_all_fail_group_bypasses_salt_and_matches_progpo_lata():
    rewards = torch.zeros((2, 4))
    mask = torch.ones((2, 4), dtype=torch.long)
    progress = np.array([0.1, 0.9])
    outcomes = np.array([0.0, 0.0])
    groups = np.array(["task", "task"])
    traces = np.empty(2, dtype=object)
    traces[0] = [_step(0, 4, "same", "same")]
    traces[1] = [_step(0, 4, "same", "same")]
    config = SimpleNamespace(
        turn_discount=SimpleNamespace(alpha=1.0),
        progpo=SimpleNamespace(
            tau_reward=1e-3,
            tau_progress=1e-4,
            lambda_aux=0.3,
            lambda_fixed=False,
        ),
        salt=SimpleNamespace(history_length=3),
    )

    baseline, _ = compute_grpo_progpo_lata_outcome_advantage(
        rewards,
        mask,
        groups,
        progress_scores=progress,
        outcome_rewards=outcomes,
        config=config,
    )
    combined, _ = compute_grpo_salt_progpo_lata_outcome_advantage(
        rewards,
        mask,
        groups,
        progress_scores=progress,
        outcome_rewards=outcomes,
        salt_steps=traces,
        config=config,
    )

    torch.testing.assert_close(combined, baseline)


def test_turn_lata_is_invariant_to_observation_token_gap():
    mask = torch.zeros((2, 12), dtype=torch.long)
    mask[0, [0, 2]] = 1
    mask[1, [0, 10]] = 1
    traces = np.empty(2, dtype=object)
    traces[0] = [
        _step(0, 1, "a", "o"),
        _step(2, 3, "b", "p"),
    ]
    traces[1] = [
        _step(0, 1, "a", "o"),
        _step(10, 11, "b", "p"),
    ]

    weights, diagnostics = compute_lata_turn_weights(mask, traces, alpha=1.05)

    assert weights[0, 0].item() == pytest.approx(weights[1, 0].item())
    assert weights[0, 2].item() == pytest.approx(weights[1, 10].item())
    assert weights[0, 0].item() / weights[0, 2].item() == pytest.approx(1.05)
    assert diagnostics["lata/uncovered_token_rate"] == 0.0


def test_turn_lata_does_not_exponentiate_with_tokens_inside_one_turn():
    mask = torch.ones((1, 1000), dtype=torch.long)
    traces = np.empty(1, dtype=object)
    traces[0] = [_step(0, 1000, "one-turn", "done")]

    weights, _ = compute_lata_turn_weights(mask, traces, alpha=1.05)

    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_turn_lata_zero_policy_tokens_are_finite_and_zero():
    mask = torch.zeros((1, 4), dtype=torch.long)
    traces = np.empty(1, dtype=object)
    traces[0] = []

    weights, diagnostics = compute_lata_turn_weights(mask, traces, alpha=1.05)

    assert torch.isfinite(weights).all()
    assert torch.equal(weights, torch.zeros_like(weights))
    assert diagnostics["lata/metadata_fallback_samples"] == 1


def test_all_fail_progpo_uses_real_turn_lata_weights():
    rewards = torch.zeros((2, 2))
    mask = torch.ones((2, 2), dtype=torch.long)
    traces = np.empty(2, dtype=object)
    for sample_idx in range(2):
        traces[sample_idx] = [
            _step(0, 1, "first", "state-1"),
            _step(1, 2, "second", "state-2"),
        ]

    advantages, _ = compute_grpo_salt_progpo_lata_outcome_advantage(
        rewards,
        mask,
        np.array(["task", "task"]),
        progress_scores=np.array([0.1, 0.9]),
        outcome_rewards=np.array([0.0, 0.0]),
        salt_steps=traces,
    )

    assert abs(advantages[0, 0].item() / advantages[0, 1].item()) == pytest.approx(
        1.05
    )
    assert abs(advantages[1, 0].item() / advantages[1, 1].item()) == pytest.approx(
        1.05
    )
