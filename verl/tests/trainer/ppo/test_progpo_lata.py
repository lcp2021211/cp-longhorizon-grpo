import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_progpo_lata_outcome_advantage,
    compute_progpo_group_scores,
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
