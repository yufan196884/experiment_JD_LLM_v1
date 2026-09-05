import math

import torch

from jd.rewards.length import (
    compute_log_length_penalty,
    compute_correctness_conditioned_length_reward,
)


def test_no_penalty_through_free_budget():
    lengths = torch.tensor(
        [0, 100, 511, 512]
    )

    rewards = compute_log_length_penalty(
        lengths,
        free_tokens=512,
        scale_tokens=512,
        coefficient=0.3,
    )

    assert torch.equal(
        rewards,
        torch.zeros_like(rewards),
    )


def test_log_penalty_keeps_decreasing_for_long_outputs():
    lengths = torch.tensor(
        [513, 1024, 8192, 16384]
    )

    rewards = compute_log_length_penalty(
        lengths,
        free_tokens=512,
        scale_tokens=512,
        coefficient=0.3,
    )

    assert rewards[0] > rewards[1]
    assert rewards[1] > rewards[2]
    assert rewards[2] > rewards[3]


def test_log_penalty_matches_formula():
    lengths = torch.tensor([1024])

    reward = compute_log_length_penalty(
        lengths,
        free_tokens=512,
        scale_tokens=512,
        coefficient=0.3,
    )

    expected = -0.3 * math.log1p(
        (1024 - 512) / 512
    )

    assert torch.allclose(
        reward,
        torch.tensor([expected]),
    )


def test_math_conditioned_reward_never_prefers_incorrect():
    lengths = torch.tensor(
        [512, 8192, 80000]
    )

    correctness = torch.tensor(
        [True, True, False]
    )

    rewards = compute_correctness_conditioned_length_reward(
        lengths,
        correctness,
        free_tokens=512,
        scale_tokens=512,
        coefficient=0.3,
    )

    assert rewards[0] == 1.0
    assert 0.0 < rewards[1] < 1.0
    assert rewards[2] == 0.0