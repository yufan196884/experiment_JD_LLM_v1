import torch

from jd.training.advantages import (
    compute_reward_decoupled_advantages,
)


def test_advantages_support_five_objectives():
    rewards = torch.tensor(
        [
            [1.0, 0.1, 0.2, 1.0,  0.0],
            [1.0, 0.5, 0.4, 0.8, -0.2],
            [0.0, 0.2, 0.8, 0.4, -0.8],
            [1.0, 0.9, 0.1, 1.0, -1.0],
        ]
    )

    advantages = (
        compute_reward_decoupled_advantages(
            rewards
        )
    )

    assert advantages.shape == (
        4,
        5,
    )

    assert torch.allclose(
        advantages.mean(dim=0),
        torch.zeros(5),
        atol=1e-5,
    )

def test_constant_length_reward_has_zero_advantages():
    rewards = torch.tensor(
        [
            [1.0, 0.1, 0.2, 1.0, -0.5],
            [0.0, 0.4, 0.3, 0.8, -0.5],
            [1.0, 0.2, 0.7, 0.9, -0.5],
            [1.0, 0.8, 0.1, 1.0, -0.5],
        ]
    )

    advantages = compute_reward_decoupled_advantages(
        rewards
    )

    assert torch.equal(
        advantages[:, 4],
        torch.zeros(4),
    )
