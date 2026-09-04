from __future__ import annotations

import torch


# =============================================================================
# GDPO-style reward-decoupled advantage estimation
# =============================================================================

DEFAULT_ADVANTAGE_EPS = 1e-4


def compute_reward_decoupled_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = DEFAULT_ADVANTAGE_EPS,
) -> torch.Tensor:
    """
    Compute group-relative advantages independently for every reward objective.

    This implements only the reward-decoupled normalization step from GDPO.

    We intentionally DO NOT:
      - sum advantages across reward objectives;
      - apply preference weights;
      - perform GDPO's final batch-wise advantage normalization;
      - scalarize the reward or advantage vectors.

    Expected tensor layout:

        [..., G, M]

    where:
        G = number of independent rollouts for one conditioning group
        M = number of reward objectives

    Examples:

        [G, M]
            One prompt/preference group.

        [B, G, M]
            A batch of B prompt/preference groups.

    Normalization is performed along the rollout dimension G independently
    for every reward objective M:

        A[g, i] = (R[g, i] - mean_g(R[:, i]))
                  / (std_g(R[:, i]) + eps)

    The returned tensor has the same shape as the input tensor.

    Args:
        rewards:
            Reward tensor with shape [..., G, M].

        eps:
            Numerical stability constant added to each per-objective standard
            deviation. The default 1e-4 matches the official GDPO
            implementation.

    Returns:
        Tensor of reward-decoupled advantages with shape [..., G, M].

    Raises:
        ValueError:
            If the tensor has fewer than two dimensions, contains fewer than
            two rollouts, has zero reward objectives, contains non-finite
            values, or eps is not positive.
    """
    if rewards.ndim < 2:
        raise ValueError(
            "Expected rewards with shape [..., G, M], "
            f"but received shape {tuple(rewards.shape)}."
        )

    num_rollouts = rewards.shape[-2]
    num_rewards = rewards.shape[-1]

    if num_rollouts < 2:
        raise ValueError(
            "GDPO-style group-relative normalization requires at least "
            f"two rollouts per group, but received G={num_rollouts}."
        )

    if num_rewards < 1:
        raise ValueError(
            "Expected at least one reward objective."
        )

    if eps <= 0.0:
        raise ValueError(
            f"eps must be positive, but received {eps}."
        )

    if not bool(
        torch.isfinite(rewards).all()
    ):
        raise ValueError(
            "Reward tensor contains NaN or infinite values."
        )

    # Reward statistics should not be computed in float16 / bfloat16.
    # Convert low-precision or integral rewards to float32 while preserving
    # float64 inputs when explicitly supplied.
    if (
        not torch.is_floating_point(rewards)
        or rewards.dtype in (
            torch.float16,
            torch.bfloat16,
        )
    ):
        working_rewards = rewards.to(
            dtype=torch.float32
        )
    else:
        working_rewards = rewards

    # The final two dimensions are:
    #
    #     [..., G, M]
    #           ^
    #           |
    #       normalize here
    #
    # PyTorch's correction=1 gives the sample standard deviation and matches
    # the default torch.std(...) behaviour used by the official GDPO code.
    reward_std, reward_mean = torch.std_mean(
        working_rewards,
        dim=-2,
        correction=1,
        keepdim=True,
    )

    centered_rewards = (
        working_rewards
        - reward_mean
    )

    advantages = (
        centered_rewards
        / (reward_std + eps)
    )

    # If an objective gives exactly the same reward to every rollout, then
    # there is no group-relative information for that objective.
    #
    # Its complete advantage stream should therefore be zero.
    zero_variance = (
        reward_std == 0
    )

    advantages = torch.where(
        zero_variance.expand_as(
            advantages
        ),
        torch.zeros_like(
            advantages
        ),
        advantages,
    )

    return advantages