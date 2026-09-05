from __future__ import annotations

import torch


LENGTH_OBJECTIVE_NAME = "response_length"

DEFAULT_LENGTH_FREE_TOKENS = 512
DEFAULT_LENGTH_SCALE_TOKENS = 512.0
DEFAULT_LENGTH_LOG_COEFFICIENT = 0.3


def compute_log_length_penalty(
    lengths: torch.Tensor,
    *,
    free_tokens: int = DEFAULT_LENGTH_FREE_TOKENS,
    scale_tokens: float = DEFAULT_LENGTH_SCALE_TOKENS,
    coefficient: float = DEFAULT_LENGTH_LOG_COEFFICIENT,
) -> torch.Tensor:
    """
    Shifted logarithmic response-length reward.

    For L <= free_tokens:

        r(L) = 0

    For L > free_tokens:

        r(L)
            = -coefficient
              * log(
                    1
                    + (L - free_tokens) / scale_tokens
                )

    This is intended for settings such as the current Maze experiment,
    where excessively long outputs should be discouraged even if the
    final answer fails to parse.

    The logarithmic tail does not hard-saturate, so an 8k response,
    16k response, etc. remain distinguishable.
    """
    if free_tokens < 0:
        raise ValueError(
            f"free_tokens must be non-negative, got {free_tokens}."
        )

    if scale_tokens <= 0.0:
        raise ValueError(
            f"scale_tokens must be positive, got {scale_tokens}."
        )

    if coefficient < 0.0:
        raise ValueError(
            f"coefficient must be non-negative, got {coefficient}."
        )

    if not torch.is_tensor(lengths):
        raise TypeError("lengths must be a torch.Tensor.")

    if bool((lengths < 0).any()):
        raise ValueError("lengths must be non-negative.")

    working_lengths = lengths.to(torch.float32)

    excess = (
        working_lengths
        - float(free_tokens)
    ).clamp_min(0.0)

    return (
        -float(coefficient)
        * torch.log1p(
            excess / float(scale_tokens)
        )
    )


def compute_correctness_conditioned_length_reward(
    lengths: torch.Tensor,
    correctness: torch.Tensor,
    *,
    free_tokens: int,
    scale_tokens: float,
    coefficient: float = DEFAULT_LENGTH_LOG_COEFFICIENT,
) -> torch.Tensor:
    """
    Future math/Olympiad version.

    Incorrect responses receive 0.

    Correct responses receive:

        1                                      if L <= free_tokens

        (1 + (L-free_tokens)/scale_tokens)
            ** (-coefficient)                  otherwise

    Equivalently, for correct responses this is:

        exp(compute_log_length_penalty(...))

    Therefore:

        incorrect               -> 0
        short + correct         -> 1
        long + correct          -> between 0 and 1

    A correct response is never ranked below an incorrect response
    merely because it is long.
    """
    if correctness.shape != lengths.shape:
        raise ValueError(
            "correctness and lengths must have identical shapes: "
            f"lengths={tuple(lengths.shape)}, "
            f"correctness={tuple(correctness.shape)}."
        )

    penalty = compute_log_length_penalty(
        lengths,
        free_tokens=free_tokens,
        scale_tokens=scale_tokens,
        coefficient=coefficient,
    )

    correct_reward = torch.exp(
        penalty
    )

    return torch.where(
        correctness.bool(),
        correct_reward,
        torch.zeros_like(correct_reward),
    )