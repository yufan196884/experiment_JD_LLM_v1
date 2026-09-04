from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_PPO_CLIP_EPS = 0.2


# =============================================================================
# Completion token log-probability scoring
# =============================================================================


def _validate_completion_scoring_inputs(
    *,
    sequences: torch.Tensor,
    full_attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    prompt_length: int,
) -> tuple[int, int]:
    """Validate the token-alignment tensors used for completion scoring."""
    if sequences.ndim != 2:
        raise ValueError(
            "sequences must have shape [G, P + C], "
            f"got {tuple(sequences.shape)}."
        )

    if full_attention_mask.shape != sequences.shape:
        raise ValueError(
            "full_attention_mask must have the same shape as sequences: "
            f"sequences={tuple(sequences.shape)}, "
            f"mask={tuple(full_attention_mask.shape)}."
        )

    if completion_ids.ndim != 2:
        raise ValueError(
            "completion_ids must have shape [G, C], "
            f"got {tuple(completion_ids.shape)}."
        )

    if completion_mask.shape != completion_ids.shape:
        raise ValueError(
            "completion_mask must have the same shape as completion_ids: "
            f"completion_ids={tuple(completion_ids.shape)}, "
            f"completion_mask={tuple(completion_mask.shape)}."
        )

    if prompt_length <= 0:
        raise ValueError(
            f"prompt_length must be positive, got {prompt_length}."
        )

    num_sequences, sequence_length = sequences.shape

    if prompt_length >= sequence_length:
        raise ValueError(
            "prompt_length must leave at least one completion token: "
            f"prompt_length={prompt_length}, "
            f"sequence_length={sequence_length}."
        )

    completion_length = sequence_length - prompt_length

    expected_completion_shape = (
        num_sequences,
        completion_length,
    )

    if tuple(completion_ids.shape) != expected_completion_shape:
        raise ValueError(
            "completion_ids shape is inconsistent with sequences and "
            "prompt_length: "
            f"expected {expected_completion_shape}, "
            f"got {tuple(completion_ids.shape)}."
        )

    sequence_completion_ids = sequences[
        :,
        prompt_length:,
    ]

    if not torch.equal(
        sequence_completion_ids,
        completion_ids,
    ):
        raise ValueError(
            "completion_ids must exactly equal sequences[:, prompt_length:]."
        )

    if sequences.device != full_attention_mask.device:
        raise ValueError(
            "sequences and full_attention_mask must be on the same device."
        )

    if sequences.device != completion_ids.device:
        raise ValueError(
            "sequences and completion_ids must be on the same device."
        )

    if sequences.device != completion_mask.device:
        raise ValueError(
            "sequences and completion_mask must be on the same device."
        )

    return num_sequences, completion_length


def completion_token_logprobs(
    model: nn.Module,
    *,
    sequences: torch.Tensor,
    full_attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    prompt_length: int,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Score sampled completion tokens under a causal language model.

    Let:
        G = number of rollouts
        P = prompt length
        C = padded completion length

    Inputs:
        sequences:             [G, P + C]
        full_attention_mask:   [G, P + C]
        completion_ids:        [G, C]
        completion_mask:       [G, C]

    Returns:
        token_logprobs:        [G, C]

    A causal LM's logits at position j predict token j + 1. Therefore the
    first completion token, at absolute position P, is predicted by logits at
    position P - 1. The correct alignment is:

        target_ids        = sequences[:, P:]
        completion_logits = logits[:, P - 1 : -1, :]

    The caller decides whether this is old-policy scoring (`no_grad` + detach)
    or current-policy scoring (gradients enabled).
    """
    num_sequences, completion_length = _validate_completion_scoring_inputs(
        sequences=sequences,
        full_attention_mask=full_attention_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        prompt_length=prompt_length,
    )

    if micro_batch_size is None:
        micro_batch_size = num_sequences

    if micro_batch_size <= 0:
        raise ValueError(
            "micro_batch_size must be positive or None, "
            f"got {micro_batch_size}."
        )

    scored_chunks: list[torch.Tensor] = []

    for start in range(0, num_sequences, micro_batch_size):
        end = min(start + micro_batch_size, num_sequences)

        chunk_sequences = sequences[start:end]
        chunk_attention_mask = full_attention_mask[start:end]
        chunk_target_ids = completion_ids[start:end]
        chunk_completion_mask = completion_mask[start:end].bool()

        outputs = model(
            input_ids=chunk_sequences,
            attention_mask=chunk_attention_mask,
            use_cache=False,
        )

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise TypeError(
                "Model forward output does not expose a `.logits` tensor."
            )

        if logits.ndim != 3:
            raise ValueError(
                "Model logits must have shape [G, P + C, vocab], "
                f"got {tuple(logits.shape)}."
            )

        if logits.shape[:2] != chunk_sequences.shape:
            raise ValueError(
                "Model logits have unexpected batch/sequence dimensions: "
                f"logits={tuple(logits.shape)}, "
                f"input_ids={tuple(chunk_sequences.shape)}."
            )

        completion_logits = logits[:, prompt_length - 1 : -1, :]

        if completion_logits.shape[1] != completion_length:
            raise RuntimeError(
                "Causal-LM completion alignment produced the wrong number "
                "of token positions: "
                f"expected {completion_length}, "
                f"got {completion_logits.shape[1]}."
            )

        # -cross_entropy(logits, target) == log p(target).
        #
        # This avoids explicitly materializing a second
        # [batch, C, vocab] log_softmax tensor.
        token_logprobs = -F.cross_entropy(
            completion_logits.transpose(1, 2),
            chunk_target_ids,
            reduction="none",
        )

        token_logprobs = token_logprobs.masked_fill(
            ~chunk_completion_mask,
            0.0,
        )

        scored_chunks.append(token_logprobs)

    return torch.cat(scored_chunks, dim=0)


# =============================================================================
# Per-objective PPO / GRPO clipped surrogate losses
# =============================================================================


def _validate_per_objective_ppo_inputs(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float,
) -> int:
    """
    Validate generic tensor shapes for vector-valued PPO losses.

    Expected shapes:

        current_logprobs: [..., T]
        old_logprobs:     [..., T]
        completion_mask:  [..., T]
        advantages:       [..., M]

    where:
        T = padded completion-token length
        M = number of reward objectives

    Examples:

        [G, T], [G, M]
            one prompt/preference group;

        [B, G, T], [B, G, M]
            a batch of B prompt/preference groups.
    """
    if current_logprobs.ndim < 2:
        raise ValueError(
            "current_logprobs must have shape [..., T] with at least one "
            "rollout dimension, "
            f"got {tuple(current_logprobs.shape)}."
        )

    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError(
            "old_logprobs must have the same shape as current_logprobs: "
            f"current={tuple(current_logprobs.shape)}, "
            f"old={tuple(old_logprobs.shape)}."
        )

    if completion_mask.shape != current_logprobs.shape:
        raise ValueError(
            "completion_mask must have the same shape as token logprobs: "
            f"logprobs={tuple(current_logprobs.shape)}, "
            f"mask={tuple(completion_mask.shape)}."
        )

    if advantages.ndim != current_logprobs.ndim:
        raise ValueError(
            "advantages must have shape [..., M] with the same leading "
            "rollout dimensions as token logprobs [..., T]: "
            f"logprobs={tuple(current_logprobs.shape)}, "
            f"advantages={tuple(advantages.shape)}."
        )

    if advantages.shape[:-1] != current_logprobs.shape[:-1]:
        raise ValueError(
            "advantages and token logprobs must have identical leading "
            "rollout dimensions: "
            f"logprobs={tuple(current_logprobs.shape)}, "
            f"advantages={tuple(advantages.shape)}."
        )

    num_objectives = advantages.shape[-1]

    if num_objectives < 1:
        raise ValueError(
            "Expected at least one reward objective."
        )

    if current_logprobs.shape[-1] < 1:
        raise ValueError(
            "Expected at least one completion token position."
        )

    if not (0.0 < clip_eps < 1.0):
        raise ValueError(
            "clip_eps must satisfy 0 < clip_eps < 1, "
            f"got {clip_eps}."
        )

    if completion_mask.dtype != torch.bool:
        raise TypeError(
            "completion_mask must be a boolean tensor, "
            f"got dtype={completion_mask.dtype}."
        )

    if not torch.is_floating_point(current_logprobs):
        raise TypeError(
            "current_logprobs must be floating point."
        )

    if not torch.is_floating_point(old_logprobs):
        raise TypeError(
            "old_logprobs must be floating point."
        )

    if current_logprobs.device != old_logprobs.device:
        raise ValueError(
            "current_logprobs and old_logprobs must be on the same device."
        )

    if current_logprobs.device != advantages.device:
        raise ValueError(
            "advantages must be on the same device as token logprobs."
        )

    if current_logprobs.device != completion_mask.device:
        raise ValueError(
            "completion_mask must be on the same device as token logprobs."
        )

    # PPO must compare the current policy against a fixed behaviour policy.
    #
    # old_logprobs should have been computed under torch.no_grad() and/or
    # explicitly detached immediately after rollout collection.
    if old_logprobs.requires_grad:
        raise ValueError(
            "old_logprobs must be fixed/detached. PPO epochs must compare "
            "against the same stored old-policy log-probabilities."
        )

    # The reward-derived advantages are fixed training targets. We do not
    # backpropagate through reward calculation or GDPO normalization.
    if advantages.requires_grad:
        raise ValueError(
            "advantages must be treated as fixed Monte-Carlo training "
            "targets and must not require gradients."
        )

    valid_token_counts = completion_mask.sum(
        dim=-1
    )

    if bool(
        (valid_token_counts == 0).any()
    ):
        raise ValueError(
            "Every rollout must contain at least one valid completion token."
        )

    valid_current = current_logprobs.masked_select(
        completion_mask
    )
    valid_old = old_logprobs.masked_select(
        completion_mask
    )

    if not bool(
        torch.isfinite(valid_current).all()
    ):
        raise ValueError(
            "current_logprobs contains NaN or infinite values on valid tokens."
        )

    if not bool(
        torch.isfinite(valid_old).all()
    ):
        raise ValueError(
            "old_logprobs contains NaN or infinite values on valid tokens."
        )

    if not bool(
        torch.isfinite(advantages).all()
    ):
        raise ValueError(
            "advantages contains NaN or infinite values."
        )

    return num_objectives


def per_objective_ppo_losses(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float = DEFAULT_PPO_CLIP_EPS,
) -> list[torch.Tensor]:
    """
    Compute one independent PPO clipped surrogate loss per reward objective.

    This function is intentionally task- and model-agnostic.

    The number of objectives M is inferred from:

        advantages.shape[-1]

    Nothing is hard-coded for Maze or M=4.

    -----------------------------------------------------------------------
    Tensor layout
    -----------------------------------------------------------------------

        current_logprobs: [..., T]
        old_logprobs:     [..., T]
        completion_mask:  [..., T]
        advantages:       [..., M]

    The leading dimensions identify rollouts.

    Typical examples:

        current_logprobs: [G, T]
        advantages:       [G, M]

    for one prompt/preference group, or:

        current_logprobs: [B, G, T]
        advantages:       [B, G, M]

    for B prompt/preference groups.

    -----------------------------------------------------------------------
    PPO ratio
    -----------------------------------------------------------------------

    For rollout g and completion token t:

        rho[g, t]
            = exp(
                current_logprob[g, t]
                - old_logprob[g, t]
            )

    The same probability ratio is shared by every reward objective.

    -----------------------------------------------------------------------
    Per-objective clipped surrogate
    -----------------------------------------------------------------------

    For objective i:

        s[g, t, i]
            = min(
                rho[g, t] * A[g, i],
                clip(
                    rho[g, t],
                    1 - eps,
                    1 + eps,
                ) * A[g, i],
            )

    A[g, i] is a rollout-level scalar.

    It is therefore broadcast across every valid completion token belonging
    to rollout g.

    -----------------------------------------------------------------------
    Response normalization
    -----------------------------------------------------------------------

    We first average over the valid completion tokens inside each response:

        response_score[g, i]
            =
                sum_t mask[g, t] * s[g, t,i]
                --------------------------------
                sum_t mask[g, t]

    We then average response scores over all rollout/group dimensions:

        S_i
            = mean_g response_score[g, i]

    Finally:

        L_i = -S_i

    because PPO maximizes the surrogate objective while PyTorch optimizers
    minimize losses.

    -----------------------------------------------------------------------
    IMPORTANT: no scalarization
    -----------------------------------------------------------------------

    This function deliberately does NOT calculate:

        sum_i omega[i] * L_i

    and does NOT perform:

        total_loss.backward()

    The M objective losses must remain separate so that the next stage can
    calculate:

        g_i = grad(L_i)

    and construct the Jacobian:

        J = [
            g_1^T
            ...
            g_M^T
        ]

    for Jacobian Descent / UPGrad.

    Returns:
        Python list of length M.

        Every element is a 0-dimensional scalar tensor:

            len(losses) == M
            loss.ndim == 0

        For the current Maze task, M=4:

            losses = [
                completion_loss,
                gold_loss,
                diamond_loss,
                lava_loss,
            ]

        but this function supports arbitrary M >= 1.
    """
    num_objectives = _validate_per_objective_ppo_inputs(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        completion_mask=completion_mask,
        clip_eps=clip_eps,
    )

    # -----------------------------------------------------------------------
    # 1. Use float32 for PPO-ratio arithmetic when model token log-probs are
    #    float16 / bfloat16.
    #
    # Casting current_logprobs to float32 does NOT detach it, so the autograd
    # path back into the current model remains intact.
    # -----------------------------------------------------------------------

    if current_logprobs.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        working_current = current_logprobs.float()
    else:
        working_current = current_logprobs

    working_old = old_logprobs.to(
        dtype=working_current.dtype
    )

    working_advantages = advantages.to(
        dtype=working_current.dtype
    )

    working_mask = completion_mask.to(
        dtype=working_current.dtype
    )

    # -----------------------------------------------------------------------
    # 2. Compute token-level PPO ratios.
    #
    #       rho = pi_theta / pi_old
    #
    #           = exp(
    #                 log pi_theta
    #                 - log pi_old
    #             )
    #
    # Shapes:
    #
    #       current_logprobs : [..., T]
    #       old_logprobs     : [..., T]
    #       ratio            : [..., T]
    #
    # -----------------------------------------------------------------------

    log_ratio = (
        working_current
        - working_old
    )

    ratio = torch.exp(
        log_ratio
    )

    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_eps,
        max=1.0 + clip_eps,
    )

    # -----------------------------------------------------------------------
    # 3. Insert the objective dimension.
    #
    # Before:
    #
    #       ratio       [..., T]
    #       advantages  [..., M]
    #
    # After:
    #
    #       ratio       [..., 1, T]
    #       advantages  [..., M, 1]
    #
    # Broadcasting therefore gives:
    #
    #       surrogate   [..., M, T]
    #
    # -----------------------------------------------------------------------

    ratio = ratio.unsqueeze(
        -2
    )

    clipped_ratio = clipped_ratio.unsqueeze(
        -2
    )

    working_mask = working_mask.unsqueeze(
        -2
    )

    broadcast_advantages = working_advantages.unsqueeze(
        -1
    )

    # -----------------------------------------------------------------------
    # 4. Construct the unclipped and clipped PPO surrogate for every
    #    objective independently.
    #
    #       rho * A_i
    #
    # and:
    #
    #       clip(rho) * A_i
    #
    # -----------------------------------------------------------------------

    unclipped_surrogate = (
        ratio
        * broadcast_advantages
    )

    clipped_surrogate = (
        clipped_ratio
        * broadcast_advantages
    )

    # torch.minimum correctly implements PPO's sign-dependent clipping
    # behaviour:
    #
    #   positive advantage:
    #       excessive probability increases are clipped;
    #
    #   negative advantage:
    #       excessive probability decreases are clipped.
    token_surrogate = torch.minimum(
        unclipped_surrogate,
        clipped_surrogate,
    )

    # -----------------------------------------------------------------------
    # 5. Response-length normalization.
    #
    # Every rollout should have equal weight regardless of whether its model
    # completion contains, for example, 40 tokens or 200 tokens.
    #
    # First calculate:
    #
    #       sum_t m[g,t] s[g,t,i]
    #       ----------------------
    #          sum_t m[g,t]
    #
    # Shapes:
    #
    #       token_surrogate       [..., M, T]
    #       per_rollout_surrogate [..., M]
    #
    # -----------------------------------------------------------------------

    valid_token_counts = completion_mask.sum(
        dim=-1,
    ).to(
        dtype=working_current.dtype
    )

    per_rollout_surrogate = (
        (
            token_surrogate
            * working_mask
        ).sum(
            dim=-1
        )
        / valid_token_counts.unsqueeze(
            -1
        )
    )

    # -----------------------------------------------------------------------
    # 6. Average equally over all rollout/group dimensions.
    #
    # For [G, M]:
    #
    #       [G, M] -> [M]
    #
    # For [B, G, M]:
    #
    #       [B, G, M] -> [M]
    #
    # More generally, flatten every leading dimension while preserving the
    # final objective dimension M.
    # -----------------------------------------------------------------------

    per_objective_scores = per_rollout_surrogate.reshape(
        -1,
        num_objectives,
    ).mean(
        dim=0
    )

    # -----------------------------------------------------------------------
    # 7. PPO's surrogate is maximized.
    #
    # Jacobian Descent / AdamW are formulated in terms of losses that are
    # minimized, so negate each objective score.
    #
    #       L_i = -S_i
    #
    # -----------------------------------------------------------------------

    loss_values = -per_objective_scores

    # Convert the [M] tensor into M independent scalar tensors.
    #
    # For Maze:
    #
    #     [
    #         completion_loss,
    #         gold_loss,
    #         diamond_loss,
    #         lava_loss,
    #     ]
    #
    # Another dataset may have any number M of objectives.
    losses = list(
        loss_values.unbind(
            dim=0
        )
    )

    # -----------------------------------------------------------------------
    # 8. Architectural invariants required by the next Jacobian-Descent
    #    stage.
    # -----------------------------------------------------------------------

    if len(losses) != num_objectives:
        raise RuntimeError(
            "Internal error: PPO loss count does not match objective count."
        )

    if any(
        loss.ndim != 0
        for loss in losses
    ):
        raise RuntimeError(
            "Internal error: every per-objective PPO loss must be scalar."
        )

    return losses