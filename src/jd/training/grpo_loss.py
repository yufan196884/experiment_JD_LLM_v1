from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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

        # -cross_entropy(logits, target) == log p(target). This avoids
        # explicitly materializing a second [batch, C, vocab] log_softmax.
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
