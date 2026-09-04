from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase


ChatMessage = Mapping[str, Any]


@dataclass
class GenerationBatch:
    """
    Several sampled completions for one chat prompt.

    Let:
        G = number of sampled completions
        P = prompt length
        C = padded generated-completion length

    Shapes:
        prompt_input_ids:      [1, P]
        prompt_attention_mask: [1, P]
        sequences:             [G, P + C]
        full_attention_mask:   [G, P + C]
        completion_ids:        [G, C]
        completion_mask:       [G, C]
    """

    texts: list[str]

    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor

    sequences: torch.Tensor
    full_attention_mask: torch.Tensor

    completion_ids: torch.Tensor
    completion_mask: torch.Tensor

    @property
    def prompt_length(self) -> int:
        return int(
            self.prompt_input_ids.shape[1]
        )

    @property
    def num_generations(self) -> int:
        return int(
            self.sequences.shape[0]
        )


def _model_device(
    model: nn.Module,
) -> torch.device:
    try:
        return next(
            model.parameters()
        ).device
    except StopIteration as exc:
        raise ValueError(
            "Model has no parameters."
        ) from exc


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    *,
    enable_thinking: bool = False,
) -> str:
    """
    Render chat messages using the checkpoint's own chat template.

    This is important because Qwen chat control tokens should come from the
    official tokenizer/template rather than being manually constructed.
    """
    if not messages:
        raise ValueError(
            "messages must contain at least one chat message."
        )

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    if not isinstance(
        rendered,
        str,
    ):
        raise TypeError(
            "tokenizer.apply_chat_template() "
            "did not return a string."
        )

    return rendered


def _normalize_eos_ids(
    eos_token_id: int | Sequence[int] | None,
) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()

    if isinstance(
        eos_token_id,
        int,
    ):
        return (
            eos_token_id,
        )

    return tuple(
        int(token_id)
        for token_id in eos_token_id
    )


def _build_completion_mask(
    completion_ids: torch.Tensor,
    *,
    eos_token_id: int | Sequence[int] | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """
    Include sampled tokens through the first EOS and mask padding after it.

    EOS itself remains part of the sampled action and should therefore be
    eligible to participate in a policy-gradient loss.
    """
    if completion_ids.ndim != 2:
        raise ValueError(
            "completion_ids must have shape [G, C], "
            f"got {tuple(completion_ids.shape)}."
        )

    eos_ids = _normalize_eos_ids(
        eos_token_id
    )

    mask = torch.ones_like(
        completion_ids,
        dtype=torch.bool,
    )

    for row_index in range(
        completion_ids.shape[0]
    ):
        row = completion_ids[
            row_index
        ]

        eos_positions: list[int] = []

        for eos_id in eos_ids:
            positions = torch.nonzero(
                row == eos_id,
                as_tuple=False,
            ).flatten()

            if positions.numel() > 0:
                eos_positions.append(
                    int(
                        positions[0].item()
                    )
                )

        if eos_positions:
            first_eos = min(
                eos_positions
            )

            mask[
                row_index,
                first_eos + 1 :,
            ] = False

            continue

        # If a row has no EOS but was padded to the length of a longer
        # generation, exclude the padding itself.
        if pad_token_id is not None:
            pad_positions = torch.nonzero(
                row == pad_token_id,
                as_tuple=False,
            ).flatten()

            if (
                pad_positions.numel()
                > 0
            ):
                first_pad = int(
                    pad_positions[
                        0
                    ].item()
                )

                mask[
                    row_index,
                    first_pad:,
                ] = False

    return mask


@torch.inference_mode()
def generate_completions(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    *,
    num_generations: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 20,
    enable_thinking: bool = False,
) -> GenerationBatch:
    """
    Sample `num_generations` completions for one prompt.

    This low-level helper uses Hugging Face Transformers directly. That is the
    simplest architecture for the initial on-policy GRPO/JD experiment because
    the rollout policy and the trainable policy are the same in-process model.

    Generation is intentionally inference-only. During the GRPO update, token
    log-probabilities should be recomputed with gradients enabled.
    """
    if num_generations <= 0:
        raise ValueError(
            "num_generations must be positive."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be positive."
        )

    if temperature <= 0.0:
        raise ValueError(
            "temperature must be > 0 because "
            "GRPO requires stochastic sampling."
        )

    if not (
        0.0
        < top_p
        <= 1.0
    ):
        raise ValueError(
            "top_p must lie in (0, 1]."
        )

    if top_k < 0:
        raise ValueError(
            "top_k must be non-negative."
        )

    device = _model_device(
        model
    )

    rendered_prompt = render_chat_prompt(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
    )

    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_input_ids = encoded[
        "input_ids"
    ].to(
        device
    )

    prompt_attention_mask = encoded[
        "attention_mask"
    ].to(
        device
    )

    prompt_length = int(
        prompt_input_ids.shape[1]
    )

    was_training = model.training
    model.eval()

    try:
        generated = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            do_sample=True,
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        if was_training:
            model.train()

    sequences = (
        generated.sequences
        if hasattr(
            generated,
            "sequences",
        )
        else generated
    )

    if sequences.ndim != 2:
        raise ValueError(
            "Expected generated sequences with shape [G, P + C], "
            f"got {tuple(sequences.shape)}."
        )

    completion_ids = sequences[
        :,
        prompt_length:,
    ]

    completion_mask = _build_completion_mask(
        completion_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    repeated_prompt_mask = (
        prompt_attention_mask.expand(
            sequences.shape[0],
            -1,
        )
    )

    full_attention_mask = torch.cat(
        [
            repeated_prompt_mask,
            completion_mask.to(
                dtype=prompt_attention_mask.dtype
            ),
        ],
        dim=1,
    )

    texts = [
        tokenizer.decode(
            completion_ids[
                index
            ][
                completion_mask[
                    index
                ]
            ],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for index in range(
            completion_ids.shape[0]
        )
    ]

    return GenerationBatch(
        texts=texts,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        sequences=sequences,
        full_attention_mask=full_attention_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
    )
