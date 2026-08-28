from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

DTypeName = Literal[
    "auto",
    "float32",
    "float16",
    "bfloat16",
]


@dataclass
class ModelBundle:
    """Objects required by the model/generation/training layers."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def resolve_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Resolve a device for the initial single-process implementation.

    We deliberately avoid `device_map="auto"` here. For RL/Jacobian training,
    it is simpler and safer initially to keep the trainable model on one
    explicit device.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_dtype(
    dtype: DTypeName | torch.dtype = "auto",
    *,
    device: torch.device,
) -> torch.dtype:
    """Choose a practical dtype for the selected device."""
    if isinstance(dtype, torch.dtype):
        return dtype

    explicit = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype != "auto":
        try:
            return explicit[dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unknown dtype {dtype!r}. "
                f"Expected one of {('auto', *explicit.keys())}."
            ) from exc

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def freeze_vision_parameters(
    model: nn.Module,
) -> int:
    """
    Freeze the vision tower of the multimodal Qwen3.5 checkpoint.

    The Maze task is text-only, so there is no reason to optimize vision
    parameters. Returns the number of scalar parameters frozen.
    """
    frozen = 0

    vision_markers = (
        "visual",
        "vision_model",
        "vision_tower",
    )

    for name, parameter in model.named_parameters():
        lowered = name.lower()

        if any(
            marker in lowered
            for marker in vision_markers
        ):
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += parameter.numel()

    return frozen


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """Return `(trainable_parameters, total_parameters)`."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return trainable, total


def load_model_and_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    *,
    device: str | torch.device | None = None,
    dtype: DTypeName | torch.dtype = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    gradient_checkpointing: bool = False,
    freeze_vision: bool = True,
) -> ModelBundle:
    """
    Load Qwen3.5-0.8B for text-only RL training.

    Qwen/Qwen3.5-0.8B is published as a multimodal checkpoint, so the model is
    loaded through `AutoModelForImageTextToText`. The Maze experiment supplies
    text only, therefore generation/training only pass `input_ids` and
    `attention_mask`.

    The model is placed on one explicit device rather than using
    `device_map="auto"`. This is intentional for the first TorchJD/UPGrad
    implementation: the optimizer and Jacobian computation should operate on
    the same in-process model parameters.
    """
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(
        dtype,
        device=resolved_device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    # Decoder-only models should be left-padded for batched generation.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer defines neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
        low_cpu_mem_usage=True,
    )

    model.to(resolved_device)

    if freeze_vision:
        freeze_vision_parameters(model)

    # Training forward passes should not keep autoregressive KV caches.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    text_config = getattr(
        model.config,
        "text_config",
        None,
    )
    if (
        text_config is not None
        and hasattr(text_config, "use_cache")
    ):
        text_config.use_cache = False

    if gradient_checkpointing:
        if not hasattr(
            model,
            "gradient_checkpointing_enable",
        ):
            raise TypeError(
                "Loaded model does not support gradient checkpointing."
            )

        model.gradient_checkpointing_enable()

    model.train()

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
    )
