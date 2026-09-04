from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from jd.training.grpo_loss import completion_token_logprobs


class NextTokenOracle(nn.Module):
    """Toy causal model whose position j favors input_ids[j + 1]."""

    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        next_ids = torch.roll(input_ids, shifts=-1, dims=1)
        logits = F.one_hot(
            next_ids,
            num_classes=self.vocab_size,
        ).to(torch.float32) * self.scale
        return SimpleNamespace(logits=logits)


def test_completion_logprobs_have_correct_causal_alignment_and_mask():
    model = NextTokenOracle(vocab_size=11)

    sequences = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    prompt_length = 3
    completion_ids = sequences[:, prompt_length:]
    completion_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    full_attention_mask = torch.ones_like(sequences)

    scores = completion_token_logprobs(
        model,
        sequences=sequences,
        full_attention_mask=full_attention_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        prompt_length=prompt_length,
        micro_batch_size=1,
    )

    expected_valid_logprob = 4.0 - torch.log(
        torch.exp(torch.tensor(4.0)) + 10.0
    )

    assert scores.shape == completion_ids.shape
    assert torch.allclose(
        scores[completion_mask],
        torch.full_like(scores[completion_mask], expected_valid_logprob),
        atol=1e-6,
    )
    assert torch.equal(
        scores[~completion_mask],
        torch.zeros_like(scores[~completion_mask]),
    )

    scores.sum().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_completion_ids_must_match_sequence_suffix():
    model = NextTokenOracle(vocab_size=11)
    sequences = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    completion_ids = torch.tensor([[3, 5]], dtype=torch.long)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.bool)

    with pytest.raises(ValueError, match="must exactly equal"):
        completion_token_logprobs(
            model,
            sequences=sequences,
            full_attention_mask=torch.ones_like(sequences),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            prompt_length=2,
        )
