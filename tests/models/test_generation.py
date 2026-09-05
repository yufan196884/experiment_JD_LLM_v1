import torch

from jd.models.generation import GenerationBatch


def test_completion_lengths_follow_completion_mask():
    batch = GenerationBatch(
        texts=["a", "b"],
        prompt_input_ids=torch.zeros((1, 2), dtype=torch.long),
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.long),
        sequences=torch.zeros((2, 6), dtype=torch.long),
        full_attention_mask=torch.ones((2, 6), dtype=torch.long),
        completion_ids=torch.zeros((2, 4), dtype=torch.long),
        completion_mask=torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        ),
    )

    assert torch.equal(
        batch.completion_lengths,
        torch.tensor([2, 4]),
    )