from __future__ import annotations

import torch

from src.part2.experiment import seed_everything
from src.part2.model import DenseTransformer, ModelConfig


def test_cached_greedy_generation_matches_full_forward() -> None:
    seed_everything(5, True)
    model = DenseTransformer(
        ModelConfig(
            vocab_size=24,
            d_model=16,
            n_heads=4,
            n_layers=2,
            mlp_hidden_size=32,
            max_sequence_length=12,
            dropout=0.0,
        )
    ).eval()
    prompt = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
    cached = model.greedy_generate(prompt, max_new_tokens=4)
    full = prompt.clone()
    with torch.inference_mode():
        for _ in range(4):
            next_token = model(full)["logits"][:, -1].argmax(dim=-1, keepdim=True)
            full = torch.cat((full, next_token), dim=1)
    torch.testing.assert_close(cached, full)
