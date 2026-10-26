from __future__ import annotations

import pytest
import torch

from src.part2.experiment import make_optimizer, seed_everything
from src.part2.model import DenseTransformer, ModelConfig

SETTINGS = {
    "adamw": {
        "lr": 0.01,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    },
    "nadamw": {
        "lr": 0.005,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    },
    "lion": {
        "lr": 0.003,
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    },
    "muon": {
        "lr": 0.03,
        "momentum": 0.95,
        "weight_decay": 0.0,
        "ns_steps": 5,
        "ns_eps": 1e-7,
        "adamw_lr": 0.01,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "adamw_weight_decay": 0.0,
        "max_grad_norm": 1.0,
    },
}


@pytest.mark.parametrize("optimizer_name", list(SETTINGS))
def test_tiny_causal_lm_overfits(optimizer_name: str) -> None:
    seed_everything(11, True)
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden_size=32,
        max_sequence_length=6,
        dropout=0.0,
    )
    model = DenseTransformer(config)
    optimizer = make_optimizer(optimizer_name, model, SETTINGS[optimizer_name])
    rows = torch.tensor([[4, 5, 6, 7, 8, 9, 10], [5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)
    input_ids, labels = rows[:, :-1], rows[:, 1:]
    model.eval()
    initial = float(model(input_ids, labels)["loss"].item())
    model.train()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels)["loss"]
        loss.backward()
        optimizer.step()
    model.eval()
    final = float(model(input_ids, labels)["loss"].item())
    assert final < initial * 0.75, (optimizer_name, initial, final)
