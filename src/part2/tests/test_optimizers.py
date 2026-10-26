from __future__ import annotations

import pytest
import torch

from src.part2.model import DenseTransformer, ModelConfig
from src.part2.optimizers import (
    AdamW,
    Lion,
    Muon,
    NAdamW,
    SophiaG,
    build_muon_parameter_groups,
)
from src.part2.optimizers.muon import newton_schulz_orthogonalize


def optimizer_for(name: str, parameters, *, weight_decay: float = 0.0):
    if name == "adamw":
        return AdamW(parameters, lr=0.01, weight_decay=weight_decay, max_grad_norm=None)
    if name == "nadamw":
        return NAdamW(parameters, lr=0.01, weight_decay=weight_decay, max_grad_norm=None)
    if name == "lion":
        return Lion(parameters, lr=0.01, weight_decay=weight_decay, max_grad_norm=None)
    if name == "sophia":
        return SophiaG(parameters, lr=0.01, weight_decay=weight_decay, max_grad_norm=None)
    if name == "muon":
        return Muon(
            [{"params": list(parameters), "use_muon": False}],
            lr=0.01,
            weight_decay=weight_decay,
            max_grad_norm=None,
        )
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["adamw", "nadamw", "lion", "muon", "sophia"])
def test_step_changes_parameter_with_gradient(name: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = optimizer_for(name, [parameter])
    parameter.grad = torch.tensor([0.5, -0.25])
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter, before)


@pytest.mark.parametrize("name", ["adamw", "nadamw", "lion", "muon", "sophia"])
def test_zero_and_missing_gradient_decay_semantics(name: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = optimizer_for(name, [parameter], weight_decay=0.2)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.998, -1.996]))
    before = parameter.detach().clone()
    parameter.grad = None
    optimizer.step()
    torch.testing.assert_close(parameter, before)


def _tensor_state_keys(optimizer, parameter) -> set[str]:
    return {key for key, value in optimizer.state[parameter].items() if torch.is_tensor(value)}


def test_expected_optimizer_state_tensors() -> None:
    for optimizer_class in (AdamW, NAdamW):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = optimizer_class([parameter], max_grad_norm=None)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert _tensor_state_keys(optimizer, parameter) == {"exp_avg", "exp_avg_sq"}

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = Lion([parameter], max_grad_norm=None)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert _tensor_state_keys(optimizer, parameter) == {"exp_avg"}

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = SophiaG([parameter], max_grad_norm=None)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert _tensor_state_keys(optimizer, parameter) == {"exp_avg", "hessian"}

    matrix = torch.nn.Parameter(torch.ones(3, 2))
    vector = torch.nn.Parameter(torch.ones(2))
    optimizer = Muon(
        [
            {"params": [matrix], "use_muon": True},
            {"params": [vector], "use_muon": False},
        ],
        max_grad_norm=None,
    )
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    assert _tensor_state_keys(optimizer, matrix) == {"momentum_buffer"}
    assert _tensor_state_keys(optimizer, vector) == {"exp_avg", "exp_avg_sq"}


def test_muon_parameter_routing() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden_size=32,
        max_sequence_length=8,
        dropout=0.0,
    )
    model = DenseTransformer(config)
    groups = build_muon_parameter_groups(
        model.named_parameters(),
        muon_lr=0.02,
        adamw_lr=0.001,
        weight_decay=0.1,
        adamw_weight_decay=0.01,
    )
    optimizer = Muon(groups, lr=0.02)
    routes = {row["name"]: row["route"] for row in optimizer.routing_table()}
    assert routes["blocks.0.attention.qkv.weight"] == "muon"
    assert routes["blocks.0.mlp.up.weight"] == "muon"
    assert routes["embedding.weight"] == "adamw_fallback"
    assert routes["blocks.0.attention_norm.weight"] == "adamw_fallback"
    assert routes["blocks.0.attention_norm.bias"] == "adamw_fallback"
    assert all(
        parameter.ndim == 2
        for group in optimizer.param_groups
        if group["use_muon"]
        for parameter in group["params"]
    )


def test_newton_schulz_is_finite_and_shape_preserving() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(9, 5, generator=generator)
    result = newton_schulz_orthogonalize(matrix, steps=5)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()
    singular_values = torch.linalg.svdvals(result)
    assert float((singular_values - 1).abs().mean()) < 0.35


def test_sophia_update_rule_matches_hand_computation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    optimizer = SophiaG(
        [parameter],
        lr=0.1,
        betas=(0.5, 0.5),
        rho=1.0,
        weight_decay=0.0,
        tokens_per_batch=2,
        max_grad_norm=None,
    )
    # Hessian EMA from a sampled-label gradient of [1, 10, 0]: h = 0.5 * g^2.
    parameter.grad = torch.tensor([1.0, 10.0, 0.0])
    assert optimizer.update_hessian()
    torch.testing.assert_close(
        optimizer.state[parameter]["hessian"], torch.tensor([0.5, 50.0, 0.0])
    )
    # m = 0.5 * g = [2, 2, 2]; ratio = |m| / (rho * B * h) clipped at 1.
    parameter.grad = torch.tensor([4.0, 4.0, 4.0])
    optimizer.step()
    expected_ratio = torch.tensor([min(2.0 / 1.0, 1.0), 2.0 / 100.0, 1.0])
    torch.testing.assert_close(parameter.detach(), 1.0 - 0.1 * expected_ratio)


def test_sophia_rejects_non_finite_hessian_estimate() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = SophiaG([parameter], max_grad_norm=None)
    parameter.grad = torch.tensor([float("inf"), 1.0])
    assert not optimizer.update_hessian()
    assert not optimizer.state[parameter]
    parameter.grad = torch.tensor([4.0, 8.0])
    assert optimizer.update_hessian(gradient_divisor=4.0)
    torch.testing.assert_close(
        optimizer.state[parameter]["hessian"], 0.01 * torch.tensor([1.0, 4.0])
    )
