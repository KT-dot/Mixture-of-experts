"""Update-rule checks against independent scalar re-implementations.

Each reference below is written element by element in plain Python floats straight
from the equations (Wen et al., 2025, Appendix A; Chen et al., 2023 for Lion; Jordan
et al., 2024 for Muon), so it shares no tensor code with the optimizers under test.
"""

from __future__ import annotations

import math

import torch

from src.part2.optimizers import AdamW, Lion, Muon, NAdamW
from src.part2.optimizers.muon import newton_schulz_orthogonalize

GRADIENTS = [[0.5, -1.5, 2.0], [-0.25, 0.75, 1.0], [1.25, 0.1, -3.0]]
START = [1.0, -2.0, 0.5]


def clip_scale(gradient: list[float], max_norm: float | None) -> float:
    if max_norm is None:
        return 1.0
    norm = math.sqrt(sum(value * value for value in gradient))
    return min(1.0, max_norm / (norm + 1e-12))


def reference_adam(nesterov: bool, lr, beta1, beta2, eps, wd, max_norm):
    w, m, v = list(START), [0.0] * 3, [0.0] * 3
    for t, raw in enumerate(GRADIENTS, start=1):
        scale = clip_scale(raw, max_norm)
        for i, g in enumerate(raw):
            g *= scale
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            moment = beta1 * m[i] + (1 - beta1) * g if nesterov else m[i]
            m_hat = moment / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            w[i] = w[i] - lr * wd * w[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
    return w


def run_optimizer(optimizer_class, **kwargs) -> list[float]:
    parameter = torch.nn.Parameter(torch.tensor(START, dtype=torch.float64))
    optimizer = optimizer_class([parameter], **kwargs)
    for gradient in GRADIENTS:
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step()
    return parameter.detach().tolist()


def test_adamw_matches_scalar_reference_with_clipping() -> None:
    settings = {"lr": 0.05, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.1}
    for max_norm in (None, 1.0):
        observed = run_optimizer(AdamW, max_grad_norm=max_norm, **settings)
        expected = reference_adam(False, 0.05, 0.9, 0.99, 1e-8, 0.1, max_norm)
        assert all(math.isfinite(value) for value in observed)
        for got, want in zip(observed, expected, strict=True):
            assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


def test_nadamw_matches_scalar_reference_and_differs_from_adamw() -> None:
    settings = {"lr": 0.05, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.1}
    observed = run_optimizer(NAdamW, max_grad_norm=None, **settings)
    expected = reference_adam(True, 0.05, 0.9, 0.99, 1e-8, 0.1, None)
    for got, want in zip(observed, expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)
    plain = reference_adam(False, 0.05, 0.9, 0.99, 1e-8, 0.1, None)
    assert max(abs(a - b) for a, b in zip(observed, plain, strict=True)) > 1e-4


def test_lion_matches_scalar_reference() -> None:
    lr, beta1, beta2, wd = 0.01, 0.9, 0.99, 0.3
    w, m = list(START), [0.0] * 3
    for raw in GRADIENTS:
        for i, g in enumerate(raw):
            interpolated = beta1 * m[i] + (1 - beta1) * g
            sign = (interpolated > 0) - (interpolated < 0)
            w[i] = w[i] - lr * wd * w[i] - lr * sign
            m[i] = beta2 * m[i] + (1 - beta2) * g
    observed = run_optimizer(Lion, lr=lr, betas=(beta1, beta2), weight_decay=wd, max_grad_norm=None)
    for got, want in zip(observed, w, strict=True):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


def test_newton_schulz_approximates_the_polar_factor() -> None:
    generator = torch.Generator().manual_seed(11)
    for shape in ((6, 4), (4, 6), (8, 8)):
        matrix = torch.randn(*shape, generator=generator, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        polar = u @ vh
        approximation = newton_schulz_orthogonalize(matrix, steps=5).double()
        cosine = torch.sum(approximation * polar) / (approximation.norm() * polar.norm())
        assert float(cosine) > 0.95, (shape, float(cosine))


def test_muon_matrix_update_follows_nesterov_momentum_and_shape_scaling() -> None:
    lr, beta, wd = 0.02, 0.95, 0.1
    generator = torch.Generator().manual_seed(3)
    start = torch.randn(6, 3, generator=generator)
    gradients = [torch.randn(6, 3, generator=generator) for _ in range(3)]
    matrix = torch.nn.Parameter(start.clone())
    optimizer = Muon(
        [{"params": [matrix], "use_muon": True}],
        lr=lr,
        momentum=beta,
        weight_decay=wd,
        ns_steps=5,
        max_grad_norm=None,
    )
    expected = start.clone()
    momentum = torch.zeros_like(start)
    for gradient in gradients:
        matrix.grad = gradient.clone()
        optimizer.step()
        momentum = beta * momentum + gradient
        direction = newton_schulz_orthogonalize(beta * momentum + gradient, steps=5)
        expected = expected * (1 - lr * wd) - lr * math.sqrt(6 / 3) * direction
    torch.testing.assert_close(matrix.detach(), expected, rtol=1e-5, atol=1e-6)
