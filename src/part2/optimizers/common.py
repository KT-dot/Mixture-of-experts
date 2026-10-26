"""Shared validation and gradient utilities for the hand-written optimizers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def validate_adam_hyperparameters(
    lr: float, betas: tuple[float, float], eps: float, weight_decay: float
) -> None:
    if lr < 0:
        raise ValueError(f"learning rate must be non-negative, got {lr}")
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError(f"betas must be in [0, 1), got {betas}")
    if eps <= 0:
        raise ValueError(f"epsilon must be positive, got {eps}")
    if weight_decay < 0:
        raise ValueError(f"weight decay must be non-negative, got {weight_decay}")


def validate_max_grad_norm(max_grad_norm: float | None) -> None:
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or null")


def global_gradient_scale(param_groups: Iterable[dict], max_grad_norm: float | None) -> float:
    """Return the standard global L2 clipping scale without modifying gradients."""
    if max_grad_norm is None:
        return 1.0
    total_sq = 0.0
    for group in param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("sparse gradients are not supported")
            norm = torch.linalg.vector_norm(gradient.detach().float()).item()
            total_sq += norm * norm
    total_norm = math.sqrt(total_sq)
    return min(1.0, max_grad_norm / (total_norm + 1e-12))


def dense_float_gradient(parameter: torch.Tensor, scale: float) -> torch.Tensor:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("dense_float_gradient called without a gradient")
    if gradient.is_sparse:
        raise RuntimeError("sparse gradients are not supported")
    if torch.is_complex(gradient):
        raise RuntimeError("complex gradients are not supported")
    return gradient.detach().float().mul(scale)


def decoupled_decay(parameter: torch.Tensor, lr: float, weight_decay: float) -> None:
    """Apply w <- (1 - lr * lambda) w to a parameter that received a gradient."""
    if weight_decay:
        parameter.mul_(1.0 - lr * weight_decay)
