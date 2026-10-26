"""Manual Muon with an internal, manually implemented AdamW fallback."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from .common import (
    decoupled_decay,
    dense_float_gradient,
    global_gradient_scale,
    validate_adam_hyperparameters,
    validate_max_grad_norm,
)


def newton_schulz_orthogonalize(
    matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the polar factor with Muon's quintic Newton--Schulz iteration.

    This is the official Muon quintic update corresponding to Algorithm 8's
    Newton--Schulz operation. Computation is float32 for numerical stability:
      X <- X / (||X||_F + eps)
      A <- X X^T
      X <- 3.4445 X + (-4.7750 A + 2.0315 A^2) X
    Tall matrices are transposed during the iteration and restored afterward.
    Primary implementation: https://github.com/KellerJordan/Muon
    Assignment reference: https://arxiv.org/abs/2509.02046 (Algorithm 8)
    """
    if matrix.ndim != 2:
        raise ValueError(f"Newton--Schulz requires a matrix, got {matrix.ndim}D")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = matrix.float()
    x = x / (torch.linalg.vector_norm(x) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x.mT if transposed else x


def is_muon_parameter(name: str, parameter: torch.Tensor) -> bool:
    """Route only 2D weight matrices inside Transformer blocks through Muon.

    Embeddings, the LM head (including tied embedding/head weights), LayerNorm
    gains, biases, scalars, and every non-2D tensor use the custom AdamW path.
    """
    return parameter.ndim == 2 and name.startswith("blocks.") and name.endswith(".weight")


def build_muon_parameter_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    adamw_weight_decay: float,
) -> list[dict]:
    muon_parameters: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    fallback_parameters: list[torch.nn.Parameter] = []
    fallback_names: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if is_muon_parameter(name, parameter):
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            fallback_parameters.append(parameter)
            fallback_names.append(name)
    groups: list[dict] = []
    if muon_parameters:
        groups.append(
            {
                "params": muon_parameters,
                "param_names": muon_names,
                "use_muon": True,
                "lr": muon_lr,
                "weight_decay": weight_decay,
            }
        )
    if fallback_parameters:
        groups.append(
            {
                "params": fallback_parameters,
                "param_names": fallback_names,
                "use_muon": False,
                "lr": adamw_lr,
                "weight_decay": adamw_weight_decay,
            }
        )
    return groups


class Muon(Optimizer):
    """Muon for Transformer matrices and custom AdamW for all other tensors.

    Muon matrix state is one momentum tensor. Fallback tensors have the two
    float32 moment tensors required by AdamW. Parameter groups must set the
    boolean ``use_muon`` explicitly; this keeps routing auditable and testable.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        if ns_steps < 0 or ns_eps <= 0:
            raise ValueError("ns_steps must be non-negative and ns_eps positive")
        validate_adam_hyperparameters(lr, adamw_betas, adamw_eps, weight_decay)
        validate_max_grad_norm(max_grad_norm)
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "ns_eps": ns_eps,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if "use_muon" not in group:
                raise ValueError("every Muon parameter group must declare use_muon")
            validate_adam_hyperparameters(
                group["lr"], group["adamw_betas"], group["adamw_eps"], group["weight_decay"]
            )
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.ndim != 2:
                        raise ValueError("Muon groups may contain only 2D parameters")

    def routing_table(self) -> list[dict]:
        rows = []
        for group_index, group in enumerate(self.param_groups):
            names = group.get(
                "param_names", [f"parameter_{i}" for i in range(len(group["params"]))]
            )
            for name, parameter in zip(names, group["params"], strict=True):
                rows.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "route": "muon" if group["use_muon"] else "adamw_fallback",
                        "group": group_index,
                    }
                )
        return rows

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        clipping_values = {group["max_grad_norm"] for group in self.param_groups}
        if len(clipping_values) != 1:
            raise ValueError("all parameter groups must use one max_grad_norm")
        scale = global_gradient_scale(self.param_groups, next(iter(clipping_values)))

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = dense_float_gradient(parameter, scale)
                state = self.state[parameter]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    momentum = state["momentum_buffer"]
                    beta = group["momentum"]
                    # Algorithm 8: m_t = beta*m_(t-1)+g; u=beta*m_t+g.
                    momentum.mul_(beta).add_(gradient)
                    update = momentum.mul(beta).add(gradient)
                    update = newton_schulz_orthogonalize(update, group["ns_steps"], group["ns_eps"])
                    scale_factor = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                    update.mul_(scale_factor)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["step"] += 1
                    beta1, beta2 = group["adamw_betas"]
                    moment = state["exp_avg"]
                    variance = state["exp_avg_sq"]
                    moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    update = (moment / bias1) / ((variance / bias2).sqrt().add_(group["adamw_eps"]))
                decoupled_decay(parameter, group["lr"], group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss
