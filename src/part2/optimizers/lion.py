"""Manual Lion implementation based on the primary Lion algorithm."""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .common import (
    decoupled_decay,
    dense_float_gradient,
    global_gradient_scale,
    validate_max_grad_norm,
)


class Lion(Optimizer):
    """Evolved Sign Momentum (arXiv:2302.06675), with decoupled decay.

    Lion stores one full-size momentum tensor per parameter. AdamW and NAdamW
    store both first- and second-moment tensors, so Lion uses half their optimizer
    state tensor memory (excluding scalar/bookkeeping state).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        validate_max_grad_norm(max_grad_norm)
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)

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
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = dense_float_gradient(parameter, scale)
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                momentum = state["exp_avg"]
                # u = sign(beta1*m_(t-1) + (1-beta1)*g_t)
                update = momentum.mul(beta1).add(gradient, alpha=1.0 - beta1).sign_()
                decoupled_decay(parameter, group["lr"], group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                # m_t = beta2*m_(t-1) + (1-beta2)*g_t
                momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
        return loss
