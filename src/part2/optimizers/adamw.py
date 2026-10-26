"""Manual AdamW implementation following Algorithm 1 of arXiv:2509.02046."""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .common import (
    decoupled_decay,
    dense_float_gradient,
    global_gradient_scale,
    validate_adam_hyperparameters,
    validate_max_grad_norm,
)


class AdamW(Optimizer):
    """Adam with bias correction and decoupled weight decay.

    No ready-made optimizer is called. Moment buffers are kept in float32 so the
    implementation also behaves sensibly when model parameters use reduced precision.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        validate_adam_hyperparameters(lr, betas, eps, weight_decay)
        validate_max_grad_norm(max_grad_norm)
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
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
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                numerator = state["exp_avg"] / bias1
                denominator = (state["exp_avg_sq"] / bias2).sqrt().add_(group["eps"])
                update = numerator / denominator
                decoupled_decay(parameter, group["lr"], group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss
