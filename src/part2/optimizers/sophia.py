"""Manual SophiaG (Hessian-approximation category, bonus) from arXiv:2305.14342."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .common import (
    decoupled_decay,
    dense_float_gradient,
    global_gradient_scale,
    validate_max_grad_norm,
)


class SophiaG(Optimizer):
    """Sophia with the Gauss-Newton-Bartlett diagonal Hessian estimator.

    Every step, for clipped gradient g:
      m_t = beta1*m_(t-1) + (1-beta1)*g
      w <- w - lr*wd*w
      w <- w - lr * sign(m_t) * min(|m_t| / (rho * B * h + eps), 1)
    Every k steps the training loop calls ``update_hessian`` after a backward pass on
    labels sampled from the model itself, giving gradient g_hat, and
      h <- beta2*h + (1-beta2)*g_hat*g_hat.
    B is the number of tokens per batch; it rescales h because the sampled loss is a
    per-token mean. This follows the authors' reference implementation
    (github.com/Liuhong99/Sophia, sophia.py and train_sophiag.py).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        tokens_per_batch: int = 1,
        eps: float = 1e-15,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if rho <= 0 or eps <= 0:
            raise ValueError("rho and eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        if tokens_per_batch < 1:
            raise ValueError("tokens_per_batch must be positive")
        validate_max_grad_norm(max_grad_norm)
        defaults = {
            "lr": lr,
            "betas": betas,
            "rho": rho,
            "weight_decay": weight_decay,
            "tokens_per_batch": tokens_per_batch,
            "eps": eps,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)

    def _state(self, parameter: torch.Tensor) -> dict:
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["hessian"] = torch.zeros_like(parameter, dtype=torch.float32)
        return state

    @torch.no_grad()
    def update_hessian(self, gradient_divisor: float = 1.0) -> bool:
        """Fold the current (sampled-label) gradients into the Hessian EMA.

        ``gradient_divisor`` undoes any loss scaling applied before backward. If any
        gradient is non-finite (FP16 overflow) the whole estimate is discarded and
        False is returned, mirroring how GradScaler skips an optimizer step.
        """
        gradients = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float() / gradient_divisor
                gradients.append((group, parameter, gradient))
        if not gradients:
            return False
        if not all(bool(torch.isfinite(gradient).all()) for _, _, gradient in gradients):
            return False
        for group, parameter, gradient in gradients:
            beta2 = group["betas"][1]
            self._state(parameter)["hessian"].mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )
        return True

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
            beta1 = group["betas"][0]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = dense_float_gradient(parameter, scale)
                state = self._state(parameter)
                state["step"] += 1
                moment = state["exp_avg"]
                moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                denominator = state["hessian"].mul(group["rho"] * group["tokens_per_batch"])
                ratio = moment.abs().div_(denominator.add_(group["eps"])).clamp_(max=1.0)
                update = moment.sign().mul_(ratio)
                decoupled_decay(parameter, group["lr"], group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss

    def clipped_fraction(self) -> float:
        """Fraction of coordinates whose last update hit the clip (|ratio| == 1)."""
        clipped = 0
        total = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                denominator = state["hessian"] * group["rho"] * group["tokens_per_batch"]
                ratio = state["exp_avg"].abs() / (denominator + group["eps"])
                clipped += int((ratio >= 1.0).sum())
                total += ratio.numel()
        return math.nan if total == 0 else clipped / total
