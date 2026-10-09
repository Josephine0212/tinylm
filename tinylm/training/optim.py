from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """AdamW (Loshchilov & Hutter, 2019) with decoupled weight decay."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                g = p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                step_size = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
        return loss


def cosine_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """Linear warmup to max_lr, cosine decay to min_lr at cosine_cycle_iters, then constant."""
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


@torch.no_grad()
def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> Tensor:
    """Scale all gradients in place so their global L2 norm is <= max_l2_norm. Returns the pre-clip norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total > max_l2_norm:
        scale = max_l2_norm / (total + eps)
        for g in grads:
            g.mul_(scale)
    return total
