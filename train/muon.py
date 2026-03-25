from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


def _orthogonalize_newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    # Muon-style orthogonalization for 2D updates.
    x = matrix
    transposed = False
    if x.size(0) < x.size(1):
        x = x.t()
        transposed = True

    x = x / (x.norm() + eps)
    for _ in range(steps):
        x = 1.5 * x - 0.5 * (x @ x.t() @ x)

    if transposed:
        x = x.t()
    return x


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                g = grad.float()
                if weight_decay != 0.0:
                    g = g + weight_decay * p.data.float()

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                m = state["momentum"]
                m.mul_(beta).add_(g, alpha=1.0 - beta)

                update = m
                if update.ndim == 2:
                    update = _orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)

                p.data.add_(update.to(dtype=p.dtype), alpha=-lr)

        return loss
