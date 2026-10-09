from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over all leading dims. logits (..., V), targets (...)."""
    logits = logits.float()
    max_logit = logits.amax(dim=-1, keepdim=True)
    log_sum_exp = (logits - max_logit).exp().sum(dim=-1).log() + max_logit.squeeze(-1)
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logit).mean()


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """Sample random windows from a 1D token array (works with np.memmap without loading it all)."""
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length + 1)[None, :]
    chunk = torch.from_numpy(dataset[idx].astype(np.int64))
    x, y = chunk[:, :-1], chunk[:, 1:]
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes], extra: dict | None = None) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    if extra:
        payload.update(extra)
    torch.save(payload, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> int:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
