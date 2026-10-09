"""TinyLM: a small GPT-style language model built from scratch in PyTorch."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import ModelConfig, TransformerLM
from .tokenizer import Tokenizer

__version__ = "0.1.0"


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> TransformerLM:
    """Load a TransformerLM from a checkpoint written by scripts/train.py."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TransformerLM(ModelConfig(**payload["model_config"]))
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def load_tokenizer(tokenizer_dir: str | Path, special_tokens: list[str] | None = None) -> Tokenizer:
    d = Path(tokenizer_dir)
    return Tokenizer.from_files(d / "vocab.json", d / "merges.txt", special_tokens or ["<|endoftext|>"])


__all__ = ["ModelConfig", "Tokenizer", "TransformerLM", "load_model", "load_tokenizer"]
