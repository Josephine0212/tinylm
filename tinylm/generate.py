"""Autoregressive decoding with KV cache, temperature and nucleus (top-p) sampling."""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import TransformerLM, softmax
from .tokenizer import Tokenizer


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    stop_token: str | None = "<|endoftext|>"
    seed: int | None = None


def sample_next(logits: Tensor, temperature: float, top_p: float, generator: torch.Generator | None = None) -> int:
    """logits: (vocab,). temperature 0 -> greedy."""
    if temperature <= 0:
        return int(logits.argmax())
    probs = softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(-1)
        # keep the smallest prefix whose mass >= top_p (always keep the top token)
        keep = cum - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=generator)
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_ids(model: TransformerLM, prompt_ids: list[int], cfg: SamplingConfig,
                 stop_id: int | None = None, use_kv_cache: bool = True) -> Iterator[int]:
    """Yield new token ids one at a time."""
    model.eval()
    device = next(model.parameters()).device
    ctx = model.cfg.context_length
    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    ids = list(prompt_ids)[-ctx:] or [stop_id if stop_id is not None else 0]
    cache = model.new_kv_cache() if use_kv_cache else None
    pending = ids  # tokens not yet fed to the model

    for _ in range(cfg.max_new_tokens):
        if cache is not None and cache.seq_len + len(pending) > ctx:
            # Context window full: re-prefill from the most recent ctx-1 tokens (RoPE positions restart at 0).
            cache = model.new_kv_cache()
            pending = ids[-(ctx - 1):]
        if cache is None:
            window = ids[-ctx:]
            logits = model(torch.tensor([window], device=device))[0, -1]
        else:
            logits = model(torch.tensor([pending], device=device), kv_cache=cache)[0, -1]
        nxt = sample_next(logits.cpu(), cfg.temperature, cfg.top_p, gen)
        if stop_id is not None and nxt == stop_id:
            return
        ids.append(nxt)
        pending = [nxt]
        yield nxt


def stream_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, cfg: SamplingConfig,
                use_kv_cache: bool = True) -> Iterator[str]:
    """Yield decoded text pieces; buffers partial UTF-8 sequences so every piece is valid text."""
    stop_id = None
    if cfg.stop_token and cfg.stop_token.encode() in tokenizer.token_to_id:
        stop_id = tokenizer.token_to_id[cfg.stop_token.encode()]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for tid in generate_ids(model, tokenizer.encode(prompt), cfg, stop_id, use_kv_cache):
        piece = decoder.decode(tokenizer.vocab[tid])
        if piece:
            yield piece
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


def generate_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, cfg: SamplingConfig) -> str:
    return "".join(stream_text(model, tokenizer, prompt, cfg))
