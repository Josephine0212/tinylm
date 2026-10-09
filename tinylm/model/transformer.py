from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .layers import Embedding, KVCache, Linear, RMSNorm, RotaryPositionalEmbedding, TransformerBlock


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.token_embeddings = Embedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        rope = RotaryPositionalEmbedding(cfg.rope_theta, cfg.d_model // cfg.num_heads, cfg.context_length,
                                         device=device)
        # One RoPE module shared by every layer (its buffers are not parameters).
        self.layers = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.num_heads, cfg.d_ff, rope, device=device, dtype=dtype)
            for _ in range(cfg.num_layers)
        )
        self.ln_final = RMSNorm(cfg.d_model, device=device, dtype=dtype)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        """token_ids: (batch, seq) -> logits (batch, seq, vocab)."""
        x = self.token_embeddings(token_ids)
        # Absolute positions of the new tokens; computed once, before any layer appends to the cache.
        past = kv_cache.seq_len if kv_cache is not None else 0
        positions = torch.arange(past, past + token_ids.shape[-1], device=token_ids.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, token_positions=positions, kv_cache=kv_cache, layer_idx=i)
        return self.lm_head(self.ln_final(x))

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embeddings.weight.numel()
        return n

    def new_kv_cache(self) -> KVCache:
        return KVCache(self.cfg.num_layers)

    @torch.no_grad()
    def estimate_flops_per_token(self) -> int:
        """Forward FLOPs per token (matmuls only), useful for MFU / accounting."""
        c = self.cfg
        per_layer = 2 * (4 * c.d_model * c.d_model + 3 * c.d_model * c.d_ff) + 2 * 2 * c.context_length * c.d_model
        return c.num_layers * per_layer + 2 * c.d_model * c.vocab_size
