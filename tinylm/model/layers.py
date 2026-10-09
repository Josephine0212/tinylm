"""Transformer building blocks implemented from scratch.

Only `torch.nn.Parameter`, `torch.nn.Module` and basic tensor ops are used — no
`nn.Linear`, `nn.Embedding`, `nn.functional.scaled_dot_product_attention`, etc.
"""

from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class Linear(nn.Module):
    """y = x W^T (no bias). Weight stored as (d_out, d_in), truncated-normal init."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # upcast for numerically stable mean of squares
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE with precomputed cos/sin buffers; rotates adjacent pairs (x0,x1), (x2,x3), ..."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE needs an even head dimension"
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(x.dtype)  # (..., seq, d_k/2)
        sin = self.sin[token_positions].to(x.dtype)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def softmax(x: Tensor, dim: int) -> Tensor:
    x = x - x.amax(dim=dim, keepdim=True)  # subtract max for numerical stability
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """mask: bool, True = attend. Shapes (..., q, d_k), (..., k, d_k), (..., k, d_v)."""
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einsum(softmax(scores, dim=-1), V, "... q k, ... k d -> ... q d")


class KVCache:
    """Per-layer key/value cache for incremental decoding."""

    def __init__(self, num_layers: int):
        self.k: list[Tensor | None] = [None] * num_layers
        self.v: list[Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[-2]

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=-2)
            v = torch.cat([self.v[layer], v], dim=-2)
        self.k[layer], self.v[layer] = k, v
        return k, v



class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        seq_len = x.shape[-2]
        split = "... s (h d) -> ... h s d"
        q = rearrange(self.q_proj(x), split, h=self.num_heads)
        k = rearrange(self.k_proj(x), split, h=self.num_heads)
        v = rearrange(self.v_proj(x), split, h=self.num_heads)

        past = kv_cache.seq_len if kv_cache is not None else 0
        if token_positions is None:
            token_positions = torch.arange(past, past + seq_len, device=x.device)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        total = k.shape[-2]
        # Causal mask: query i (absolute position total-seq_len+i) may attend to keys <= its position.
        q_idx = torch.arange(total - seq_len, total, device=x.device)[:, None]
        k_idx = torch.arange(total, device=x.device)[None, :]
        mask = k_idx <= q_idx
        out = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(rearrange(out, "... h s d -> ... s (h d)"))


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(RMSNorm(x)), then + FFN(RMSNorm(.))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding | None,
                 device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None, kv_cache: KVCache | None = None,
                layer_idx: int = 0) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions, kv_cache, layer_idx)
        return x + self.ffn(self.ln2(x))
