"""Tests for functionality beyond the CS336 reference suite: KV cache, sampling, tokenizer IO, API server."""

from __future__ import annotations

import json

import pytest
import torch

from tinylm.generate import SamplingConfig, generate_ids, sample_next, stream_text
from tinylm.model import ModelConfig, TransformerLM
from tinylm.tokenizer import Tokenizer, train_bpe

from .common import FIXTURES_PATH

SMALL = ModelConfig(vocab_size=300, context_length=32, d_model=32, num_layers=2, num_heads=4, d_ff=64)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    vocab, merges = train_bpe(FIXTURES_PATH / "tinystories_sample.txt", 300, ["<|endoftext|>"])
    return Tokenizer(vocab, merges, ["<|endoftext|>"])


@pytest.fixture(scope="module")
def model() -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(SMALL).eval()


def test_kv_cache_matches_full_forward(model):
    ids = torch.randint(0, SMALL.vocab_size, (2, 20))
    full = model(ids)
    cache = model.new_kv_cache()
    prefill = model(ids[:, :12], kv_cache=cache)
    steps = [model(ids[:, i: i + 1], kv_cache=cache) for i in range(12, 20)]
    incremental = torch.cat([prefill] + steps, dim=1)
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-4)


def test_greedy_generation_same_with_and_without_cache(model):
    cfg = SamplingConfig(max_new_tokens=40, temperature=0.0)  # 40 > context_length: exercises re-prefill
    a = list(generate_ids(model, [1, 2, 3], cfg, use_kv_cache=False))
    b = list(generate_ids(model, [1, 2, 3], cfg, use_kv_cache=True))
    assert len(a) == 40
    # Identical until the window first overflows (afterwards the cached path re-anchors positions).
    first_overflow = SMALL.context_length - 3
    assert a[:first_overflow] == b[:first_overflow]


def test_top_p_restricts_support():
    logits = torch.tensor([10.0, 9.0, -10.0, -10.0])
    picks = {sample_next(logits, 1.0, 0.5, torch.Generator().manual_seed(s)) for s in range(50)}
    assert picks == {0}
    picks = {sample_next(logits, 1.0, 0.99, torch.Generator().manual_seed(s)) for s in range(200)}
    assert picks <= {0, 1} and 1 in picks


def test_seeded_sampling_is_deterministic(model):
    cfg = SamplingConfig(max_new_tokens=15, temperature=1.0, top_p=0.95, seed=123)
    assert list(generate_ids(model, [5], cfg)) == list(generate_ids(model, [5], cfg))


def test_tokenizer_save_load_roundtrip(tokenizer, tmp_path):
    tokenizer.save(tmp_path / "vocab.json", tmp_path / "merges.txt")
    loaded = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.txt", ["<|endoftext|>"])
    text = "Once upon a time 🐱 there was a café.<|endoftext|>The end."
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_stream_text_yields_valid_utf8(model, tokenizer):
    pieces = list(stream_text(model, tokenizer, "Hi", SamplingConfig(max_new_tokens=30, temperature=1.0, seed=0)))
    assert all(isinstance(p, str) for p in pieces)


def test_server_endpoints(model, tokenizer):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from server.app import create_app

    client = TestClient(create_app(model, tokenizer))
    assert client.get("/health").json()["ready"] is True

    r = client.post("/v1/generate", json={"prompt": "Once", "max_new_tokens": 10, "seed": 1})
    assert r.status_code == 200 and isinstance(r.json()["text"], str)

    r = client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 0})
    assert r.status_code == 422  # validation

    with client.stream("POST", "/v1/generate/stream", json={"prompt": "Once", "max_new_tokens": 10}) as s:
        body = "".join(s.iter_text())
    assert s.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in body
    deltas = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {\"delta\"")]
    assert len(deltas) >= 1


def test_server_not_ready_without_checkpoint():
    from fastapi.testclient import TestClient

    from server.app import create_app

    client = TestClient(create_app())
    assert client.get("/health").json()["ready"] is False
    assert client.post("/v1/generate", json={"prompt": "hi"}).status_code == 503
