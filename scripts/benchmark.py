"""Benchmarks used in the README.

  bpe     naive BPE (full recount each merge) vs. tinylm's incremental + parallel trainer
  decode  generation throughput with and without the KV cache

Examples:
    uv run python scripts/benchmark.py bpe --input tests/fixtures/tinystories_sample_5M.txt --vocab-size 1000
    uv run python scripts/benchmark.py decode --ckpt runs/tinystories-17m/best.pt --tokens 256
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import torch

from tinylm.tokenizer import train_bpe
from tinylm.tokenizer.pretokenize import count_pretokens


def naive_bpe(path: str, vocab_size: int, special_tokens: list[str]):
    """Textbook BPE: rescan every word for every merge. O(merges x corpus)."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    counts = count_pretokens(text, special_tokens)
    words = {tuple(bytes([b]) for b in w): c for w, c in counts.items()}
    n_vocab, merges = 256 + len(special_tokens), []
    while n_vocab < vocab_size:
        pairs: Counter = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        new_words = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] = new_words.get(tuple(out), 0) + c
        words = new_words
        n_vocab += 1
    return merges


def bench_bpe(args) -> None:
    sp = ["<|endoftext|>"]
    t0 = time.perf_counter()
    _, fast = train_bpe(args.input, args.vocab_size, sp)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    slow = naive_bpe(args.input, args.vocab_size, sp)
    t_slow = time.perf_counter() - t0
    print(json.dumps({"input": args.input, "vocab_size": args.vocab_size, "naive_sec": round(t_slow, 2),
                      "tinylm_sec": round(t_fast, 2), "speedup": round(t_slow / t_fast, 1),
                      "identical_merges": fast == slow}, indent=2))


def bench_decode(args) -> None:
    from tinylm import load_model
    from tinylm.generate import SamplingConfig, generate_ids
    from tinylm.training import pick_device

    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    prompt = list(range(1, args.prompt_len + 1))
    res = {"device": device}
    for use_cache in (False, True):
        cfg = SamplingConfig(max_new_tokens=args.tokens, temperature=1.0, top_p=1.0, seed=0, stop_token=None)
        list(generate_ids(model, prompt, SamplingConfig(max_new_tokens=8, seed=0), use_kv_cache=use_cache))  # warmup
        t0 = time.perf_counter()
        n = sum(1 for _ in generate_ids(model, prompt, cfg, stop_id=None, use_kv_cache=use_cache))
        dt = time.perf_counter() - t0
        res["kv_cache" if use_cache else "no_cache"] = round(n / dt, 1)
    res["speedup"] = round(res["kv_cache"] / res["no_cache"], 2)
    print(json.dumps({"tokens_per_sec": res}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bpe")
    b.add_argument("--input", required=True)
    b.add_argument("--vocab-size", type=int, default=1000)
    d = sub.add_parser("decode")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--tokens", type=int, default=256)
    d.add_argument("--prompt-len", type=int, default=16)
    d.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    {"bpe": bench_bpe, "decode": bench_decode}[args.cmd](args)


if __name__ == "__main__":
    main()
