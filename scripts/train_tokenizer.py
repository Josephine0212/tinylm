"""Train a byte-level BPE tokenizer and report time / peak memory / compression.

Example:
    uv run python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --out artifacts/tokenizer-tinystories-10k
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from tinylm.tokenizer import Tokenizer, train_bpe


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return usage / 1e6 if sys.platform == "darwin" else usage / 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens, num_workers=args.workers)
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(vocab, merges, args.special_tokens)
    tok.save(out / "vocab.json", out / "merges.txt")

    specials = {t.encode("utf-8") for t in args.special_tokens}
    longest = max((v for v in vocab.values() if v not in specials), key=len)
    stats = {
        "input": args.input,
        "vocab_size": len(vocab),
        "train_seconds": round(elapsed, 2),
        "peak_rss_mb_main_process": round(peak_rss_mb(), 1),
        "longest_token": longest.decode("utf-8", errors="replace"),
        "longest_token_bytes": len(longest),
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
