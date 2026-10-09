"""Tokenize a text corpus into a flat uint16 token file (np.memmap-friendly), in parallel.

Example:
    uv run python scripts/encode_dataset.py --input data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer artifacts/tokenizer-tinystories-10k --out data/tinystories_valid.bin
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tinylm import load_tokenizer
from tinylm.tokenizer.pretokenize import find_chunk_boundaries

SPECIAL = "<|endoftext|>"
_TOK = None


def _init(tokenizer_dir: str) -> None:
    global _TOK
    _TOK = load_tokenizer(tokenizer_dir, [SPECIAL])


def _encode_chunk(args: tuple[str, int, int, str]) -> tuple[str, int, int]:
    path, start, end, part_path = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = np.asarray(_TOK.encode(text), dtype=np.uint16)
    ids.tofile(part_path)
    return part_path, len(ids), end - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer, [SPECIAL])
    assert tok.vocab_size <= 65536, "uint16 storage needs vocab <= 65536"

    t0 = time.perf_counter()
    with open(args.input, "rb") as f:
        bounds = find_chunk_boundaries(f, args.workers * 8, SPECIAL.encode())
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(args.input, s, e, str(Path(tmp) / f"part{i:05d}.bin"))
                for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
            results = list(pool.map(_encode_chunk, jobs))
        total_tokens = sum(n for _, n, _ in results)
        total_bytes = sum(b for _, _, b in results)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = np.memmap(args.out, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        offset = 0
        for part, n, _ in results:
            out[offset: offset + n] = np.fromfile(part, dtype=np.uint16)
            offset += n
        out.flush()
    dt = time.perf_counter() - t0
    print(f"{total_tokens:,} tokens from {total_bytes / 1e6:.1f} MB in {dt:.1f}s "
          f"({total_bytes / 1e6 / dt:.1f} MB/s, {total_bytes / max(total_tokens, 1):.2f} bytes/token) -> {args.out}")


if __name__ == "__main__":
    main()
