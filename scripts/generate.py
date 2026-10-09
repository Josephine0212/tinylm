"""Sample text from a trained checkpoint (streams to stdout).

Example:
    uv run python scripts/generate.py --ckpt runs/tinystories-17m/best.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import sys
import time

from tinylm import load_model, load_tokenizer
from tinylm.generate import SamplingConfig, stream_text
from tinylm.training import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to the tokenizer recorded in the checkpoint")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-kv-cache", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    tok_dir = args.tokenizer or torch.load(args.ckpt, map_location="cpu", weights_only=False)["tokenizer_dir"]
    tok = load_tokenizer(tok_dir)
    model = load_model(args.ckpt, pick_device(args.device))
    cfg = SamplingConfig(args.max_new_tokens, args.temperature, args.top_p, seed=args.seed)

    sys.stdout.write(args.prompt)
    t0, n = time.perf_counter(), 0
    for piece in stream_text(model, tok, args.prompt, cfg, use_kv_cache=not args.no_kv_cache):
        sys.stdout.write(piece)
        sys.stdout.flush()
        n += 1
    dt = time.perf_counter() - t0
    print(f"\n\n[{n} pieces in {dt:.2f}s, {n / dt:.1f} pieces/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
