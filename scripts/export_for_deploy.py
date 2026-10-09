"""Strip optimizer state from a training checkpoint -> small inference-only file for deployment.

Example:
    uv run python scripts/export_for_deploy.py runs/tinystories-17m/best.pt deploy/model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    src, dst = sys.argv[1], Path(sys.argv[2])
    payload = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: payload[k] for k in ("model", "model_config", "tokenizer_dir", "iteration")}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    print(f"{src} ({Path(src).stat().st_size / 1e6:.1f} MB) -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
