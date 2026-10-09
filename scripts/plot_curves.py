"""Plot train/val loss for one or more runs.

Example:
    uv run python scripts/plot_curves.py runs/tinystories-17m runs/lr1e-3 --out docs/loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="docs/loss.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in args.runs:
        recs = [json.loads(line) for line in open(Path(run) / "metrics.jsonl")]
        name = Path(run).name
        line, = ax.plot([r["step"] for r in recs], [r["train_loss"] for r in recs], alpha=0.5, label=f"{name} train")
        val = [r for r in recs if "val_loss" in r]
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", color=line.get_color(),
                label=f"{name} val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_ylim(top=min(ax.get_ylim()[1], 6))
    ax.grid(alpha=0.3)
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
