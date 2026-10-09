"""Train a TransformerLM on a pre-tokenized uint16 dataset.

Example:
    uv run python scripts/train.py --config configs/tinystories_mps.json
    uv run python scripts/train.py --config configs/tinystories_mps.json --set max_lr=1e-3 run_name=lr1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tinylm.model import ModelConfig, TransformerLM
from tinylm.training import AdamW, clip_grad_norm, cosine_lr, cross_entropy, get_batch, load_checkpoint, pick_device
from tinylm.training import save_checkpoint as _save


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, data, cfg, device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg["eval_batches"]):
        x, y = get_batch(data, cfg["batch_size"], model.cfg.context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override top-level keys, e.g. max_lr=1e-3")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--compile", action="store_true", help="torch.compile (CUDA only; unsupported on MPS)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    cfg.update(parse_overrides(args.set))
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_data = np.memmap(cfg["train_data"], dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg["val_data"], dtype=np.uint16, mode="r")

    mcfg = ModelConfig(**cfg["model"])
    model = TransformerLM(mcfg).to(device)
    opt = AdamW(model.parameters(), lr=cfg["max_lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    start_step = load_checkpoint(args.resume, model, opt) if args.resume else 0
    fwd = torch.compile(model) if args.compile else model

    tokens_per_step = cfg["batch_size"] * mcfg.context_length
    print(f"device={device} params(non-emb)={model.num_parameters() / 1e6:.2f}M "
          f"train_tokens={len(train_data):,} tokens/step={tokens_per_step:,} "
          f"total_tokens={tokens_per_step * cfg['max_steps']:,}")

    extra = {"model_config": mcfg.to_dict(), "tokenizer_dir": cfg.get("tokenizer_dir")}

    def save(name: str, step: int) -> None:
        _save(model, opt, step, run_dir / name, extra=extra)

    log = open(run_dir / "metrics.jsonl", "a")
    best_val = math.inf
    t_start = time.perf_counter()
    t_last, tokens_since = t_start, 0
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        lr = cosine_lr(step, cfg["max_lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["max_steps"])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, cfg["batch_size"], mcfg.context_length, device)
        loss = cross_entropy(fwd(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = clip_grad_norm(model.parameters(), cfg["grad_clip"])
        opt.step()
        tokens_since += tokens_per_step

        is_last = step + 1 == cfg["max_steps"]
        if (step + 1) % cfg["log_every"] == 0 or is_last:
            sync(device)
            now = time.perf_counter()
            rec = {"step": step + 1, "train_loss": round(loss.item(), 4), "lr": lr,
                   "grad_norm": round(float(gnorm), 4), "tokens_per_sec": round(tokens_since / (now - t_last)),
                   "elapsed_sec": round(now - t_start, 1)}
            t_last, tokens_since = now, 0
            if (step + 1) % cfg["eval_every"] == 0 or is_last:
                rec["val_loss"] = round(evaluate(model, val_data, cfg, device), 4)
                if rec["val_loss"] < best_val:
                    best_val = rec["val_loss"]
                    save("best.pt", step + 1)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v}" for k, v in rec.items()))
            if not math.isfinite(rec["train_loss"]):
                raise SystemExit("loss diverged (NaN/inf) — try a lower max_lr")
        if (step + 1) % cfg["ckpt_every"] == 0 or is_last:
            save("last.pt", step + 1)

    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min, best val loss {best_val:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
