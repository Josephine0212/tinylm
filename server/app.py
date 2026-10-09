"""FastAPI inference server for TinyLM with server-sent-event (SSE) token streaming.

Run:
    TINYLM_CKPT=runs/tinystories-17m/best.pt uv run uvicorn server.app:app --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from tinylm import TransformerLM, load_model, load_tokenizer
from tinylm.generate import SamplingConfig, stream_text
from tinylm.tokenizer import Tokenizer
from tinylm.training import pick_device

STATIC = Path(__file__).parent / "static"


class GenerateRequest(BaseModel):
    prompt: str = Field("Once upon a time", max_length=2000)
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    seed: int | None = None


class GenerateResponse(BaseModel):
    text: str
    num_pieces: int
    latency_ms: float


def create_app(model: TransformerLM | None = None, tokenizer: Tokenizer | None = None) -> FastAPI:
    """App factory. With no arguments, loads the checkpoint named by TINYLM_CKPT at startup."""
    state: dict = {"model": model, "tokenizer": tokenizer}
    # One generation at a time: keeps MPS/CPU memory predictable on a laptop-sized deployment.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ckpt = os.environ.get("TINYLM_CKPT")
        if state["model"] is None and ckpt:  # otherwise /health reports not-ready
            tok_dir = os.environ.get("TINYLM_TOKENIZER") or torch.load(
                ckpt, map_location="cpu", weights_only=False)["tokenizer_dir"]
            state["model"] = load_model(ckpt, pick_device(os.environ.get("TINYLM_DEVICE")))
            state["tokenizer"] = load_tokenizer(tok_dir)
        yield

    app = FastAPI(title="TinyLM", version="0.1.0", lifespan=lifespan)

    def _require() -> tuple[TransformerLM, Tokenizer]:
        if state["model"] is None or state["tokenizer"] is None:
            raise HTTPException(503, "model not loaded; set TINYLM_CKPT")
        return state["model"], state["tokenizer"]

    def _cfg(req: GenerateRequest) -> SamplingConfig:
        return SamplingConfig(req.max_new_tokens, req.temperature, req.top_p, seed=req.seed)

    @app.get("/health")
    def health() -> dict:
        m = state["model"]
        return {
            "ready": m is not None,
            "device": str(next(m.parameters()).device) if m is not None else None,
            "params_non_embedding": m.num_parameters() if m is not None else None,
            "context_length": m.cfg.context_length if m is not None else None,
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        model, tok = _require()
        t0 = time.perf_counter()
        with lock:
            pieces = list(stream_text(model, tok, req.prompt, _cfg(req)))
        return GenerateResponse(text="".join(pieces), num_pieces=len(pieces),
                                latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    @app.post("/v1/generate/stream")
    def generate_stream(req: GenerateRequest) -> StreamingResponse:
        model, tok = _require()

        def events() -> Iterator[str]:
            t0 = time.perf_counter()
            n = 0
            with lock:
                for piece in stream_text(model, tok, req.prompt, _cfg(req)):
                    n += 1
                    yield f"data: {json.dumps({'delta': piece})}\n\n"
            stats = {"num_pieces": n, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            yield f"event: done\ndata: {json.dumps(stats)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
