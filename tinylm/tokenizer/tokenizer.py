from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from functools import lru_cache

from .pretokenize import PAT, special_split_pattern


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode char mapping (used only for human-readable files)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    """Byte-level BPE tokenizer: encode/decode with special tokens and streaming encode."""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None, cache_size: int = 50_000):
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens or [])
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = b
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.merges = list(merges)
        self._splitter = special_split_pattern(self.special_tokens)
        self._special_ids = {t: self.token_to_id[t.encode("utf-8")] for t in self.special_tokens}
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ construction / IO
    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike,
                   special_tokens: list[str] | None = None) -> Tokenizer:
        decoder = {c: b for b, c in bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(i): bytes(decoder[c] for c in tok) for tok, i in raw.items()}
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2 and not line.startswith("#version"):
                    merges.append((bytes(decoder[c] for c in parts[0]), bytes(decoder[c] for c in parts[1])))
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike) -> None:
        enc = bytes_to_unicode()
        to_str = lambda b: "".join(enc[x] for x in b)  # noqa: E731
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump({to_str(b): i for i, b in self.vocab.items()}, f, ensure_ascii=False)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{to_str(a)} {to_str(b)}\n")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ encoding
    def _bpe(self, token: bytes) -> tuple[int, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in token]
        ranks = self.merge_ranks
        while len(parts) > 1:
            best_rank, best_i = None, -1
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            parts[best_i: best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        ids = tuple(self.token_to_id[p] for p in parts)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()  # bounded memory; cheap and good enough for natural text
        self._cache[token] = ids
        return ids

    def _encode_plain(self, text: str, out: list[int]) -> None:
        for m in PAT.finditer(text):
            out.extend(self._bpe(m.group().encode("utf-8")))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        segments = self._splitter.split(text) if self._splitter else [text]
        for seg in segments:
            if not seg:
                continue
            sid = self._special_ids.get(seg)
            if sid is not None:
                ids.append(sid)
            else:
                self._encode_plain(seg, ids)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle) with O(chunk) memory."""
        for chunk in iterable:
            yield from self.encode(chunk)

    # ------------------------------------------------------------------ decoding
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)
