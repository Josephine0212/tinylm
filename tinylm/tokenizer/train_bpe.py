"""Byte-level BPE training.

Two optimizations over the naive algorithm (recount every pair, every merge):

1. Parallel pre-tokenization: the corpus is split at special-token boundaries and each chunk is
   pre-tokenized in its own process; per-chunk Counters are summed.
2. Incremental merging: we keep global pair counts plus an inverted index pair -> words containing it.
   A merge only touches words that contain the merged pair, and the best pair is found with a
   lazily-invalidated max-heap instead of a full scan.
"""

from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from .pretokenize import count_pretokens, find_chunk_boundaries

PARALLEL_THRESHOLD_BYTES = 32 * 1024 * 1024  # below this, process start-up costs more than it saves


class _Desc:
    """Reverses ordering so heapq (a min-heap) breaks count ties by the lexicographically *largest* pair."""

    __slots__ = ("v",)

    def __init__(self, v: tuple[bytes, bytes]):
        self.v = v

    def __lt__(self, other: _Desc) -> bool:
        return self.v > other.v

    def __eq__(self, other) -> bool:
        return self.v == other.v


def _count_chunk(args: tuple[str, int, int, list[str]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return count_pretokens(text, special_tokens)


def pretokenize_file(input_path: str | os.PathLike, special_tokens: list[str],
                     num_workers: int | None = None, chunks_per_worker: int = 4) -> Counter[bytes]:
    input_path = str(input_path)
    size = os.path.getsize(input_path)
    if size < PARALLEL_THRESHOLD_BYTES or not special_tokens:
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            return count_pretokens(f.read(), special_tokens)

    num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
    with open(input_path, "rb") as f:
        # More chunks than workers keeps every chunk small -> bounded peak memory per process.
        bounds = find_chunk_boundaries(f, num_workers * chunks_per_worker, special_tokens[0].encode("utf-8"))
    jobs = [(input_path, s, e, special_tokens) for s, e in zip(bounds[:-1], bounds[1:])]
    total: Counter[bytes] = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for c in pool.map(_count_chunk, jobs):
            total.update(c)
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
    **_,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    byte_offset = len(vocab)
    for b in range(256):
        vocab[byte_offset + b] = bytes([b])

    pretoken_counts = pretokenize_file(input_path, special_tokens, num_workers=num_workers)

    # Each distinct pre-token becomes a "word": a list of token ids, with a frequency.
    words: list[list[int]] = [[byte_offset + b for b in tok] for tok in pretoken_counts]
    freqs: list[int] = list(pretoken_counts.values())

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            pair_words[pair].add(wi)

    def entry(pair: tuple[int, int]):
        return (-pair_counts[pair], _Desc((vocab[pair[0]], vocab[pair[1]])), pair)

    heap = [entry(p) for p in pair_counts]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and heap:
        neg, _, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -neg or -neg <= 0:
            continue  # stale heap entry
        a, b = best
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        touched: set[tuple[int, int]] = set()
        for wi in pair_words.pop(best, ()):
            w, f = words[wi], freqs[wi]
            for pair in zip(w, w[1:]):
                pair_counts[pair] -= f
                touched.add(pair)
            merged: list[int] = []
            i = 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[wi] = merged
            for pair in zip(merged, merged[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(wi)
                touched.add(pair)
        pair_counts.pop(best, None)
        for pair in touched:
            if pair == best:
                continue
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, entry(pair))
            else:
                pair_counts.pop(pair, None)
    return vocab, merges
