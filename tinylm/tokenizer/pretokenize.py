"""Pre-tokenization shared by BPE training and encoding."""

from __future__ import annotations

import os
from collections import Counter
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization pattern.
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def special_split_pattern(special_tokens: list[str]) -> re.Pattern | None:
    """Regex that splits on special tokens; longest first so overlapping tokens resolve greedily."""
    if not special_tokens:
        return None
    alts = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.compile(f"({alts})")


def count_pretokens(text: str, special_tokens: list[str]) -> Counter[bytes]:
    """Count pre-tokens in `text`, never letting a pre-token cross a special-token boundary."""
    counts: Counter[bytes] = Counter()
    splitter = special_split_pattern(special_tokens)
    segments = splitter.split(text) if splitter else [text]
    specials = set(special_tokens)
    for seg in segments:
        if not seg or seg in specials:
            continue
        counts.update(m.group().encode("utf-8") for m in PAT.finditer(seg))
    return counts


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Byte offsets that split the file into ~equal chunks, each boundary aligned to a special token."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if buf == b"":
                boundaries[bi] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos + found
                break
            pos += mini
    return sorted(set(boundaries))
