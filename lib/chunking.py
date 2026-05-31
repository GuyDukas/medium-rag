"""Token-based chunking shared by the ingestion pipeline.

Chunking is done by tokens (via tiktoken) so that `chunk_size` and
`overlap_ratio` are meaningful and match the values reported by /api/stats.
"""
from __future__ import annotations

from typing import Iterator

import tiktoken

# text-embedding-3-small uses the cl100k_base encoding.
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> Iterator[str]:
    """Yield overlapping token windows of `text` decoded back to strings.

    Args:
        text: the article body.
        chunk_size: max tokens per chunk (<= 1024 per spec).
        overlap_ratio: fraction of overlap between consecutive chunks, [0, 0.3].
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    tokens = _ENC.encode(text or "")
    if not tokens:
        return

    overlap = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap)

    start = 0
    n = len(tokens)
    while start < n:
        window = tokens[start : start + chunk_size]
        chunk = _ENC.decode(window).strip()
        if chunk:
            yield chunk
        if start + chunk_size >= n:
            break
        start += step
