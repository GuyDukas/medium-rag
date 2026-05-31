"""Read the Medium CSV, derive a stable article_id, and produce token chunks.

There is no id column in the dataset, so `article_id` is the 0-based row index
(as a string) and is carried through to Pinecone metadata and the API response.

Run directly to preview chunking on a small subset without embedding:
    python -m ingest.chunk --limit 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import config
from lib.chunking import chunk_text, count_tokens

# Article bodies can be very large single CSV fields.
csv.field_size_limit(10**9)

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "medium-english-50mb.csv",
)


@dataclass
class Chunk:
    chunk_id: str  # f"{article_id}-{i}", used as the Pinecone vector id
    article_id: str
    title: str
    url: str
    text: str


def iter_articles(csv_path: str, limit: int | None = None) -> Iterator[dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield {
                "article_id": str(i),
                "title": (row.get("title") or "").strip(),
                "url": (row.get("url") or "").strip(),
                "text": row.get("text") or "",
            }


def iter_chunks(
    csv_path: str = DEFAULT_CSV,
    limit: int | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    overlap_ratio: float = config.OVERLAP_RATIO,
) -> Iterator[Chunk]:
    for art in iter_articles(csv_path, limit):
        for i, piece in enumerate(chunk_text(art["text"], chunk_size, overlap_ratio)):
            yield Chunk(
                chunk_id=f"{art['article_id']}-{i}",
                article_id=art["article_id"],
                title=art["title"],
                url=art["url"],
                text=piece,
            )


def _main() -> None:
    ap = argparse.ArgumentParser(description="Preview chunking on a subset.")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    n_articles = 0
    n_chunks = 0
    last_article = None
    for c in iter_chunks(args.csv, limit=args.limit):
        if c.article_id != last_article:
            n_articles += 1
            last_article = c.article_id
            print(f"\n=== article {c.article_id}: {c.title[:70]} ===")
        n_chunks += 1
        print(f"  chunk {c.chunk_id} ({count_tokens(c.text)} tok): {c.text[:90]!r}")

    print(f"\n{n_articles} articles -> {n_chunks} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.OVERLAP_RATIO})")


if __name__ == "__main__":
    _main()
