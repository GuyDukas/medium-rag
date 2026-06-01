"""Offline pipeline: chunk -> embed (cached) -> upsert to Pinecone.

Budget discipline (graded): embeddings are cached to a local JSONL keyed by a
hash of (model, text). Re-running never re-embeds unchanged chunks, so an
interrupted run resumes cheaply.

Validate on a small subset FIRST, then scale:
    python -m ingest.embed_and_upload --limit 20 --create-index   # dry checks
    python -m ingest.embed_and_upload --limit 20                  # ~20 articles
    python -m ingest.embed_and_upload                             # full corpus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from lib import config
from ingest.chunk import DEFAULT_CSV, Chunk, iter_chunks

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "embeddings.jsonl")


def _cache_key(text: str) -> str:
    h = hashlib.sha256()
    h.update(config.EMBED_MODEL.encode())
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def load_cache() -> dict[str, list[float]]:
    if not os.path.exists(CACHE_PATH):
        return {}
    cache: dict[str, list[float]] = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[rec["k"]] = rec["v"]
    return cache


def append_cache(items: list[tuple[str, list[float]]]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        for k, v in items:
            f.write(json.dumps({"k": k, "v": v}) + "\n")


def ensure_index(pc: Pinecone) -> None:
    existing = {ix["name"] for ix in pc.list_indexes()}
    if config.PINECONE_INDEX in existing:
        print(f"Index '{config.PINECONE_INDEX}' already exists.")
        return
    print(f"Creating index '{config.PINECONE_INDEX}' ({config.EMBED_DIM}d, cosine)...")
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=config.EMBED_DIM,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)
    print("Index ready.")


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def run(limit: int | None, batch_size: int, create_index: bool) -> None:
    client = OpenAI(api_key=config.require("LLMOD_API_KEY"), base_url=config.LLMOD_BASE_URL)
    pc = Pinecone(api_key=config.require("PINECONE_API_KEY"))

    if create_index:
        ensure_index(pc)
    index = pc.Index(config.PINECONE_INDEX)

    cache = load_cache()
    print(f"Loaded {len(cache)} cached embeddings.")

    chunks = list(iter_chunks(DEFAULT_CSV, limit=limit))
    print(f"{len(chunks)} chunks to process (limit={limit}).")

    n_embedded = 0
    n_cached = 0
    n_upserted = 0
    buffer: list[Chunk] = []

    def flush(buf: list[Chunk]) -> None:
        nonlocal n_embedded, n_cached, n_upserted
        if not buf:
            return
        # Split into cached vs. to-embed.
        to_embed = [c for c in buf if _cache_key(c.text) not in cache]
        if to_embed:
            vectors = embed_batch(client, [c.text for c in to_embed])
            new_items = []
            for c, vec in zip(to_embed, vectors):
                cache[_cache_key(c.text)] = vec
                new_items.append((_cache_key(c.text), vec))
            append_cache(new_items)
            n_embedded += len(to_embed)
        n_cached += len(buf) - len(to_embed)

        upserts = [
            {
                "id": c.chunk_id,
                "values": cache[_cache_key(c.text)],
                "metadata": {
                    "article_id": c.article_id,
                    "title": c.title,
                    "url": c.url,
                    "authors": c.authors,
                    "text": c.text,
                },
            }
            for c in buf
        ]
        index.upsert(vectors=upserts)
        n_upserted += len(upserts)
        print(f"  upserted {n_upserted}/{len(chunks)} "
              f"(embedded={n_embedded}, from cache={n_cached})")

    for c in chunks:
        buffer.append(c)
        if len(buffer) >= batch_size:
            flush(buffer)
            buffer = []
    flush(buffer)

    print(f"\nDone. {n_upserted} vectors upserted "
          f"({n_embedded} newly embedded, {n_cached} reused from cache).")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Chunk, embed, and upload to Pinecone.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max articles to process (validate small first).")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Chunks per embed+upsert batch.")
    ap.add_argument("--create-index", action="store_true",
                    help="Create the Pinecone index if it doesn't exist.")
    args = ap.parse_args()
    run(args.limit, args.batch_size, args.create_index)


if __name__ == "__main__":
    _main()
