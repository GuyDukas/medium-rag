"""Local hyperparameter tuning harness (budget-safe — no Pinecone).

For each (chunk_size, overlap) combo we chunk a small article subset, embed the
chunks (reusing the on-disk embedding cache so nothing is paid for twice), and
retrieve with in-memory cosine similarity against the hand-crafted eval set in
`tests/eval_set.py`. `top_k` is then swept for free on the best chunk/overlap
config. Results are printed as comparison tables to justify the chosen values.

    python -m ingest.experiment                 # default sweep
    python -m ingest.experiment --subset 150    # smaller/cheaper
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openai import OpenAI

from lib import config
from ingest.chunk import iter_chunks
from ingest.embed_and_upload import _cache_key, append_cache, embed_batch, load_cache

# (chunk_size, overlap_ratio) combinations to compare.
CHUNK_COMBOS = [
    (256, 0.1),
    (256, 0.2),
    (512, 0.1),
    (512, 0.2),
    (768, 0.1),
    (768, 0.2),
]
TOPK_SWEEP = [3, 5, 8, 12, 20]


def get_embeddings(client: OpenAI, texts: list[str], cache: dict) -> dict[str, list[float]]:
    """Return {text: vector}, embedding (and caching) only cache-misses."""
    misses = [t for t in dict.fromkeys(texts) if _cache_key(t) not in cache]
    for i in range(0, len(misses), 100):
        batch = misses[i : i + 100]
        vecs = embed_batch(client, batch)
        new_items = []
        for t, v in zip(batch, vecs):
            cache[_cache_key(t)] = v
            new_items.append((_cache_key(t), v))
        append_cache(new_items)
    return {t: cache[_cache_key(t)] for t in texts}


def build_corpus(client: OpenAI, subset: int, chunk_size: int, overlap: float, cache: dict):
    """Chunk the subset and return (matrix [n,d] normalized, article_ids[n])."""
    chunks = list(iter_chunks(limit=subset, chunk_size=chunk_size, overlap_ratio=overlap))
    texts = [c.text for c in chunks]
    emb = get_embeddings(client, texts, cache)
    mat = np.array([emb[t] for t in texts], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    article_ids = [c.article_id for c in chunks]
    return mat, article_ids, len(chunks)


def retrieve(qvec: np.ndarray, mat: np.ndarray, article_ids: list[str], top_k: int):
    """Return deduped article_ids ranked by best chunk score (highest first)."""
    sims = mat @ qvec
    order = np.argsort(-sims)
    seen, ranked = set(), []
    for idx in order:
        aid = article_ids[idx]
        if aid in seen:
            continue
        seen.add(aid)
        ranked.append(aid)
        if len(ranked) >= top_k:
            break
    return ranked


def evaluate(cases, qvecs, mat, article_ids, top_k: int) -> dict:
    """Score the eval set. Returns aggregate + per-type metrics."""
    hits = 0
    rr_sum = 0.0  # reciprocal rank for single-target types
    n_single = 0
    multi_ok = 0
    n_multi = 0
    by_type: dict[str, list[int]] = {}

    for case in cases:
        ranked = retrieve(qvecs[case["question"]], mat, article_ids, top_k)
        targets = set(case["targets"])
        qtype = case["qtype"]

        if qtype == "multi_result":
            found = sum(1 for aid in ranked if aid in targets)
            ok = found >= case.get("min_distinct", 1)
            multi_ok += int(ok)
            n_multi += 1
        else:
            present = [i for i, aid in enumerate(ranked, 1) if aid in targets]
            ok = len(present) > 0
            if ok:
                rr_sum += 1.0 / present[0]
            n_single += 1

        hits += int(ok)
        by_type.setdefault(qtype, []).append(int(ok))

    return {
        "hit_rate": hits / len(cases),
        "mrr_single": (rr_sum / n_single) if n_single else 0.0,
        "multi_rate": (multi_ok / n_multi) if n_multi else 0.0,
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    }


def main() -> None:
    from tests.eval_set import EVAL_CASES

    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=200, help="Articles to tune on.")
    args = ap.parse_args()

    client = OpenAI(api_key=config.require("LLMOD_API_KEY"), base_url=config.LLMOD_BASE_URL)
    cache = load_cache()
    print(f"Loaded {len(cache)} cached embeddings. Tuning on {args.subset} articles.\n")

    # Pre-embed all questions once (cached, shared across combos).
    questions = [c["question"] for c in EVAL_CASES]
    qemb = get_embeddings(client, questions, cache)
    qvecs = {}
    for q in questions:
        v = np.array(qemb[q], dtype=np.float32)
        qvecs[q] = v / (np.linalg.norm(v) + 1e-9)

    # --- Part A: compare (chunk_size, overlap) at a fixed k ---
    fixed_k = config.TOP_K
    print(f"=== Part A: chunk_size/overlap comparison @ top_k={fixed_k} ===")
    print(f"{'chunk':>6} {'ovlp':>5} {'chunks':>7} {'hit':>6} {'mrr':>6} {'multi':>6}")
    results = []
    corpora = {}
    for cs, ov in CHUNK_COMBOS:
        mat, aids, n = build_corpus(client, args.subset, cs, ov, cache)
        corpora[(cs, ov)] = (mat, aids)
        m = evaluate(EVAL_CASES, qvecs, mat, aids, fixed_k)
        results.append(((cs, ov), m, n))
        print(f"{cs:>6} {ov:>5} {n:>7} {m['hit_rate']:>6.2f} "
              f"{m['mrr_single']:>6.2f} {m['multi_rate']:>6.2f}")

    # Pick best combo: hit_rate, then mrr, then fewer chunks (cheaper).
    best = max(results, key=lambda r: (r[1]["hit_rate"], r[1]["mrr_single"], -r[2]))
    (bcs, bov), bm, bn = best
    print(f"\nBest chunk/overlap: chunk_size={bcs}, overlap={bov} "
          f"(hit={bm['hit_rate']:.2f}, mrr={bm['mrr_single']:.2f})")
    print("  per-type:", {k: round(v, 2) for k, v in bm["by_type"].items()})

    # --- Part B: sweep top_k on the best combo ---
    print(f"\n=== Part B: top_k sweep @ chunk_size={bcs}, overlap={bov} ===")
    print(f"{'top_k':>6} {'hit':>6} {'mrr':>6} {'multi':>6}")
    mat, aids = corpora[(bcs, bov)]
    topk_results = []
    for k in TOPK_SWEEP:
        m = evaluate(EVAL_CASES, qvecs, mat, aids, k)
        topk_results.append((k, m))
        print(f"{k:>6} {m['hit_rate']:>6.2f} {m['mrr_single']:>6.2f} {m['multi_rate']:>6.2f}")

    # Smallest k that maximizes hit_rate (smaller k = cheaper context).
    best_hit = max(m["hit_rate"] for _, m in topk_results)
    best_k = min(k for k, m in topk_results if m["hit_rate"] == best_hit)

    print("\n" + "=" * 50)
    print("RECOMMENDED:")
    print(f"  chunk_size   = {bcs}")
    print(f"  overlap_ratio= {bov}")
    print(f"  top_k        = {best_k}")
    print("=" * 50)


if __name__ == "__main__":
    main()
