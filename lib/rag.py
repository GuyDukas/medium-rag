"""Shared retrieval + generation logic used by the deployed API.

Flow: embed the question -> query Pinecone -> build the augmented prompt ->
call the chat model -> return the response, the retrieved context, and the
exact prompt that was sent (per the /api/prompt contract).
"""
from __future__ import annotations

import functools
from typing import Any

from openai import OpenAI
from pinecone import Pinecone

from . import config


@functools.lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI(
        api_key=config.require("LLMOD_API_KEY"),
        base_url=config.LLMOD_BASE_URL,
    )


@functools.lru_cache(maxsize=1)
def _index():
    pc = Pinecone(api_key=config.require("PINECONE_API_KEY"))
    return pc.Index(config.PINECONE_INDEX)


def embed_query(text: str) -> list[float]:
    resp = _openai().embeddings.create(model=config.EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return the top-k matching chunks as plain dicts."""
    top_k = top_k or config.TOP_K
    vector = embed_query(question)
    res = _index().query(vector=vector, top_k=top_k, include_metadata=True)

    matches = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        matches.append(
            {
                "article_id": str(md.get("article_id", "")),
                "title": md.get("title", ""),
                "url": md.get("url", ""),
                "chunk": md.get("text", ""),
                "score": round(float(m.get("score", 0.0)), 4),
            }
        )
    return matches


def dedup_by_article(
    matches: list[dict[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    """Keep the highest-scoring chunk per distinct article_id, preserving order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in matches:
        aid = m["article_id"]
        if aid in seen:
            continue
        seen.add(aid)
        out.append(m)
        if limit and len(out) >= limit:
            break
    return out


def build_user_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    """Compose the user message: numbered context passages + the question."""
    if not matches:
        context_block = "(no relevant passages were retrieved)"
    else:
        parts = []
        for i, m in enumerate(matches, 1):
            parts.append(
                f"[{i}] article_id={m['article_id']} | title: {m['title']}\n"
                f"url: {m['url']}\n"
                f"passage: {m['chunk']}"
            )
        context_block = "\n\n".join(parts)

    return (
        "Use only the following retrieved Medium article context to answer.\n\n"
        f"=== CONTEXT ===\n{context_block}\n=== END CONTEXT ===\n\n"
        f"Question: {question}"
    )


def generate(question: str) -> dict[str, Any]:
    """Run the full RAG pipeline and return the /api/prompt payload."""
    matches = retrieve(question)
    user_prompt = build_user_prompt(question, matches)

    completion = _openai().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()

    return {
        "response": answer,
        "context": matches,
        "Augmented_prompt": {
            "System": config.SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
