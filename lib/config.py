"""Central configuration: model names, RAG hyperparameters, and env loading.

This is the single source of truth. /api/stats reads the hyperparameters from
here, so changing them in one place updates the reported config automatically.
"""
from __future__ import annotations

import os

try:
    # Load .env when running locally. On Vercel the vars come from the
    # dashboard, and python-dotenv may not be installed there — that's fine.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# --- Models (via the LLMod proxy, OpenAI-compatible) ---------------------
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
EMBED_MODEL = "4UHRUIN-text-embedding-3-small"  # 1536 dims
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
EMBED_DIM = 1536

# --- Pinecone ------------------------------------------------------------
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "medium-rag")
PINECONE_METRIC = "cosine"

# --- RAG hyperparameters (reported verbatim by /api/stats) ---------------
# Constraints from the assignment:
#   chunk_size   : int, max 1024 tokens
#   overlap_ratio: float in [0, 0.3]
#   top_k        : int in [1, 30]
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 8

# How many distinct articles a multi-result topic listing returns.
MAX_DISTINCT_ARTICLES = 3


# --- Required chat system prompt (verbatim; do not alter the constraints) -
SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. If the "
    "answer cannot be determined from the provided context, respond: \"I don't know "
    "based on the provided Medium articles data.\" Always explain your answer using the "
    "given context, quoting or paraphrasing the relevant article passage or metadata "
    "when helpful."
)

# Scoped clarification (permitted by the spec: constraints stay verbatim above,
# response-style clarifications may be appended). Resolves the conflict between
# "respond exactly" for the no-answer case and "always explain" for real answers.
SYSTEM_PROMPT += (
    " Important: when the answer cannot be determined from the provided context, your "
    "entire reply must be exactly \"I don't know based on the provided Medium articles "
    "data.\" with no additional words, explanation, or punctuation. Only add an "
    "explanation when you are actually answering from the context."
)

# Exact string to return when the answer is not in context.
NO_ANSWER = "I don't know based on the provided Medium articles data."


def require(name: str) -> str:
    """Fetch a required env var or raise a clear error."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in .env (local) or Vercel Environment Variables (deploy)."
        )
    return val


def stats() -> dict:
    """The exact payload returned by GET /api/stats."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
