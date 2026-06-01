"""Offline contract-shape tests for the API payloads (no network / no keys).

Mocks retrieval and the chat client so we can assert the exact JSON structure
required by /api/prompt and /api/stats without touching Pinecone or LLMod.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import config, rag

FAKE_MATCHES = [
    {"article_id": "12", "title": "A", "url": "http://x/a", "authors": "Jane Doe",
     "chunk": "alpha text", "score": 0.42},
    {"article_id": "12", "title": "A", "url": "http://x/a", "authors": "Jane Doe",
     "chunk": "alpha two", "score": 0.40},
    {"article_id": "7", "title": "B", "url": "http://x/b", "authors": "John Roe",
     "chunk": "beta text", "score": 0.31},
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChat:
    def __init__(self, content):
        self._content = content
        self.completions = self

    def create(self, **_):
        return _FakeCompletion(self._content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _patch(monkey_matches, answer):
    rag.retrieve = lambda *a, **k: list(monkey_matches)        # type: ignore
    rag._openai = lambda: _FakeClient(answer)                  # type: ignore


def test_prompt_payload_shape():
    _patch(FAKE_MATCHES, "Final answer from context.")
    out = rag.generate("any question")

    assert set(out.keys()) == {"response", "context", "Augmented_prompt"}
    assert out["response"] == "Final answer from context."

    assert isinstance(out["context"], list) and out["context"]
    for c in out["context"]:
        # Exact spec shape — no extra fields (no url/authors leak into context).
        assert set(c.keys()) == {"article_id", "title", "chunk", "score"}
        assert isinstance(c["article_id"], str)
        assert isinstance(c["score"], float)

    ap = out["Augmented_prompt"]
    assert set(ap.keys()) == {"System", "User"}          # exact casing
    assert ap["System"] == config.SYSTEM_PROMPT
    assert "any question" in ap["User"]


def test_user_prompt_includes_context_and_ids():
    prompt = rag.build_user_prompt("Q?", FAKE_MATCHES)
    assert "article_id=12" in prompt
    assert "alpha text" in prompt
    assert "Q?" in prompt
    assert "Jane Doe" in prompt          # authors available to the model (Fix 2)


def test_dedup_returns_distinct_articles():
    deduped = rag.dedup_by_article(FAKE_MATCHES, limit=3)
    ids = [m["article_id"] for m in deduped]
    assert ids == ["12", "7"]          # highest-scoring chunk per article, order kept


def test_stats_payload_exact_keys():
    s = config.stats()
    assert list(s.keys()) == ["chunk_size", "overlap_ratio", "top_k"]


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
