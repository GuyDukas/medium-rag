"""Offline sanity checks (no network / no API keys needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import config
from lib.chunking import chunk_text, count_tokens


def test_chunk_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = list(chunk_text(text, chunk_size=100, overlap_ratio=0.2))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c) <= 100


def test_empty_text_yields_nothing():
    assert list(chunk_text("", 100, 0.2)) == []


def test_no_overlap_covers_all_tokens():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = list(chunk_text(text, chunk_size=50, overlap_ratio=0.0))
    assert len(chunks) >= 1


def test_stats_shape_and_constraints():
    s = config.stats()
    assert set(s) == {"chunk_size", "overlap_ratio", "top_k"}
    assert isinstance(s["chunk_size"], int) and 0 < s["chunk_size"] <= 1024
    assert 0 <= s["overlap_ratio"] <= 0.3
    assert isinstance(s["top_k"], int) and 1 <= s["top_k"] <= 30


def test_system_prompt_is_verbatim():
    assert config.SYSTEM_PROMPT.startswith(
        "You are a Medium-article assistant that answers questions strictly and only"
    )
    assert config.NO_ANSWER == "I don't know based on the provided Medium articles data."
    assert config.NO_ANSWER in config.SYSTEM_PROMPT


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
