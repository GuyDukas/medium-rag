"""Start the real Vercel handlers as local HTTP servers and assert the exact
API contract end-to-end (hits live LLMod + Pinecone). Costs one small /api/prompt
call. Run: python tests/live_http_check.py
"""
import json
import os
import sys
import threading
import urllib.request
from http.server import HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.prompt import handler as PromptHandler
from api.stats import handler as StatsHandler


def serve(handler_cls):
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def check_stats():
    _, port = serve(StatsHandler)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats") as r:
        body = json.loads(r.read())
    assert list(body.keys()) == ["chunk_size", "overlap_ratio", "top_k"], body
    print("GET /api/stats OK ->", body)


def check_prompt(question, expect_no_answer=False):
    _, port = serve(PromptHandler)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/prompt",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        body = json.loads(r.read())

    assert set(body.keys()) == {"response", "context", "Augmented_prompt"}, body.keys()
    assert set(body["Augmented_prompt"].keys()) == {"System", "User"}
    if body["context"]:
        assert set(body["context"][0].keys()) == {
            "article_id", "title", "chunk", "url", "score"
        }
    tag = "(expect no-answer)" if expect_no_answer else "(expect grounded)"
    print(f"\nPOST /api/prompt {tag}: {question!r}")
    print("  response:", body["response"][:160])
    print("  #distinct context articles:",
          len({c['article_id'] for c in body['context']}))


if __name__ == "__main__":
    check_stats()
    check_prompt("How can smell training change your brain?")
    check_prompt("List a few articles with advice on becoming a better writer.")
    check_prompt("What is the capital of Australia?", expect_no_answer=True)
    print("\nAll HTTP contract checks passed.")
