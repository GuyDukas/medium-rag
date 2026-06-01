"""GET / — a small landing response so the bare URL shows the API is live.

The graded endpoints are /api/prompt and /api/stats; this just avoids a bare 404
at the root.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (required name)
        payload = {
            "service": "Medium Article RAG Assistant",
            "status": "ok",
            "endpoints": {
                "POST /api/prompt": {"question": "your natural language question"},
                "GET /api/stats": "current RAG hyperparameters",
            },
            "repo": "https://github.com/GuyDukas/medium-rag",
        }
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
