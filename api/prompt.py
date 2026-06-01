"""POST /api/prompt — answer a question from the Medium corpus via RAG."""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

# Make the repo-root `lib` package importable from the serverless function.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import rag  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (required name)
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            question = (data.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "Missing 'question' in request body."})
                return

            result = rag.generate(question)
            self._send(200, result)
        except Exception:
            # Log the full traceback to the Vercel function logs for debugging,
            # but return a generic message so internals aren't exposed publicly.
            traceback.print_exc()
            self._send(500, {"error": "Internal server error."})
