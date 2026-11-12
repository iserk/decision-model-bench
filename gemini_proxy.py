"""Serve Google's hosted Gemini Decisions API as `POST /v1/systemone`, so both benchmarks run it
like every other system.

The Gemini wire format follows the Vercel AI SDK's Google provider, the first public client of the
API (packages/google/src/google-decision-model.ts, added 11 November 2026):

- request: `POST {base}/decisions` with an `x-goog-api-key` header and
  `{"model", "input": [{"type": "text", "text"}], "questions"}`. A JSON state is sent as its JSON
  text, as the SDK does. Every question needs instructions; a `noul` question's `true`/`false`
  criteria become `true_description`/`false_description`.
- response: `answers` already in the `/v1/systemone` shape (`noul`, `choice`, `score`,
  `probabilities`, `confidence`), plus `usage.total_input_tokens`/`total_output_tokens` and,
  for a refused request, `prompt_feedback.block_reason`.

Each response's `latency_ms` is the time spent waiting for Google, so the benchmarks can show the
translator's own overhead. A 5xx or dropped connection is retried up to three times (after 1, 2
and 4 s), so a passing outage isn't scored as a wrong answer. The key is read from
GEMINI_API_KEY only, and never logged.

    GEMINI_API_KEY=... uv run gemini_proxy.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

KEY_ENV = "GEMINI_API_KEY"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-decision-1"
RETRY_STATUSES = {500, 502, 503, 504}
RETRY_DELAYS_S = (1, 2, 4)


def to_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def to_gemini_question(q: dict) -> dict:
    kind, criteria = q.get("type"), q.get("criteria")
    if q.get("instructions") is None:
        raise ValueError("Gemini Decisions requires instructions for every question")
    out = {"type": kind, "instructions": to_text(q["instructions"])}
    if kind == "noul":
        if criteria:
            desc = {f"{k}_description": to_text(criteria[k])
                    for k in ("true", "false") if criteria.get(k) is not None}
            if desc:
                out["criteria"] = desc
    elif kind == "choice":
        out["criteria"] = {k: None if v is None else to_text(v) for k, v in criteria.items()}
    elif kind == "score":
        out["criteria"] = [to_text(level) for level in criteria]
    else:
        raise ValueError(f"unsupported question type {kind!r}")
    if kind != "noul" and not 2 <= len(out["criteria"]) <= 255:
        raise ValueError("Gemini Decisions requires 2 to 255 options or score levels")
    return out


def to_gemini(body: dict, model: str) -> dict:
    questions = body.get("questions") or {}
    if not questions:
        raise ValueError("at least one question is required")
    return {"model": model, "input": [{"type": "text", "text": to_text(body.get("state"))}],
            "questions": {qid: to_gemini_question(q) for qid, q in questions.items()}}


ANSWER_FIELDS = ("type", "noul", "choice", "score", "probabilities", "confidence")


def from_gemini(resp: dict, model: str, upstream_ms: float) -> dict:
    usage = resp.get("usage") or {}
    return {"answers": {qid: {k: a[k] for k in ANSWER_FIELDS if k in a}
                        for qid, a in (resp.get("answers") or {}).items()},
            "model": model, "latency_ms": upstream_ms,
            "usage": {"input_tokens": usage.get("total_input_tokens"),
                      "output_tokens": usage.get("total_output_tokens")}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like the other servers
    client: httpx.Client
    model: str

    def reply(self, status: int, payload: dict, headers: dict | None = None) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.reply(200, {"status": "ok", "model": self.model, "upstream": str(self.client.base_url)})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/systemone":
            self.reply(404, {"error": "not found"})
            return
        try:
            request = to_gemini(json.loads(body), self.model)
        except (ValueError, TypeError, AttributeError) as exc:
            self.reply(422, {"error": str(exc)})
            return
        t0 = time.perf_counter()
        for attempt in range(1 + len(RETRY_DELAYS_S)):
            # A passing outage at Google (5xx or a dropped connection) is retried, so that it
            # isn't scored as a wrong answer. The time waited counts towards upstream_ms.
            try:
                r = self.client.post("/decisions", json=request)
                if r.status_code not in RETRY_STATUSES:
                    break
                problem = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                r, problem = None, type(exc).__name__
            if attempt == len(RETRY_DELAYS_S):
                break
            self.log_message("upstream %s, retrying (%d of %d)", problem, attempt + 1,
                             len(RETRY_DELAYS_S))
            time.sleep(RETRY_DELAYS_S[attempt])
        if r is None:
            self.reply(502, {"error": f"upstream: {problem}"})
            return
        upstream_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            # Google's status and message pass through: the benchmarks retry or stop on 429,
            # 401 and 403 as they do for any server. Retry-After is kept for the 429 backoff.
            retry = {"Retry-After": r.headers["retry-after"]} if "retry-after" in r.headers else {}
            self.log_message("upstream HTTP %d: %s", r.status_code, r.text[:300])
            self.reply(r.status_code, {"error": f"upstream HTTP {r.status_code}: {r.text[:300]}"}, retry)
            return
        resp = r.json()
        block = (resp.get("prompt_feedback") or {}).get("block_reason")
        if block:  # a refusal answers no question; the harness counts the task as wrong
            self.reply(422, {"error": f"blocked by Gemini: {block}"})
            return
        self.reply(200, from_gemini(resp, self.model, upstream_ms))

    def log_message(self, fmt: str, *args) -> None:
        if not fmt.startswith('"'):  # errors only, not one line per request
            sys.stderr.write(f"{self.log_date_time_string()} {fmt % args}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", BASE_URL),
                    help="Gemini API base, or GEMINI_BASE_URL (default: %(default)s)")
    args = ap.parse_args()
    key = os.environ.get(KEY_ENV)
    if not key:
        sys.exit(f"set {KEY_ENV}")
    Handler.model = args.model
    Handler.client = httpx.Client(base_url=args.base_url, timeout=120,
                                  headers={"x-goog-api-key": key},
                                  limits=httpx.Limits(max_connections=64))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving {args.model} from {args.base_url} on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
