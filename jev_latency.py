"""Split Jev's response time into network round trip and server time.

    TYPESAFE_API_KEY=... uv run jev_latency.py [--n 20]

1. Connection setup on fresh connections: DNS, TCP connect (about one round trip), TLS.
2. On one kept-open connection (as bench.py uses):
   - an unauthenticated GET the API rejects at once: about one round trip plus edge time;
   - real one-question and eight-question requests.
   Real minus rejected is about the time spent past the edge (auth, queueing, the model).
"""

from __future__ import annotations

import argparse
import os
import socket
import ssl
import statistics
import time
from urllib.parse import urlparse

import httpx

from bench import JEV_KEY_ENV, JEV_MODEL, JEV_URL
from workload import CASES, LONG_STATE, MULTI_QUESTIONS

TIMING_HEADERS = ("server-timing", "x-response-time", "x-process-time", "x-envoy-upstream-service-time",
                  "x-runtime", "cf-ray", "server", "via", "x-cache", "x-amz-cf-pop", "fly-request-id")


def stats(xs: list[float]) -> str:
    xs = sorted(xs)
    return (f"min {xs[0]:6.1f}  p50 {statistics.median(xs):6.1f}  "
            f"p95 {xs[int(0.95 * (len(xs) - 1))]:6.1f}  mean {statistics.fmean(xs):6.1f} ms")


def connection_phases(host: str, n: int) -> dict[str, list[float]]:
    """DNS, TCP and TLS times of `n` fresh connections, with the standard library."""
    out: dict[str, list[float]] = {"dns": [], "tcp": [], "tls": []}
    ctx = ssl.create_default_context()
    for _ in range(n):
        t0 = time.perf_counter()
        addr = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)[0][4]
        t1 = time.perf_counter()
        sock = socket.create_connection(addr[:2], timeout=10)
        t2 = time.perf_counter()
        tls = ctx.wrap_socket(sock, server_hostname=host)
        t3 = time.perf_counter()
        tls.close()
        out["dns"].append((t1 - t0) * 1000)
        out["tcp"].append((t2 - t1) * 1000)
        out["tls"].append((t3 - t2) * 1000)
    return out


def timed(client: httpx.Client, method: str, path: str, **kw) -> tuple[float, httpx.Response]:
    t0 = time.perf_counter()
    r = client.request(method, path, **kw)
    return (time.perf_counter() - t0) * 1000, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="samples per measurement")
    args = ap.parse_args()
    key = os.environ.get(JEV_KEY_ENV)
    if not key:
        raise SystemExit(f"set {JEV_KEY_ENV}")
    host = urlparse(JEV_URL).hostname
    print(f"host {host} -> {socket.gethostbyname(host)}")

    print(f"\n1. fresh connections (n={args.n})")
    for phase, xs in connection_phases(host, args.n).items():
        print(f"  {phase:<4} {stats(xs)}")

    one = {"q": CASES[0]["q"]}
    eight = MULTI_QUESTIONS
    auth = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=JEV_URL, timeout=60) as client:
        # Open the connection, then warm both request shapes.
        client.get("/v1/systemone")
        for _ in range(3):
            client.post("/v1/systemone", headers=auth,
                        json={"model": JEV_MODEL, "state": CASES[0]["state"], "questions": one})

        rows: dict[str, list[float]] = {"rejected GET (no auth)": [], "POST, 1 question": [],
                                        "POST, 8 questions, long state": []}
        last: dict[str, httpx.Response] = {}
        for _ in range(args.n):  # interleaved, so drift hits every row alike
            ms, r = timed(client, "GET", "/v1/systemone")
            rows["rejected GET (no auth)"].append(ms); last["get"] = r
            ms, r = timed(client, "POST", "/v1/systemone", headers=auth,
                          json={"model": JEV_MODEL, "state": CASES[0]["state"], "questions": one})
            r.raise_for_status(); rows["POST, 1 question"].append(ms); last["one"] = r
            ms, r = timed(client, "POST", "/v1/systemone", headers=auth,
                          json={"model": JEV_MODEL, "state": LONG_STATE, "questions": eight})
            r.raise_for_status(); rows["POST, 8 questions, long state"].append(ms); last["eight"] = r

    print(f"\n2. one kept-open connection (n={args.n}, interleaved)")
    for name, xs in rows.items():
        print(f"  {name:<31} {stats(xs)}")
    print(f"  (the rejected GET returned HTTP {last['get'].status_code})")

    base = statistics.median(rows["rejected GET (no auth)"])
    print("\n3. split of the median request time")
    for name in ("POST, 1 question", "POST, 8 questions, long state"):
        total = statistics.median(rows[name])
        print(f"  {name:<31} total {total:6.1f} ms = network+edge {base:6.1f} ms ({base / total:4.0%})"
              f" + past the edge {total - base:6.1f} ms ({(total - base) / total:4.0%})")

    print("\n4. response headers that may carry timing or routing")
    for label, r in (("rejected GET", last["get"]), ("POST, 1 question", last["one"])):
        found = {h: r.headers[h] for h in TIMING_HEADERS if h in r.headers}
        print(f"  {label}: {found or 'none of ' + ', '.join(TIMING_HEADERS[:4]) + ', ...'}")
    print(f"  served by {last['one'].json().get('model')}")


if __name__ == "__main__":
    main()
