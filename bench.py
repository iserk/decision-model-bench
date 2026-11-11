"""Benchmark decision-model servers: accuracy, latency, multi-question cost and throughput.

Every system serves the same `POST /v1/systemone` API, each from its own virtualenv.
A target is SYSTEM:DEVICE.

    uv run bench.py                                   # all targets below
    uv run bench.py --targets strands:mlx decider:mps
    uv run bench.py --url http://127.0.0.1:8000       # an already-running server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from envinfo import ENVS, environment, scrub, scrub_file
from workload import CASES, LONG_STATE, MULTI_QUESTIONS

ROOT = Path(__file__).parent
STRANDS_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
# Pinned Hub commits of the competitor checkpoints.
DECIDER_MODEL, DECIDER_REVISION = "Mapika/decider-2b", "7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3"
OPENJEV_MODEL, OPENJEV_REVISION = "ZefanCai/Open-Jev-2B", "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
if sys.platform == "darwin":
    LOCAL_TARGETS = ["strands:mps", "strands:mlx", "decider:mps", "openjev:mps"]
else:  # CUDA when an NVIDIA driver is present, else CPU (slow; for checking the setup)
    _DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
    LOCAL_TARGETS = [f"{s}:{_DEVICE}" for s in ("strands", "decider", "openjev")]
# Jev runs only as TypeSafe's hosted API: latency includes the network round trip.
JEV_URL, JEV_MODEL, JEV_KEY_ENV = "https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY"
DEFAULT_TARGETS = LOCAL_TARGETS + (["jev:api"] if os.environ.get(JEV_KEY_ENV) else [])


def venv_bin(env: str, name: str) -> str:
    return str(ROOT / env / ".venv" / "bin" / name)


def prefetch_strands(model: str) -> dict:
    """The checkpoint is an adapter; its base model is named in its config."""
    from huggingface_hub import snapshot_download

    ckpt = Path(snapshot_download(model))  # .../snapshots/<commit>
    cfg = json.loads((ckpt / "strands_decider_config.json").read_text())
    revision = cfg.get("base_revision")
    prov = ckpt / "provenance.json"
    if not revision and prov.exists():
        meta = json.loads(prov.read_text())
        if meta.get("base_model") == cfg["base_model"]:
            revision = meta.get("base_model_revision")
    print(f"  base model {cfg['base_model']} @ {revision or 'main'}", flush=True)
    base = Path(snapshot_download(cfg["base_model"], revision=revision))
    return {"checkpoint": model, "revisions": {model: ckpt.name, cfg["base_model"]: base.name}}


def prefetch_decider(_model: str) -> dict:
    """Full merged weights, no separate base model."""
    from huggingface_hub import snapshot_download

    return {"checkpoint": snapshot_download(DECIDER_MODEL, revision=DECIDER_REVISION),
            "revisions": {DECIDER_MODEL: DECIDER_REVISION}}


def prefetch_openjev(_model: str) -> dict:
    """An adapter plus scalar head; model.json pins the base model and its revision."""
    from huggingface_hub import snapshot_download

    repo = Path(snapshot_download(OPENJEV_MODEL, revision=OPENJEV_REVISION))
    ckpt = repo / "package" / "checkpoint"
    meta = json.loads((ckpt / "model.json").read_text())
    print(f"  base model {meta['model_id']} @ {meta['revision']}", flush=True)
    snapshot_download(meta["model_id"], revision=meta["revision"])
    return {"checkpoint": str(ckpt),
            "revisions": {OPENJEV_MODEL: OPENJEV_REVISION, meta["model_id"]: meta["revision"]}}


# How to fetch, start and name each system. `cmd` and `env` get the prefetch result,
# the device and the port.
SYSTEMS = {
    "strands": {
        "label": "strands-decider-2B v1",
        "prefetch": prefetch_strands,
        "cmd": lambda p, device, port: [venv_bin(".", "strands-decider"), "serve", p["checkpoint"],
                                        "--device", device, "--port", str(port)],
        "env": lambda p, device: {},
    },
    "decider": {
        "label": "Mapika decider-2b",
        "prefetch": prefetch_decider,
        "cmd": lambda p, device, port: [venv_bin("systems/decider", "uvicorn"), "decider.serve:app",
                                        "--host", "127.0.0.1", "--port", str(port)],
        # DECIDER_IMAGES=0: text only, like strands-decider without --vision. With images on (the
        # checkpoint's default) the server also loads the vision tower, which needs torchvision.
        "env": lambda p, device: {"DECIDER_MODEL": p["checkpoint"], "DECIDER_DEVICE": device,
                                  "DECIDER_IMAGES": "0"},
    },
    "openjev": {
        "label": "Open-Jev-2B",
        # The server rejects any `model` but its own name and the wire alias `jev-latest`.
        "api_model": "open-jev",
        "prefetch": prefetch_openjev,
        # The Apple-silicon configuration from Open-Jev's docs/deployment.md.
        "cmd": lambda p, device, port: [venv_bin("systems/open-jev", "python"), "-m", "jev.server",
                                        "--checkpoint", p["checkpoint"], "--device", device,
                                        "--max-length", "4096", "--batch-size", "1",
                                        "--no-prefix-cache", "--port", str(port)],
        "env": lambda p, device: {"JEV_TORCH_DTYPE": "float32"} if device == "mps" else {},
    },
    "jev": {
        "label": f"TypeSafe Jev ({JEV_MODEL}, hosted)",
        "remote": True,
    },
}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": pct(xs, 50),
            "p95": pct(xs, 95), "min": min(xs), "max": max(xs)}


def is_correct(q: dict, ans: dict, expected) -> bool:
    if q["type"] == "noul":
        return (ans["noul"] >= 0.5) == expected
    if q["type"] == "choice":
        return ans["choice"] == expected
    # The most likely level, not the rounded expected value: that one is pulled toward the
    # middle of the scale whenever the distribution is spread out.
    probs = ans["probabilities"]
    return int(max(probs, key=probs.get)) == expected


def brier(q: dict, ans: dict, expected) -> float:
    """Squared error of the answer's distribution against the label, summed over the options
    (JevBench's definition: 0 is perfect, 2 is confidently wrong). Unlike each server's own
    `confidence`, this means the same thing for every system."""
    if q["type"] == "noul":
        return 2 * (ans["noul"] - float(expected)) ** 2
    probs = {str(k): v for k, v in ans["probabilities"].items()}
    target = str(expected)
    return sum((p - (k == target)) ** 2 for k, p in probs.items()) + (0 if target in probs else 1)


class Server:
    """A server subprocess, logging to <out_dir>/server-<name>.log."""

    def __init__(self, name: str, cmd: list[str], env: dict, port: int, log_dir: Path):
        self.url = f"http://127.0.0.1:{port}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"server-{name.replace(':', '-')}.log", "w")
        print(f"[{name}] starting: {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}",
              flush=True)
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **env}, start_new_session=True)

    def wait_ready(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see {self.log.name}")
            try:
                r = httpx.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    return r.json()
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(f"server not ready after {timeout}s; see {self.log.name}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()


def body(state, questions: dict, model: str | None) -> dict:
    return {"state": state, "questions": questions, **({"model": model} if model else {})}


def post(client: httpx.Client, state, questions: dict, model: str | None = None,
         retries: int = 5) -> tuple[dict, float]:
    """One request; a 429 is retried after Retry-After (or a backoff) and not timed."""
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        r = client.post("/v1/systemone", json=body(state, questions, model))
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code != 429 or attempt == retries:
            break
        time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
    r.raise_for_status()
    return r.json(), elapsed


def bench_accuracy_latency(client: httpx.Client, repeats: int, model: str | None) -> dict:
    """Each labeled case sequentially, `repeats` times; accuracy from the first pass."""
    lat, server_lat, rows = [], [], []
    by_type: dict[str, list[bool]] = {}
    briers = []
    for rep in range(repeats):
        for case in CASES:
            resp, ms = post(client, case["state"], {"q": case["q"]}, model)
            lat.append(ms)
            if "latency_ms" in resp:
                server_lat.append(resp["latency_ms"])
            if rep:
                continue
            ans = resp["answers"]["q"]
            ok = is_correct(case["q"], ans, case["expected"])
            by_type.setdefault(case["q"]["type"], []).append(ok)
            briers.append(brier(case["q"], ans, case["expected"]))
            rows.append({"state": str(case["state"])[:60], "type": case["q"]["type"],
                         "expected": case["expected"], "answer": ans, "correct": ok})
    all_ok = [ok for oks in by_type.values() for ok in oks]
    return {
        "accuracy": sum(all_ok) / len(all_ok),
        "accuracy_by_type": {t: sum(v) / len(v) for t, v in by_type.items()},
        "brier": statistics.fmean(briers),
        "client_latency_ms": summary(lat),
        "server_latency_ms": summary(server_lat) if server_lat else None,
        "cases": rows,
    }


def bench_multi_question(client: httpx.Client, repeats: int, model: str | None) -> dict:
    """Latency of one request on a long state as the number of questions grows."""
    names = list(MULTI_QUESTIONS)
    out = {}
    for k in (1, 2, 4, 8):
        qs = {n: MULTI_QUESTIONS[n] for n in names[:k]}
        lat = []
        for _ in range(repeats):
            resp, ms = post(client, LONG_STATE, qs, model)
            lat.append(ms)
        out[str(k)] = {**summary(lat), "input_tokens": (resp.get("usage") or {}).get("input_tokens")}
    return out


async def bench_throughput(url: str, concurrency: int, total: int, headers: dict,
                           model: str | None) -> dict:
    """`total` requests (labeled cases, cycled) with at most `concurrency` in flight."""
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    errors = 0

    async def one(i: int, client: httpx.AsyncClient) -> None:
        nonlocal errors
        case = CASES[i % len(CASES)]
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post("/v1/systemone",
                                      json=body(case["state"], {"q": case["q"]}, model))
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)
            except httpx.HTTPError:
                errors += 1

    limits = httpx.Limits(max_connections=concurrency * 2)
    async with httpx.AsyncClient(base_url=url, timeout=300, limits=limits, headers=headers) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, client) for i in range(total)))
        wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "requests": total, "errors": errors,
            "wall_s": wall, "rps": len(lat) / wall, "latency_ms": summary(lat) if lat else None}


def run_one(url: str, args: argparse.Namespace, headers: dict | None = None,
            model: str | None = None, concurrency: list[int] | None = None) -> dict:
    headers = headers or {}
    with httpx.Client(base_url=url, timeout=300, headers=headers) as client:
        # Warm-up: caches, allocators and any lazy compilation settle before measuring.
        for case in CASES[: args.warmup]:
            served_by = post(client, case["state"], {"q": case["q"]}, model)[0].get("model")
        print("  accuracy + sequential latency ...", flush=True)
        acc = bench_accuracy_latency(client, args.repeats, model)
        print("  multi-question on a long state ...", flush=True)
        multi = bench_multi_question(client, args.repeats, model)
    tput = []
    for c in concurrency or args.concurrency:
        print(f"  throughput at concurrency {c} ...", flush=True)
        tput.append(asyncio.run(bench_throughput(url, c, args.requests, headers, model)))
    return {"served_by": served_by, "accuracy_latency": acc, "multi_question": multi,
            "throughput": tput}


def print_report(results: dict) -> None:
    ok_runs = {}
    for name, r in results["runs"].items():
        if "error" in r:
            print(f"\n=== {name}: FAILED: {r['error']}")
            continue
        ok_runs[name] = r
        a = r["accuracy_latency"]
        startup = f"startup {r['startup_s']:.0f}s" if "startup_s" in r else "hosted"
        print(f"\n=== {name}  {r.get('label', '')}  [{r.get('served_by')}]  ({startup})")
        by_type = ", ".join(f"{t} {v:.0%}" for t, v in a["accuracy_by_type"].items())
        print(f"accuracy {a['accuracy']:.0%} on {len(CASES)} cases ({by_type}); "
              f"Brier {a['brier']:.3f}")
        c = a["client_latency_ms"]
        print(f"latency, one question (ms): client p50 {c['p50']:.0f} / p95 {c['p95']:.0f}"
              f" / mean {c['mean']:.0f}")
        if "edge_round_trip_ms" in r:
            e = r["edge_round_trip_ms"]["p50"]
            print(f"  of which network + edge (a request the API rejects at once): p50 {e:.0f} ms;"
                  f" past the edge ~{c['p50'] - e:.0f} ms")
        print("questions per request on a long state:")
        for k, m in r["multi_question"].items():
            print(f"  {k:>2} questions  p50 {m['p50']:7.0f} ms   {m['input_tokens'] or '-':>5} input tokens")
        print("throughput:")
        for t in r["throughput"]:
            lat = t["latency_ms"] or {}
            print(f"  concurrency {t['concurrency']:>2}: {t['rps']:6.2f} req/s   "
                  f"p50 {lat.get('p50', 0):7.0f} ms   p95 {lat.get('p95', 0):7.0f} ms   "
                  f"errors {t['errors']}")
    if len(ok_runs) > 1:
        print(f"\n=== comparison  ({len(CASES)} cases)")
        print(f"{'target':<16}{'acc':>6}{'noul':>6}{'choice':>7}{'score':>6}{'Brier':>7}"
              f"{'p50 ms':>8}{'8q ms':>8}{'req/s':>7}")
        for name, r in ok_runs.items():
            a, t = r["accuracy_latency"], r["accuracy_latency"]["accuracy_by_type"]
            best = max(x["rps"] for x in r["throughput"])
            print(f"{name:<16}{a['accuracy']:>6.0%}{t.get('noul', 0):>6.0%}{t.get('choice', 0):>7.0%}"
                  f"{t.get('score', 0):>6.0%}{a['brier']:>7.3f}{a['client_latency_ms']['p50']:>8.0f}"
                  f"{r['multi_question']['8']['p50']:>8.0f}{best:>7.1f}")


def edge_round_trip(url: str, n: int = 20) -> dict:
    """Time of an unauthenticated GET, which the API rejects at its edge: the network round trip
    plus edge time, the floor under every real request. jev_latency.py breaks it down further."""
    lat = []
    with httpx.Client(base_url=url, timeout=30) as client:
        client.get("/v1/systemone")  # open the connection
        for _ in range(n):
            t0 = time.perf_counter()
            client.get("/v1/systemone")
            lat.append((time.perf_counter() - t0) * 1000)
    return summary(lat)


def run_remote(name: str, spec: dict, args: argparse.Namespace) -> dict:
    """A hosted API: no download or server; the key comes from the environment only."""
    key = os.environ.get(JEV_KEY_ENV)
    if not key:
        print(f"[{name}] skipped: set {JEV_KEY_ENV}", flush=True)
        return {"label": spec["label"], "error": f"{JEV_KEY_ENV} is not set"}
    print(f"[{name}] calling {JEV_URL} ({JEV_MODEL})", flush=True)
    try:
        run = run_one(JEV_URL, args, headers={"Authorization": f"Bearer {key}"},
                      model=JEV_MODEL, concurrency=args.remote_concurrency or args.concurrency)
        print("  network round trip ...", flush=True)
        return {"label": spec["label"], "url": JEV_URL, **run,
                "edge_round_trip_ms": edge_round_trip(JEV_URL)}
    except Exception as exc:
        # Never echo the request (it carries the key); the status line is enough.
        msg = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}" \
            if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        print(f"[{name}] failed: {msg}", flush=True)
        return {"label": spec["label"], "error": msg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help=f"SYSTEM:DEVICE pairs; systems: {', '.join(SYSTEMS)} (default: %(default)s)")
    ap.add_argument("--model", default=STRANDS_MODEL, help="strands-decider checkpoint")
    ap.add_argument("--url", help="benchmark an already-running server instead")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--remote-concurrency", type=int, nargs="+",
                    help="concurrency levels for hosted APIs (default: --concurrency); lower them if"
                         " the API rate-limits you")
    ap.add_argument("--requests", type=int, default=50, help="requests per concurrency level")
    ap.add_argument("--startup-timeout", type=float, default=600,
                    help="seconds to wait for a server to load the (prefetched) model")
    args = ap.parse_args()

    targets = [t.split(":", 1) for t in args.targets]
    for system, _device in targets:
        if system not in SYSTEMS:
            ap.error(f"unknown system {system!r}; choose from {', '.join(SYSTEMS)}")

    out_dir = ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
    results = {"environment": environment([s for s, _ in targets if s in ENVS]),
               "models": {}, "args": vars(args), "runs": {}}

    if args.url:
        print(f"benchmarking {args.url}", flush=True)
        results["runs"][args.url] = run_one(args.url, args)
    else:
        # Download every model up front, so a slow download never counts as startup time.
        fetched = {}
        for system in dict.fromkeys(s for s, _ in targets):
            if SYSTEMS[system].get("remote"):
                continue
            print(f"prefetching {system} ...", flush=True)
            fetched[system] = SYSTEMS[system]["prefetch"](args.model)
            results["models"].update(fetched[system]["revisions"])
        for system, device in targets:
            name, spec = f"{system}:{device}", SYSTEMS[system]
            if spec.get("remote"):
                results["runs"][name] = run_remote(name, spec, args)
                continue
            server = Server(name, spec["cmd"](fetched[system], device, args.port),
                            spec["env"](fetched[system], device), args.port, out_dir)
            try:
                t0 = time.monotonic()
                health = server.wait_ready(args.startup_timeout)
                startup = time.monotonic() - t0
                print(f"[{name}] ready in {startup:.0f}s", flush=True)
                results["runs"][name] = {"label": spec["label"], "health": health,
                                         "startup_s": startup, **run_one(server.url, args)}
            except Exception as exc:  # keep going with the other targets
                print(f"[{name}] failed: {exc}", flush=True)
                results["runs"][name] = {"label": spec["label"], "error": str(exc)}
            finally:
                server.stop()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(scrub(results), indent=2, default=str))
    for log in out_dir.glob("server-*.log"):
        scrub_file(log)
    print_report(results)
    print(f"\nresults: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
