"""Accuracy and calibration on JevBench's 231 public tasks, scored by JevBench's own harness,
and, with `--suite legalbench`, on 915 LegalBench cases converted to JevBench's format.

JevBench (https://github.com/fstandhartinger/jevbench, MIT) is an independent benchmark for typed
decision models. Its `typesafe` adapter speaks `POST /v1/systemone`, so it runs unchanged against
every system here. This script starts each server, runs the harness from `systems/jevbench`, and
reports accuracy with a 95% interval, accuracy per tier, Brier score, ECE and latency, plus a
paired test between each two systems on the same tasks. See legalbench.py for the second set.

    uv run jevbench_eval.py                              # same default targets as bench.py
    uv run jevbench_eval.py --targets strands:mlx decider:mps jev:api
    uv run jevbench_eval.py --url http://127.0.0.1:8000  # an already-running server
    uv run jevbench_eval.py --suite legalbench           # the second set, same targets
    uv run jevbench_eval.py --report results/jevbench-A results/jevbench-B   # re-print, merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from itertools import combinations
from pathlib import Path

import httpx

import legalbench
from bench import (CASES, DEFAULT_TARGETS, JEV_HARNESS_COST, JEV_KEY_ENV, JEV_MODEL, JEV_URL, ROOT,
                   STRANDS_MODEL, SYSTEMS, Server, pct, post, venv_bin)
from envinfo import ENVS, environment, scrub, scrub_file

JEVBENCH_REPO = "fstandhartinger/jevbench"
JEVBENCH_COMMIT = "fcc69279a933f882ccc4489eb14bf467647dbdb7"  # tag v1.4.2.2, as in systems/jevbench
# The public set: file -> (tier name, sha256). "original" is the board's standard tier.
TASK_FILES = {
    "easy": ("easy", "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "original": ("standard", "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": ("hard", "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
TIERS = ["easy", "standard", "hard"]


# suite -> (fetch the tasks and each one's report column, the columns, title, latency note)
SUITES = {
    "jevbench": (lambda: fetch_tasks(), TIERS, f"JevBench public set, harness {JEVBENCH_COMMIT[:7]}",
                 "states up to ~3,700 tokens"),
    "legalbench": (lambda: legalbench.fetch_tasks(ROOT), legalbench.COLUMNS,
                   f"LegalBench, 8 rule-application tasks ({legalbench.REVISION[:7]})",
                   "states up to ~1,000 tokens"),
    "legalbench-jcrew": (lambda: legalbench.fetch_jcrew_clarified(ROOT), ["noul"],
                         "LegalBench jcrew_blocker, rule wording made explicit",
                         "states up to ~1,000 tokens"),
}


def fetch_tasks() -> tuple[Path, dict[str, str]]:
    """Download the public task files at the pinned commit (once), check their hashes, and
    join them. Returns the joined file and each task id's tier."""
    data = ROOT / "data" / f"jevbench-{JEVBENCH_COMMIT[:7]}"
    data.mkdir(parents=True, exist_ok=True)
    tiers, parts = {}, []
    for name, (tier, sha) in TASK_FILES.items():
        path = data / f"{name}.jsonl"
        if not path.exists():
            url = (f"https://raw.githubusercontent.com/{JEVBENCH_REPO}/{JEVBENCH_COMMIT}"
                   f"/datasets/public/{name}.jsonl")
            print(f"downloading {url}", flush=True)
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            path.write_bytes(r.content)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"{path}: sha256 mismatch; delete it and rerun")
        parts.append(raw if raw.endswith(b"\n") else raw + b"\n")
        for line in raw.decode().splitlines():
            if line.strip():
                tiers[json.loads(line)["id"]] = tier
    joined = data / "public.jsonl"
    joined.write_bytes(b"".join(parts))
    return joined, tiers


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test: b and c are the tasks only one of the two systems got right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def harness(*args: str, log: Path | None = None) -> int:
    cmd = [venv_bin("systems/jevbench", "jevbench"), *args]
    if log is None:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode
    with open(log, "w") as fh:  # the harness prints progress every 10 tasks; show it live too
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line)
            print(f"  {line}", end="", flush=True)
        return proc.wait()


def evaluate(name: str, url: str, model: str, tasks: Path, tiers: dict, out: Path,
             key_env: str = "", delay_s: float = 0.0, warmup: int = 5,
             cost: list[str] | None = None) -> dict:
    """Warm the server up, run the harness over every task, and summarize."""
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"} if key_env else {}
    with httpx.Client(base_url=url, timeout=300, headers=headers) as client:
        for case in CASES[:warmup]:
            post(client, case["state"], {"q": case["q"]}, model)
    d = out / name.replace(":", "-")
    d.mkdir(parents=True)
    run = ["run", "--tasks", str(tasks), "--adapter", "typesafe", "--endpoint", url,
           "--model", model, "--key-env", key_env, "--run-label", name,
           "--results", str(d / "results.jsonl"), "--raw-dir", str(d / "raw"),
           "--ledger", str(d / "ledger.jsonl"), "--manifest", str(d / "manifest.json"),
           "--delay-s", str(delay_s)]
    if cost:  # a hosted API behind a local translator: billed per token
        run += cost
    elif not key_env:  # local weights: no bill; the harness's cost model doesn't apply
        run += ["--reserve-usd", "0", "--cost-basis", "local_weights_compute_excluded"]
    print(f"[{name}] JevBench: {len(tiers)} tasks, serial", flush=True)
    code = harness(*run, log=d / "harness.log")
    harness("summarize", "--tasks", str(tasks), "--results", str(d / "results.jsonl"),
            "--public-export", str(d / "summary.json"))
    return summarize(d, tiers, tasks, complete=code == 0)


def by_group(groups: dict, right: set) -> dict:
    out = {}
    for g in dict.fromkeys(groups.values()):
        ids = {i for i, x in groups.items() if x == g}
        out[g] = {"n": len(ids), "correct": len(ids & right), "accuracy": len(ids & right) / len(ids)}
    return out


def summarize(d: Path, tiers: dict, tasks: Path, complete: bool = True) -> dict:
    """Our report fields from the harness's per-task records and its own summary."""
    s = json.loads((d / "summary.json").read_text())
    records = [json.loads(line) for line in open(d / "results.jsonl") if line.strip()]
    n = len(tiers)
    k = sum(bool(r["correct"]) for r in records)  # an unattempted task counts as wrong
    right = {r["task_id"] for r in records if r["correct"]}
    families = {t["id"]: t["family"] for t in map(json.loads, open(tasks)) if t["id"] in tiers}
    lat = [r["latency_s"] * 1000 for r in records if r["ok"]]
    return {
        "complete": complete and len(records) == n,
        "n": n, "attempted": len(records), "failed": sum(not r["ok"] for r in records),
        "correct": k, "accuracy": k / n, "accuracy_95ci": wilson(k, n),
        "by_tier": by_group(tiers, right), "by_family": by_group(families, right),
        "schema_validity_strict": s.get("schema_validity_strict"),
        "brier": s.get("brier_mean"), "ece": (s.get("ece") or {}).get("ece"),
        "macro_accuracy": s.get("macro_accuracy"),
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "mean": statistics.fmean(lat)}
        if lat else None,
        "served_by": s.get("model_identities"),
        "correct_ids": sorted(r["task_id"] for r in records if r["correct"]),
    }


def paired(runs: dict) -> list[dict]:
    """For each two systems: the tasks only one got right, and the exact McNemar p-value."""
    out = []
    ok = {n: set(r["correct_ids"]) for n, r in runs.items() if "correct_ids" in r}
    for a, b in combinations(ok, 2):
        only_a, only_b = len(ok[a] - ok[b]), len(ok[b] - ok[a])
        out.append({"a": a, "b": b, "only_a": only_a, "only_b": only_b,
                    "p_value": mcnemar_p(only_a, only_b)})
    return out


def print_report(results: dict) -> None:
    runs = {n: r for n, r in results["runs"].items() if "error" not in r}
    for n, r in results["runs"].items():
        if "error" in r:
            print(f"\n{n}: FAILED: {r['error']}")
    if not runs:
        return
    _, columns, title, sizes = SUITES[results.get("suite", "jevbench")]
    n_tasks = next(iter(runs.values()))["n"]
    print(f"\n=== {title} ({n_tasks} tasks)")
    print(f"{'target':<14}{'accuracy':>10}{'95% CI':>15}{''.join(f'{c[:6]:>7}' for c in columns)}"
          f"{'Brier':>7}{'ECE':>7}{'p50 ms':>8}{'p95 ms':>8}  served by")
    for n, r in sorted(runs.items(), key=lambda kv: -kv[1]["accuracy"]):
        lo, hi = r["accuracy_95ci"]
        t = r["by_tier"]
        lat = r["latency_ms"] or {}
        note = "" if r["complete"] else f"  INCOMPLETE: {r['attempted']}/{r['n']} attempted"
        tier = "".join(f"{t[x]['accuracy']:>7.1%}" if x in t else f"{'-':>7}" for x in columns)
        print(f"{n:<14}{r['correct']:>4} {r['accuracy']:>5.1%}{f'{lo:.1%}-{hi:.1%}':>15}{tier}"
              f"{r['brier'] or math.nan:>7.3f}{r['ece'] or math.nan:>7.3f}"
              f"{lat.get('p50', math.nan):>8.0f}{lat.get('p95', math.nan):>8.0f}"
              f"  {', '.join(r['served_by'] or [])}{note}")
    if results.get("paired"):
        print("\npaired (same tasks): right only by A / only by B, exact McNemar p")
        for p in results["paired"]:
            print(f"  {p['a']:<14} vs {p['b']:<14} {p['only_a']:>3} / {p['only_b']:<3}"
                  f"  p = {p['p_value']:.3f}")
    families = {f for r in runs.values() for f in r.get("by_family", {})}
    if results.get("suite", "").startswith("legalbench") and families:
        print("\naccuracy per task")
        print(f"{'task':<34}" + "".join(f"{n[:12]:>13}" for n in runs))
        for f in sorted(families):
            cells = (r.get("by_family", {}).get(f) for r in runs.values())
            print(f"{f:<34}" + "".join(f"{c['accuracy']:>13.1%}" if c else f"{'-':>13}" for c in cells))
    print(f"\nLatency is per task, one request at a time, with {sizes}; a "
          "hosted API's includes the network round trip. Brier: 0 is perfect; lower is better.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, default="jevbench", help="task set (default: %(default)s)")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help=f"SYSTEM:DEVICE pairs; systems: {', '.join(SYSTEMS)} (default: %(default)s)")
    ap.add_argument("--model", default=STRANDS_MODEL, help="strands-decider checkpoint")
    ap.add_argument("--url", help="evaluate an already-running server instead")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--remote-delay", type=float, default=0.0,
                    help="seconds between requests to a hosted API (if it rate-limits you)")
    ap.add_argument("--startup-timeout", type=float, default=600)
    ap.add_argument("--limit", type=int, help="only the first N tasks (a smoke test)")
    ap.add_argument("--report", nargs="+", metavar="DIR",
                    help="re-print saved runs (merged), without running anything")
    args = ap.parse_args()

    if args.report:
        merged = {"runs": {}}
        for d in args.report:
            saved = json.loads((Path(d) / "results.json").read_text())
            merged["suite"] = saved.get("suite", "jevbench")
            merged["runs"].update(saved["runs"])
        merged["paired"] = paired(merged["runs"])
        print_report(merged)
        return

    targets = [t.split(":", 1) for t in args.targets]
    for system, _device in targets:
        if system not in SYSTEMS:
            ap.error(f"unknown system {system!r}; choose from {', '.join(SYSTEMS)}")
    tasks, tiers = SUITES[args.suite][0]()
    out = ROOT / "results" / f"{args.suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    if args.limit:
        lines = tasks.read_text().splitlines(keepends=True)[: args.limit]
        tasks = out / "tasks.jsonl"
        tasks.write_text("".join(lines))
        tiers = {json.loads(line)["id"]: tiers[json.loads(line)["id"]] for line in lines}
    results = {"environment": environment(["jevbench"] + [s for s, _ in targets if s in ENVS]),
               "suite": args.suite,
               "jevbench": {"repo": JEVBENCH_REPO, "commit": JEVBENCH_COMMIT,
                            "tasks": {f: sha for f, (_, sha) in TASK_FILES.items()}},
               "models": {}, "args": vars(args), "runs": {}}
    if args.suite.startswith("legalbench"):
        results["legalbench"] = {"dataset": legalbench.DATASET, "revision": legalbench.REVISION,
                                 "tasks": legalbench.SHA256}

    if args.url:
        results["runs"]["url"] = evaluate("url", args.url, "default", tasks, tiers, out)
    else:
        fetched = {}
        for system in dict.fromkeys(s for s, _ in targets):
            if not SYSTEMS[system].get("remote"):
                print(f"prefetching {system} ...", flush=True)
                fetched[system] = SYSTEMS[system]["prefetch"](args.model)
                results["models"].update(fetched[system]["revisions"])
        for system, device in targets:
            name, spec = f"{system}:{device}", SYSTEMS[system]
            if spec.get("remote"):
                if not os.environ.get(JEV_KEY_ENV):
                    print(f"[{name}] skipped: set {JEV_KEY_ENV}", flush=True)
                    results["runs"][name] = {"label": spec["label"], "error": f"{JEV_KEY_ENV} is not set"}
                    continue
                try:
                    results["runs"][name] = {"label": spec["label"], **evaluate(
                        name, JEV_URL, JEV_MODEL, tasks, tiers, out, key_env=JEV_KEY_ENV,
                        delay_s=args.remote_delay, cost=JEV_HARNESS_COST)}
                except Exception as exc:
                    # Never echo the request (it carries the key); the status line is enough.
                    msg = f"HTTP {exc.response.status_code}" \
                        if isinstance(exc, httpx.HTTPStatusError) else str(exc)
                    print(f"[{name}] failed: {msg}", flush=True)
                    results["runs"][name] = {"label": spec["label"], "error": msg}
                continue
            hosted = spec.get("hosted")
            if hosted and not os.environ.get(hosted["key_env"]):
                print(f"[{name}] skipped: set {hosted['key_env']}", flush=True)
                results["runs"][name] = {"label": spec["label"], "error": f"{hosted['key_env']} is not set"}
                continue
            server = Server(name, spec["cmd"](fetched[system], device, args.port),
                            spec["env"](fetched[system], device), args.port, out)
            try:
                health = server.wait_ready(args.startup_timeout)
                print(f"[{name}] ready", flush=True)
                results["runs"][name] = {"label": spec["label"], "health": health,
                                         **evaluate(name, server.url, spec.get("api_model", system), tasks,
                                                    tiers, out,
                                                    delay_s=args.remote_delay if hosted else 0.0,
                                                    cost=hosted["harness_cost"] if hosted else None)}
            except Exception as exc:  # keep going with the other targets
                print(f"[{name}] failed: {exc}", flush=True)
                results["runs"][name] = {"label": spec["label"], "error": str(exc)}
            finally:
                server.stop()

    results["paired"] = paired(results["runs"])
    (out / "results.json").write_text(json.dumps(scrub(results), indent=2, default=str))
    for f in [*out.glob("*.log"), *out.glob("*/*.json"), *out.glob("*/*.jsonl"), *out.glob("*/*.log")]:
        scrub_file(f)
    print_report(results)
    print(f"\nresults: {out / 'results.json'}")


if __name__ == "__main__":
    main()
