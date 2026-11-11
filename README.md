# Decision-model benchmark

Benchmarks for small typed decision models on Apple silicon:
[strands-decider](https://github.com/strands-labs/strands-decider), Mapika's decider-2b,
Open-Jev-2B, and TypeSafe's hosted Jev for reference. Each one answers yes/no (`noul`),
`choice` and `score` questions over `POST /v1/systemone`. There are two benchmarks:

- **`bench.py`: speed.** It starts each server and measures:
  1. **Sequential latency** on 26 hand-written cases (10 noul, 10 choice, 6 score), one question
     per request, each case sent `--repeats` times. Their accuracy and Brier score are a smoke test
     only, too few cases to rank systems by.
  2. **Multi-question cost**: one long state (~440 tokens) with 1, 2, 4 and 8 questions per
     request. This shows prefix caching: a server that caches reads the state once, and each
     extra question adds only its own tokens.
  3. **Throughput**: `--requests` requests at each `--concurrency` level, giving req/s and p50/p95
     latency.
- **`jevbench_eval.py`: accuracy and calibration** on the 231 public tasks of
  [JevBench](https://github.com/fstandhartinger/jevbench), an independent, MIT-licensed benchmark
  for this class of model. The tasks are scored by JevBench's own harness (tag v1.4.2.2, the
  version behind the v1.4.2 [leaderboard](https://benchmarkheaven.com/jev-models)). It reports:
  - accuracy with a 95% Wilson interval, and accuracy per tier (easy 48, standard 72, hard 111);
  - Brier score and expected calibration error (ECE);
  - per-task latency, one request at a time;
  - for each two systems, the tasks only one of them got right, with an exact McNemar p-value.
    The p-value tells you whether a gap between two systems is more than noise.

## Setup

Each system has its own virtualenv (Python 3.12), built by [uv](https://docs.astral.sh/uv/)
from a committed `uv.lock`. The systems pin conflicting versions of transformers and peft, so they
can't share one environment.

```bash
uv sync                                   # strands-decider (pinned commit, mlx extra), the scripts
uv sync --project systems/decider         # decider-ai 1.10.0
uv sync --project systems/open-jev        # Open-Jev at a pinned commit
uv sync --project systems/jevbench        # JevBench's harness (only for jevbench_eval.py)
```

Before any server starts, the scripts download each checkpoint and its base model, at pinned
Hugging Face commits, into the Hugging Face cache (`~/.cache/huggingface`). Each system needs a
few GB. Set `HF_TOKEN` for faster downloads, or `HF_HOME=$PWD/.hf` to keep the weights
inside the project. `jevbench_eval.py` also downloads the task files into `data/`, at the pinned
commit, and checks their SHA-256 hashes.

## Systems

A target is `SYSTEM:DEVICE`.

| system | model | environment | Mac devices |
|---|---|---|---|
| `strands` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | `.venv` | `mps`, `mlx` |
| `decider` | `Mapika/decider-2b` @ `7614a6c` (decider-ai 1.10.0) | `systems/decider/.venv` | `mps` |
| `openjev` | `ZefanCai/Open-Jev-2B` @ `0c7aa49` | `systems/open-jev/.venv` | `mps` |
| `jev` | TypeSafe Jev, `jev-latest`, hosted at `api.typesafe.ai` | none | `api` |

The configurations differ, so state them alongside any result:

- **decider-2b** runs with images off (`DECIDER_IMAGES=0`), so it serves text only, like the
  others. Its checkpoint enables images by default, and loading the vision tower needs
  torchvision.
- **Open-Jev-2B** uses the Apple-silicon settings from its `docs/deployment.md`: float32, batch
  size 1, no prefix cache, and a 4,096-token context. That is its documented Mac setup, not
  necessarily its fastest. A hard task the server rejects as too long counts as wrong.
- **Jev** is a hosted API. Its latency includes the network round trip from wherever you run
  this. `bench.py` measures that round trip separately: it times a request the API rejects at
  its edge, and reports that time next to Jev's latency. `jev_latency.py` breaks it down into DNS,
  TCP, TLS and edge time. Jev needs `TYPESAFE_API_KEY` in the environment. It is included in the
  default targets only when that variable is set. The key is read from the environment only, and
  is never printed or written to results.

## Run

```bash
uv run bench.py                               # every local target, plus jev:api when TYPESAFE_API_KEY is set
uv run bench.py --targets strands:mlx decider:mps jev:api
uv run bench.py --url http://127.0.0.1:8000   # an already-running server
uv run bench.py --concurrency 1 2 4 8 16 --requests 100 --repeats 5

uv run jevbench_eval.py                       # same default targets
uv run jevbench_eval.py --targets strands:mlx --limit 20     # a smoke test
uv run jevbench_eval.py --report results/jevbench-A results/jevbench-B   # merge saved runs

TYPESAFE_API_KEY=... uv run jev_latency.py    # where Jev's response time goes
```

`bench.py` writes `results/<timestamp>/`. `jevbench_eval.py` writes `results/jevbench-<timestamp>/`,
with the harness's per-task records, raw responses and summary for each target. Every
`results.json` records:

- the machine (chip, model, memory, macOS);
- every package version in each environment used, with git commits;
- the Hugging Face commit of every model;
- the arguments.

The home directory is replaced with `~` in results and server logs.

### Reproducing published numbers

1. Rerun on the same chip, with all targets in one run, so they share the machine's state.
2. Repeat `bench.py` three times, and report the spread between runs, not a single run.
3. `jevbench_eval.py` is deterministic for local models; a second run is a consistency check.
4. Compare `environment` and `models` in your `results.json` with the published one before
   comparing numbers.
5. Speed varies with chip and memory bandwidth, so compare speeds only on the same chip.
6. JevBench's leaderboard composite also includes sealed tasks and its maintainers' own
   speed and cost measurements. This is not that composite: the accuracy here is on the public
   tasks only. It is directly comparable to the board's public-task accuracy.

## Example results

Measured on 11 November 2026 on a Mac mini: Apple M5 Pro, 24 GB, macOS 27.2. Software:
torch 2.15.0 and mlx 0.32.4, strands-decider at `846b69e`, decider-ai 1.10.0, and Open-Jev
at `bd41188`. Jev was called from the same machine, over a network round trip of about
140 ms. These are examples from one machine: rerun them on yours before quoting them.

### Accuracy: JevBench public set (231 tasks)

| system | correct | accuracy (95% CI) | easy | standard | hard | Brier | ECE | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| Jev (hosted, `jev-1.14.0`) | 205 | 88.7% (84.0–92.2) | 100% | 98.6% | 77.5% | 0.145 | 0.069 | 217 / 251 |
| strands-decider-2B, MLX | 181 | 78.4% (72.6–83.2) | 100% | 93.1% | 59.5% | 0.280 | 0.062 | 34 / 398 |
| strands-decider-2B, MPS | 180 | 77.9% (72.1–82.8) | 100% | 93.1% | 58.6% | 0.280 | 0.074 | 104 / 1,073 |
| decider-2b (`v11`), MPS | 180 | 77.9% (72.1–82.8) | 100% | 91.7% | 59.5% | 0.363 | 0.093 | 91 / 992 |
| Open-Jev-2B, MPS | 151 | 65.4% (59.0–71.2) | 100% | 77.8% | 42.3% | 0.475 | 0.129 | 602 / 8,684 |

Brier and ECE: lower is better. Latency is per task, one request at a time, on states of up to
~3,700 tokens.

Paired on the same tasks, the systems fall into three groups:

- Jev is first.
- strands-decider and decider-2b are tied: each gets 25 tasks right that the other gets wrong
  (McNemar p = 1.0).
- Open-Jev is third.

Every difference between groups is significant (p < 0.001). Open-Jev's 65.4% agrees with its
leaderboard entry (0.645).

### Speed: `bench.py` (one run)

| system | p50 ms, 1 question | p50 ms, 8 questions on a ~440-token state | req/s at concurrency 1 / 4 / 8 |
|---|---|---|---|
| strands-decider-2B, MLX | 26 | 152 | 37.2 / 39.4 / 39.6 |
| strands-decider-2B, MPS | 70 | 420 | 14.1 / 14.0 / 14.2 |
| decider-2b, MPS | 46 | 1,990 | 18.6 / 26.6 / 30.7 |
| Open-Jev-2B, MPS | 248 | 5,505 | 5.2 / 5.2 / 5.2 |
| Jev (hosted) | 182 | 178 | 5.0 / 20.6 / not run |

Jev was run at concurrency 1 and 4 only, since TypeSafe publishes no rate limits. Its latency
barely changes with the number of questions: about 140 ms of each request is network time.

## Files

- `bench.py`: server management, the speed benchmark and its report.
- `jevbench_eval.py`: runs JevBench's harness against each target, and reports accuracy, Brier,
  ECE, latency and paired tests.
- `envinfo.py`: records the machine and package versions, and replaces the home directory with
  `~` in results.
- `jev_latency.py`: the network breakdown for the hosted Jev API.
- `workload.py`: the 26 cases and the multi-question workload for `bench.py`.
- `systems/*/`: each competitor's serving environment, and JevBench's harness.

## License

MIT, for the code in this repository. The models and JevBench have their own licenses: the three
open models are Apache-2.0, and JevBench is MIT.
