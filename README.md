# Decision-model benchmark

Benchmarks for small typed decision models, on Apple silicon (MLX, MPS) or Linux (CUDA, CPU):
[strands-decider](https://github.com/strands-labs/strands-decider), Mapika's decider-2b,
Open-Jev-2B, and two hosted models for reference: TypeSafe's Jev and Google's Gemini Decision 1.
Each one answers yes/no (`noul`), `choice` and `score` questions over `POST /v1/systemone`.

## Structure

The benchmark has two independent parts, each with its own script. Either can be run on its own.

| part | script | task set | measures |
|---|---|---|---|
| 1. Speed | `bench.py` | our own: 26 hand-written cases and a multi-question workload | latency, prefix caching, throughput |
| 2. Accuracy and calibration | `jevbench_eval.py` | [JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public tasks, and 915 [LegalBench](https://hazyresearch.stanford.edu/legalbench) cases | accuracy, Brier, ECE, paired tests |

### 1. Speed: `bench.py`

It starts each server and measures:

1. **Sequential latency** on 26 hand-written cases (10 noul, 10 choice, 6 score), one question
   per request, each case sent `--repeats` times. Their accuracy and Brier score are a smoke test
   only, too few cases to rank systems by.
2. **Multi-question cost**: one long state (~440 tokens) with 1, 2, 4 and 8 questions per
   request. This shows prefix caching: a server that caches reads the state once, and each
   extra question adds only its own tokens.
3. **Throughput**: `--requests` requests at each `--concurrency` level, giving req/s and p50/p95
   latency.

### 2. Accuracy and calibration: `jevbench_eval.py`

It runs the 231 public tasks of [JevBench](https://github.com/fstandhartinger/jevbench), an
independent, MIT-licensed benchmark for this class of model. The tasks are scored by JevBench's
own harness (tag v1.4.2.2, the version behind the v1.4.2
[leaderboard](https://benchmarkheaven.com/jev-models)). It reports:

- accuracy with a 95% Wilson interval, and accuracy per tier (easy 48, standard 72, hard 111);
- Brier score and expected calibration error (ECE);
- per-task latency, one request at a time;
- for each two systems, the tasks only one of them got right, with an exact McNemar p-value.
  The p-value tells you whether a gap between two systems is more than noise.

About JevBench, keep in mind:

- **Contamination.** The public tasks are on GitHub, so a model may have been trained or tuned on
  them, which would inflate its score. This can't be checked from outside.
- **Origin.** The question format comes from TypeSafe's Jev, and JevBench was built around it.
  This may favour Jev and the models closest to it.

#### The second set: LegalBench (`--suite legalbench`)

To check the JevBench ranking against questions nobody here wrote, `legalbench.py` converts eight
tasks from [LegalBench](https://hazyresearch.stanford.edu/legalbench) (Guha et al., 2023) into
JevBench's task format, so the same harness scores them. LegalBench was written and labelled by
lawyers and legal scholars, not for decision models. Each of the eight tasks states a rule and
asks for its application to a short case:

| task | type | cases | question |
|---|---|---|---|
| `hearsay` | noul | 94 | Is the evidence hearsay? |
| `personal_jurisdiction` | noul | 50 | Does the court have personal jurisdiction? |
| `diversity_1` to `_6` | noul | 6 × 50 | Is there diversity jurisdiction? (six levels of complexity) |
| `proa` | noul | 95 | Does the statute create a private right of action? |
| `jcrew_blocker` | noul | 54 | Does the loan provision contain a J.Crew blocker? |
| `ucc_v_common_law` | choice, 2 | 94 | Is the contract governed by the UCC or the common law? |
| `insurance_policy_interpretation` | choice, 3 | 133 | Is the claim covered, not covered, or is it ambiguous? |
| `abercrombie` | score, 5 | 95 | Where is the trademark on the spectrum of distinctiveness? |

- Every case is asked zero-shot. The rule, from the task's own description, goes in the
  question's instructions and the case in the state. LegalBench's few-shot examples are not used.
- The diversity variants have 300 cases each. A fixed 50 of each are kept (chosen by hash), so
  that they don't outweigh the other tasks.
- The test files are downloaded from a pinned revision of the Hugging Face copy and checked by
  hash. All eight tasks are CC BY 4.0.

LegalBench has its own limits. It has been public since 2023, so it may be in the hosted models'
training data. It covers one domain: legal rule application. And one task's wording is
ambiguous; see the J.Crew note under the results.

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

| system | model | environment | devices |
|---|---|---|---|
| `strands` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | `.venv` | `mlx`, `mps`, `cuda`, `cpu` |
| `decider` | `Mapika/decider-2b` @ `7614a6c` (decider-ai 1.10.0) | `systems/decider/.venv` | `mps`, `cuda`, `cpu` |
| `openjev` | `ZefanCai/Open-Jev-2B` @ `0c7aa49` | `systems/open-jev/.venv` | `mps`, `cuda`, `cpu` |
| `jev` | TypeSafe Jev, `jev-latest`, hosted at `api.typesafe.ai` | none | `api` |
| `gemini` | Google Gemini Decision 1, `gemini-decision-1`, hosted, through `gemini_proxy.py` | `.venv` | `api` |

The default targets depend on the platform:

- **macOS:** `strands:mps strands:mlx decider:mps openjev:mps`.
- **Linux:** each local system on `cuda` when `nvidia-smi` is found, otherwise on `cpu`. CPU
  is too slow for meaningful speed numbers, but it is enough to check that everything is set up.

The same lock files install on both platforms: MLX and decider-ai's Metal kernel install only
on Apple-silicon macOS. On Linux, PyPI's torch wheels bring their own CUDA runtime, so you
need only an NVIDIA driver.

The configurations differ, so state them alongside any result:

- **decider-2b** runs with images off (`DECIDER_IMAGES=0`), so it serves text only, like the
  others. Its checkpoint enables images by default, and loading the vision tower needs
  torchvision.
- **Open-Jev-2B** uses the reference settings from its `docs/deployment.md`: batch size 1, no
  prefix cache and a 4,096-token context, plus float32 on MPS, as its docs require there. That is
  its documented setup, not necessarily its fastest. A hard task the server rejects as too long counts as wrong.
- **Jev** is a hosted API. Its latency includes the network round trip from wherever you run
  this. `bench.py` measures that round trip separately: it times a request the API rejects at
  its edge, and reports that time next to Jev's latency. `jev_latency.py` breaks it down into DNS,
  TCP, TLS and edge time. Jev needs `TYPESAFE_API_KEY` in the environment. It is included in the
  default targets only when that variable is set. The key is read from the environment only, and
  is never printed or written to results.
- **Gemini Decision 1** is Google's hosted Decisions API, which has its own request format.
  `gemini_proxy.py` translates: it serves `POST /v1/systemone` on localhost and forwards each
  request to Google. The format follows the Vercel AI SDK's Google provider, the API's first
  public client. Three things differ from Jev:
  - a JSON state is sent as JSON text, as that client does;
  - every question needs instructions, and choice and score questions need 2 to 255 options;
  - a request Google refuses counts as wrong.

  Its latency includes the network round trip, measured the same way as Jev's, plus the
  translator's own time. `bench.py` reports that time separately; it is under 1 ms. It is in
  the default targets only when `GEMINI_API_KEY` is set. Use a paid-tier key: Google may
  use free-tier requests to improve its products, and the free tier's rate limits are low.
  JevBench prices its runs at the paid tier's $0.10 per million input tokens.

## Run

```bash
uv run bench.py                               # every local target, plus jev:api when TYPESAFE_API_KEY is set
uv run bench.py --targets strands:mlx decider:mps jev:api
uv run bench.py --url http://127.0.0.1:8000   # an already-running server
uv run bench.py --concurrency 1 2 4 8 16 --requests 100 --repeats 5

uv run jevbench_eval.py                       # same default targets
uv run jevbench_eval.py --targets strands:mlx --limit 20     # a smoke test
uv run jevbench_eval.py --report results/jevbench-A results/jevbench-B   # merge saved runs
uv run jevbench_eval.py --suite legalbench    # the second set: 915 LegalBench cases

TYPESAFE_API_KEY=... uv run jev_latency.py    # where Jev's response time goes

GEMINI_API_KEY=... uv run bench.py --targets gemini:api --remote-concurrency 1 4
GEMINI_API_KEY=... uv run jevbench_eval.py --targets gemini:api
```

`bench.py` writes `results/<timestamp>/`. `jevbench_eval.py` writes `results/<suite>-<timestamp>/`,
with the harness's per-task records, raw responses and summary for each target. Every
`results.json` records:

- the machine (OS, CPU or chip, memory, NVIDIA GPUs and driver);
- every package version in each environment used, with git commits;
- the Hugging Face commit of every model;
- the arguments.

The home directory is replaced with `~` in results and server logs.

### Reproducing published numbers

1. Rerun on the same hardware, with all targets in one run, so they share the machine's state.
2. Repeat `bench.py` three times, and report the spread between runs, not a single run.
3. `jevbench_eval.py` is deterministic for local models, on either suite; a second run is a
   consistency check.
4. Compare `environment` and `models` in your `results.json` with the published one before
   comparing numbers.
5. Speed varies with chip and memory bandwidth, so compare speeds only on the same hardware.
6. JevBench's leaderboard composite also includes sealed tasks and its maintainers' own
   speed and cost measurements. This is not that composite: the accuracy here is on the public
   tasks only. It is directly comparable to the board's public-task accuracy.

## Example results

Measured on 11–12 November 2026 on a Mac mini: Apple M5 Pro, 24 GB, macOS 27.2. Software:
torch 2.15.0 and mlx 0.32.4, strands-decider at `846b69e`, decider-ai 1.10.0, and Open-Jev
at `bd41188`. Jev was called from the same machine, over a network round trip of about
140 ms during the JevBench run and 155 ms during the speed runs. Gemini Decision 1
(`gemini-decision-1`, a day after its release) was called from the same machine on 12 November,
over a round trip of about 117 ms. These are examples from one machine: rerun them on yours
before quoting them.

### Accuracy: JevBench public set (231 tasks)

| system | correct | accuracy (95% CI) | easy | standard | hard | Brier | ECE | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| Gemini Decision 1 (hosted) | 213 | 92.2% (88.0–95.0) | 100% | 100% | 83.8% | 0.118 | 0.035 | 248 / 378 |
| Jev (hosted, `jev-1.14.0`) | 205 | 88.7% (84.0–92.2) | 100% | 98.6% | 77.5% | 0.145 | 0.069 | 217 / 251 |
| strands-decider-2B, MLX | 181 | 78.4% (72.6–83.2) | 100% | 93.1% | 59.5% | 0.280 | 0.062 | 34 / 398 |
| strands-decider-2B, MPS | 180 | 77.9% (72.1–82.8) | 100% | 93.1% | 58.6% | 0.280 | 0.074 | 104 / 1,073 |
| decider-2b (`v11`), MPS | 180 | 77.9% (72.1–82.8) | 100% | 91.7% | 59.5% | 0.363 | 0.093 | 91 / 992 |
| Open-Jev-2B, MPS | 151 | 65.4% (59.0–71.2) | 100% | 77.8% | 42.3% | 0.475 | 0.129 | 602 / 8,684 |

Brier and ECE: lower is better. Latency is per task, one request at a time, on states of up to
~3,700 tokens.

Paired on the same tasks, the systems fall into three groups:

- The two hosted models are first. Gemini Decision 1 gets 12 tasks right that Jev gets wrong,
  and Jev 4 that Gemini gets wrong. That leans towards Gemini, but is not significant
  (McNemar p = 0.077).
- strands-decider and decider-2b are tied: each gets 25 tasks right that the other gets wrong
  (McNemar p = 1.0).
- Open-Jev is third.

Every difference between groups is significant (p < 0.001). Two results agree with published
numbers: Open-Jev's 65.4% with its leaderboard entry (0.645), and Gemini's 92.2% with the
public-split score in Google's launch announcement.

Cost of the 231 tasks, from the tokens each API reported:

- **Gemini:** 158,532 input tokens at $0.10 per million, $0.016.
- **Jev:** 211,309 input tokens at $0.042 per million, $0.009. That price is from OpenRouter's
  listing, not TypeSafe's own price list.

Output is free on both. The two count tokens differently, so compare the costs, not the counts.

### Accuracy: LegalBench (915 cases)

Run on 12 November 2026, same machine. Jev was rerun on its own a few minutes later (see the
harness note below), and the two runs were merged with `--report`.

| system | correct | accuracy (95% CI) | noul | choice | score | Brier | ECE | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| Jev (hosted, `jev-1.14.0`) | 733 | 80.1% (77.4–82.6) | 84.0% | 70.9% | 77.9% | 0.280 | 0.020 | 228 / 415 |
| Gemini Decision 1 (hosted) | 717 | 78.4% (75.6–80.9) | 77.1% | 78.9% | 85.3% | 0.274 | 0.059 | 410 / 549 |
| strands-decider-2B, MPS | 603 | 65.9% (62.8–68.9) | 73.0% | 64.3% | 25.3% | 0.433 | 0.043 | 110 / 135 |
| strands-decider-2B, MLX | 601 | 65.7% (62.5–68.7) | 72.5% | 64.3% | 26.3% | 0.433 | 0.047 | 37 / 51 |
| decider-2b (`v11`), MPS | 529 | 57.8% (54.6–61.0) | 62.4% | 62.1% | 18.9% | 0.688 | 0.287 | 74 / 295 |
| Open-Jev-2B, MPS | 512 | 56.0% (52.7–59.1) | 59.4% | 59.5% | 26.3% | 0.567 | 0.187 | 212 / 785 |

Accuracy per task:

| task | strands MPS | strands MLX | decider-2b | Open-Jev | Jev | Gemini |
|---|---|---|---|---|---|---|
| abercrombie | 25.3% | 26.3% | 18.9% | 26.3% | 77.9% | 85.3% |
| diversity_1 | 96.0% | 96.0% | 46.0% | 80.0% | 100% | 100% |
| diversity_2 | 76.0% | 74.0% | 46.0% | 76.0% | 100% | 100% |
| diversity_3 | 56.0% | 56.0% | 80.0% | 28.0% | 94.0% | 90.0% |
| diversity_4 | 76.0% | 76.0% | 64.0% | 48.0% | 94.0% | 74.0% |
| diversity_5 | 64.0% | 64.0% | 52.0% | 44.0% | 82.0% | 72.0% |
| diversity_6 | 62.0% | 62.0% | 48.0% | 40.0% | 86.0% | 48.0% |
| hearsay | 72.3% | 71.3% | 70.2% | 68.1% | 78.7% | 72.3% |
| insurance_policy_interpretation | 42.9% | 42.9% | 39.8% | 40.6% | 51.9% | 65.4% |
| jcrew_blocker | 79.6% | 77.8% | 61.1% | 25.9% | 16.7% | 16.7% |
| personal_jurisdiction | 48.0% | 48.0% | 56.0% | 72.0% | 86.0% | 94.0% |
| proa | 87.4% | 87.4% | 78.9% | 84.2% | 98.9% | 95.8% |
| ucc_v_common_law | 94.7% | 94.7% | 93.6% | 86.2% | 97.9% | 97.9% |

Compared with JevBench:

- **Every system scores lower,** by 9 to 20 points.
- **The hosted models are still first,** but now Jev is ahead: 89 cases right that Gemini gets
  wrong, against 73. Not significant (McNemar p = 0.24), so on both sets the two are level.
- **strands-decider pulls ahead of decider-2b.** They were tied on JevBench; here strands gets
  165 cases right that decider-2b gets wrong, against 91 (p < 0.001).
- **decider-2b and Open-Jev are tied** (181 against 164, p = 0.39). On JevBench Open-Jev was
  clearly last.
- **The three small models can't do the 5-level score task.** On `abercrombie` they score 19–26%,
  about chance (20%); the hosted models score 78–85%. The small models never pick either end of
  the scale: strands, for example, calls "Salt" for sodium chloride descriptive, not generic.
  The harness takes the most likely level, so this is the model, not the conversion.
- **Calibration:** Jev's ECE (0.020) is the best on this set; strands' (0.043) is close.
  decider-2b's (0.287) is the worst: it is often confidently wrong.

Notes on this run:

- **J.Crew blockers.** Both hosted models answer "no" to nearly every provision, and most are
  labelled "yes". The task's rule says a blocker "typically includes (1) … and (2) …", which
  they read as requiring both parts; most "yes" provisions have only the first. This is
  ambiguity in the task, not a bug, and it costs the hosted models about 45 cases each.
- **One Gemini failure.** One case got a 503 "service unavailable" from Google. The harness
  doesn't retry it and counts the case as wrong.
- **Jev's first run stopped at 750 cases.** The harness books a placeholder for each request to
  an API with no known price, and stops at a $15 cap. The script now gives Jev a per-token
  price (OpenRouter's $0.042 per million), so the harness books its real cost.

Cost of the 915 cases, from the tokens each API reported:

- **Gemini:** 198,288 input tokens at $0.10 per million, $0.020.
- **Jev:** 383,916 input tokens at $0.042 per million, $0.016.

### Speed: `bench.py` (three runs)

Three consecutive runs on 12 November 2026, with the mini's other services paused. Each cell is
the median of the three runs, with the lowest and highest in brackets where they differ.

| system | p50 ms, 1 question | p50 ms, 8 questions on a ~440-token state | req/s, concurrency 1 | req/s, concurrency 4 | req/s, concurrency 8 |
|---|---|---|---|---|---|
| strands-decider-2B, MLX | 27 | 153 (153–154) | 36.6 (36.0–37.0) | 38.4 (38.2–38.8) | 38.7 (38.6–38.8) |
| strands-decider-2B, MPS | 72 (71–72) | 423 (421–424) | 13.9 (13.9–14.0) | 14.0 (14.0–14.1) | 14.0 (13.9–14.0) |
| decider-2b, MPS | 46 (46–47) | 2,019 (2,019–2,028) | 18.5 (18.4–18.6) | 27.2 (26.1–29.9) | 31.2 (31.1–31.2) |
| Open-Jev-2B, MPS | 253 (252–254) | 5,525 (5,523–5,533) | 5.2 (5.1–5.2) | 5.2 | 5.2 |
| Jev (hosted) | 193 (190–194) | 207 (199–210) | 5.1 (5.0–5.2) | 19.5 (19.5–19.7) | not run |
| Gemini Decision 1 (hosted) | 233 (230–240) | 249 (248–253) | 4.0 (3.9–4.1) | 15.3 (14.8–15.8) | not run |

The runs agree to within a few percent, except decider-2b at concurrency 4, which varies by
about 15%. The hosted models were run at concurrency 1 and 4 only, to stay well inside their
rate limits. Their latency barely changes with the number of questions, and much of it is
network time:

- **Jev:** about 155 ms of 193 ms is the round trip to TypeSafe, leaving roughly 40 ms past its edge.
- **Gemini:** about 117 ms of 233 ms is the round trip to Google, leaving roughly 115 ms past
  its edge. The local translator adds 1–2 ms.

Gemini's speed runs were made later the same night, without pausing the mini's other services.
Those barely affect a hosted model, whose time is spent on the network and on Google's servers.

## Files

- `bench.py`: server management, the speed benchmark and its report.
- `jevbench_eval.py`: runs JevBench's harness against each target, and reports accuracy, Brier,
  ECE, latency and paired tests.
- `legalbench.py`: downloads the eight LegalBench tasks and converts them to JevBench's format.
- `envinfo.py`: records the machine and package versions, and replaces the home directory with
  `~` in results.
- `jev_latency.py`: the network breakdown for the hosted Jev API.
- `gemini_proxy.py`: serves Google's Gemini Decisions API as `POST /v1/systemone`.
- `workload.py`: the 26 cases and the multi-question workload for `bench.py`.
- `systems/*/`: each competitor's serving environment, and JevBench's harness.

## License

MIT, for the code in this repository. The models and task sets have their own licenses: the three
open models are Apache-2.0, JevBench is MIT, and the eight LegalBench tasks are CC BY 4.0
(Guha et al., "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in
Large Language Models", 2023). The tasks are downloaded at run time, not stored here.
