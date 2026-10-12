# Local AI Benchmark

An offline, zero-interaction benchmark dashboard for local LLM servers. It discovers Ollama,
LM Studio, and common OpenAI-compatible endpoints, then runs deterministic tests, scores them,
and writes JSON, CSV, PDF, and SVG chart artifacts.

## Project status

### In place

- A dependency-free local HTTP server and browser dashboard
- An offline voxel laboratory that visualizes quality, performance, context, and category scores
  while retaining exact values in an accessible table
- Discovery for Ollama, LM Studio, and OpenAI-compatible endpoints, plus a clearly marked demo model
- Seeded benchmark generation, automatic scoring, retries, cancellation, and resumable progress
- Fresh and accumulated context experiments with advertised-limit handling
- Warm-ups, latency/throughput measurements, and concurrency samples
- JSON, CSV, PDF, and SVG result exports and result comparison
- A small automated regression suite for core evaluators, deterministic generation, demo execution,
  exports, context metadata, and comparison
- A local, version-aware calibration-study protocol and CSV analysis CLI for separate human/model
  cohorts; it analyzes supplied observations but does not claim empirical calibration without data

### Still needed for broader production confidence

- More varied, domain-specific benchmark datasets; several categories currently reuse simple task
  families and do not justify broad capability claims
- Stricter evaluators for exact instruction following, semantic quality, creative quality, and full
  JSON Schema validation
- A configured local evaluator model or human-calibrated rubric for subjective answers
- Verified adapters and integration tests against multiple real server versions and malformed or
  non-streaming responses
- Automated coverage for the HTTP API, dashboard behavior, cancellation/resume failure paths,
  report contents, and concurrent benchmark requests
- Richer hardware/model metadata collection, which is currently reported as `Unknown` when it cannot
  be established
- Larger and repeated real-model studies before treating scores as representative outside this
  controlled benchmark

For exact benchmark prompts and scoring limitations, see
[Benchmark Test Documentation](Test_documentation.md). For what the repository's automated test
suite does and does not verify, see [Automated Test Suite](tests/README.md).
For the calibration pilot protocol, CSV schema, and analysis method, see
[Original Task Calibration Pilot](CALIBRATION_STUDY.md).

Analyze collected, de-identified calibration observations with:

```bash
python tools/calibration.py path/to/responses.csv --output path/to/report.json
```

Pass independent pre-study item judgments with `--item-reviews path/to/item-reviews.csv`.
See [`CALIBRATION_STUDY.md`](CALIBRATION_STUDY.md) for both CSV schemas and data-handling guidance.

The analyzer keeps task/evaluator versions, model settings, categories, and human/model cohorts
separate. It reports preliminary evidence only; no human or model calibration data is bundled with
this repository.

## Run

```bash
python app.py
```

Open http://localhost:8000. The built-in **Demo Local Model** permits a complete self-test without
an installed model. For real models, start Ollama, LM Studio, llama.cpp, or another compatible
server before opening the dashboard. Discovery never invents metadata: unavailable values are
reported as `Unknown`.

On Windows, the dashboard can also start an app-managed llama.cpp server. Choose an existing
`llama-server.exe` path or install the fixed `ggml.llamacpp` package through winget, then enter the
full path to a local `.gguf` model, a requested context size, and GPU layer count. The app binds that
server to `127.0.0.1:8080`, waits for its health endpoint, and offers a stop control; a server already
using that port must be stopped first. The app starts only the process it owns and stops it when the
app exits normally. It does not download model files. Only select an executable you trust: it runs
with the same operating-system permissions as the benchmark app.

The displayed startup context and GPU-layer values are the requested command-line settings; they do
not guarantee the model or available memory can support them. A high context or offload request may
fail during model loading. The saved benchmark result separately records this app-managed request
and any runtime values llama.cpp actually reports through `/props`. The launcher is Windows-only;
on other platforms, start llama.cpp yourself and refresh model discovery.

## Automated behavior

Selecting a model and pressing **Run benchmark** starts the selected profile without further
questions. The dashboard reports warm-up, primary tests, fresh and accumulated context tests,
performance measurements, and file-saving phases, with phase progress and pass/fail counts.
Progress is persisted in `results/current.json`; use **Resume previous run** after an interruption.
Completed files are saved under `results/` with a generated name such as
`benchmark_<timestamp>_<id>.json`, plus matching `.csv`, `.pdf`, and `_quality.svg` files. The
dashboard shows the results folder, saved-run history, and download links. Every test and failure is
retained in the JSON result. Demo outputs are marked `DEMO DATA` in the result metadata (`demo: true`).
Select a model deliberately before the run control is enabled; the demo model is never selected
automatically. Active runs can be cancelled and resumed from their most recently checkpointed phase.
Completed results include a **Test-by-test results** section with difficulty and outcome filters,
the exact prompt, model answer, evaluation rule, reason for the result, and retry/repeat records.
Each test also has a model-relative solve meter based on its evaluator score and completed repeats.
It describes only the selected model's result under that run's settings: it is not a calibrated or
universal difficulty estimate. Demo answers are explicitly marked as demo-only, and tests that could
not be evaluated are marked not measured rather than counted as failures.
Checks that could not run (for example, unavailable Docker or a model-server timeout) are
distinguished from incorrect answers and excluded from quality scoring. If there are no valid
measurements, the affected score is shown as unavailable rather than as zero.
The interface reports observable answers and evaluator evidence; it does not ask models to disclose
hidden chain-of-thought.

Before starting, the dashboard lets you set temperature, top-p, top-k, min-p, repeat penalty,
maximum output tokens, and the sampling seed. Defaults favor repeatable evaluation (`temperature: 0`,
`top_p: 1`, `seed: 12345`); settings are included in saved result configuration and passed to
Ollama or OpenAI-compatible adapters. When llama.cpp reports its startup generation defaults through
`/props`, the dashboard displays those values on the model card and loads them into the generation
controls. Edited controls are sent as request-level overrides and the actual benchmark settings are
recorded with each result. The llama.cpp context window is configured when launching `llama-server`
(for example, with `-c`); it cannot be changed for an individual request here.
Choose a **Repeatable**, **Balanced**, or **Creative** preset on the main screen; individual
sampling controls and optional 2–5 repeats of the primary test suite are collapsed under
**Advanced settings**. Repeats use consecutive sampling seeds and report per-run quality and its
standard deviation. Context-scaling and performance experiments still run once. Available server
version, context length, and GPU-offload information are shown as read-only runtime details; values
the local server does not report remain `Unknown`.

The current dependency-free engine uses deterministic generated tasks and category-specific
evaluators. It provides a stable
adapter/result contract so additional local adapters and sandboxed coding evaluators can be added
without changing the UI. Controlled benchmark measurements are not universal real-world guarantees.

Test prompts are parameterized with seeded values and duplicate fingerprints are rejected. Results
also include three deterministic bootstrap resamples of the measured item scores, including mean
and standard deviation. These resamples describe item-level variability; they are not independent
model runs. When repeated runs are requested, between-run scores are reported separately. Categories
with no applicable tests are reported as `null`/not tested rather than silently scored as zero.

The engine now includes objective numeric, choice, JSON-key, required-term, instruction-length,
and Python-code evaluators. Subjective categories use deterministic checks or finite acceptable
answers; they do not measure semantic quality. No evaluator model is configured. Any future
semantic judge should be reported separately from the deterministic score until validated against
blinded human ratings. Model-generated code runs only in a resource-limited, network-disabled Docker
container. If Docker or the preloaded `python:3-alpine` image is unavailable, coding evaluations are
marked **NOT RUN**; the app never falls back to executing model code on the host. Caller-supplied test
code is ignored.

The HTTP server binds to loopback, limits request size and concurrency, and requires a per-process
CSRF token plus same-origin requests for state-changing API operations. Benchmark model endpoints
must match a locally discovered server, and endpoint redirects are not followed. Result filenames
use generated identifiers rather than model names.

Real adapters use streaming requests where supported. The result records an estimated TTFT from
the first streamed content chunk, generation time after that chunk, estimated prompt tokens, and
output tokens. Servers that do not provide compatible streaming responses are recorded as adapter
failures rather than silently presenting a fabricated streaming metric.

## What models are tested on?

Every generated test has an ID, category, subcategory, difficulty, prompt, expected answer,
evaluation method, and maximum score. The current categories are:

- Reasoning
- Mathematics
- Coding
- Instruction following
- Structured output
- Knowledge
- Hallucination resistance
- Long-context reasoning
- Summarization
- Multilingual capability
- Creative generation
- Consistency
- Context retrieval

Tool calling is **not** currently a generated test category. The discovered `tools` capability field
is metadata only; no model is asked to invoke a tool or graded on tool-use behavior.

Tests are distributed through **Easy**, **Medium**, **Hard**, and **Uncalibrated** task-construction
tiers. The final label is explicit: the tasks have not been piloted against a human panel or a
representative model cohort, so we do not claim that they are empirically expert-hard. Profiles select
the number of generated tests:

| Profile | Tests |
| --- | ---: |
| QUICK | 25 |
| STANDARD | 100 |
| FULL | 252 |
| PERFORMANCE | 25 |
| QUALITY | 100 |
| REASONING | 40 |
| CODING | 40 |
| LONG CONTEXT | 30 |

Difficulty tiers appear in every profile, including `QUICK`. Every saved test and result records the
task-set version, evaluator version, tier rationale, and calibration status. Compare results only
within the same category, task-family subcategory, and task-set version; these difficulty tiers are
not calibrated estimates.

The Reasoning, Coding, and Summarization categories now include original task variants: route planning
with varied constraints and satisfiable/unsatisfiable cases; both inventory-event reconciliation and
opening-stock adjustment functions with generated edge-case assertions; and incident handoffs across
file-upload, checkout, and notification scenarios. Reasoning and summarization expose per-criterion
checks and partial scores; code runs in the existing resource-limited Docker evaluator. Calibration
analysis keeps category subcategories separate. Exact-response evaluators reject extra contradictory
text, and instruction/creative rubrics show lexical relevance evidence without claiming semantic
judgment. These focused tasks do not establish general reasoning, software engineering, summarization,
instruction-following, or creative ability. The category framework and scoring pipeline also do not
yet amount to deep tool-use or long-context evaluation.

## Automatic scoring

No manual grading is required for the current deterministic checks. Each answer is evaluated with the
applicable check (such as numeric matching, required terms, JSON keys and types, fact fields,
constraint satisfaction, or sandboxed assertions). The test details show the criteria and observable
evidence. These automatic checks are proxies for specified task behavior, not universal judgments of
answer quality.

- Fully correct answer: 10/10
- Incorrect answer: 0/10, except the reasoning and summarization pilots award proportional credit
  across their disclosed criteria
- Unavailable evaluator or model response: `NOT RUN`, excluded from quality scoring
- Category score: percentage of points earned in that category
- Quality score: average across completed test scores, normalized to 0–100
- Performance score: derived from measured request throughput
- Overall score: `quality × 0.70 + performance × 0.30`

Every answer, score, status, and failure is retained in the JSON result. Demo results are clearly
marked with `demo: true` and `DEMO DATA`; they must not be interpreted as model performance.
The dashboard and CSV export also include per-test prompts, answers, evaluator criteria, and outcome
explanations. “How it did it” means the observable response and checks; private chain-of-thought is
not requested or treated as evidence.

## Context handling and fairness

Each test starts a fresh independent request. The Ollama request sends a new prompt without a
conversation history, and the OpenAI-compatible request contains only the current user message.
Results record `context_reset: true`, and the report configuration records
`context_reset_per_test: true`. This prevents an earlier test's conversation from filling the
model context and artificially lowering later scores.

That reset is intentional for isolated capability tests, but it does **not** claim that the model
will always behave this way in a real application. Context accumulation is measured separately from
fresh-context quality rather than being mixed into the primary quality score.

Each run also records a separate context-scaling result with fresh and accumulated variants at
4K, 8K, 16K, 32K, 64K, 128K, and (when supported) 256K target prompt sizes. Marker positions rotate
between beginning, middle, and end. Prompt sizes are estimated from text length rather than measured
with each model's tokenizer, and each size/position is sampled only once. The accumulated variant deliberately reuses the
conversation history so context degradation can be observed; it is not mixed into the primary
fresh-context quality score.

The suite never blindly exceeds the model's advertised context limit. If metadata reports a
smaller limit, larger sizes are skipped and recorded in `context_skipped`. If the limit is
unknown, the conservative 128K ceiling is used; 256K is only attempted when explicitly
advertised. This makes large-context results transparent rather than turning an overflow into a
misleading quality failure.

Performance runs perform three warm-ups, discard their outputs, then collect latency samples,
streamed TTFT and generation time when available, and concurrency samples at levels 1, 2, and 4.
Concurrency results include request success/failure counts, elapsed time, mean latency, and estimated
throughput; they use only the selected profile's first five test prompts and are a small local sample.
Throughput is estimated output tokens divided by the measured generation time of dedicated
performance samples; token counts are estimates rather than tokenizer-reported counts. The result
labels estimated timing when the adapter does not report streamed timing. P50/P95 latency and
standard deviation are based on measured request latency. These are controlled measurements and
should not be treated as guarantees for unrelated real-world workloads.

Runs can be cancelled through the backend API. Resume state is written atomically, and corrupted
resume files are rejected instead of being executed. Final JSON results are also written through
an atomic replacement.

## Test

```bash
python -m unittest discover -s tests -v
```

See [tests/README.md](tests/README.md) for the test-by-test coverage and known gaps.
