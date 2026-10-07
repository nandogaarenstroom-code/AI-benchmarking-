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
- Stronger hostile-code isolation validation; the subprocess fallback is bounded but Docker is the
  intended defense-in-depth option
- Richer hardware/model metadata collection, which is currently reported as `Unknown` when it cannot
  be established
- Larger and repeated real-model studies before treating scores as representative outside this
  controlled benchmark

For exact benchmark prompts and scoring limitations, see
[Benchmark Test Documentation](Test_documentation.md). For what the repository's automated test
suite does and does not verify, see [Automated Test Suite](tests/README.md).

## Run

```bash
python app.py
```

Open http://localhost:8000. The built-in **Demo Local Model** permits a complete self-test without
an installed model. For real models, start Ollama, LM Studio, llama.cpp, or another compatible
server before opening the dashboard. Discovery never invents metadata: unavailable values are
reported as `Unknown`.

## Automated behavior

Selecting a model and pressing **RUN BENCHMARK** starts the selected profile without further
questions. Progress is persisted in `results/current.json`; use **RESUME PREVIOUS BENCHMARK** after
an interruption. Completed artifacts are saved under `results/` as `.json`, `.csv`, `.pdf`, and
`.svg`. Every test and failure is retained in the JSON result. Demo outputs are marked
`DEMO DATA` in the result metadata (`demo: true`).

The current dependency-free engine uses deterministic generated tasks and category-specific
evaluators. It provides a stable
adapter/result contract so additional local adapters and sandboxed coding evaluators can be added
without changing the UI. Controlled benchmark measurements are not universal real-world guarantees.

Test prompts are parameterized with seeded values and duplicate fingerprints are rejected. Results
also include three deterministic bootstrap resamples of the measured item scores, including mean
and standard deviation, so a single score is not presented without uncertainty context. Categories
with no applicable tests are reported as `null`/not tested rather than silently scored as zero.

The engine now includes objective numeric, choice, JSON-key, required-term, instruction-length,
and Python-code evaluators. Subjective categories use a clearly labelled deterministic rubric
fallback; a future configured local evaluator model can replace that fallback without changing
the result format. Code is executed only in a temporary isolated working directory with a short
timeout, restricted imports, and no network-capable adapter. When Docker is available, code tests
prefer a network-disabled, read-only container with CPU, memory, process, and timeout limits.

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
- Tool calling

Tests are distributed through four difficulty labels: **Easy**, **Medium**, **Hard**, and
**Expert**. Profiles select the number of generated tests:

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

The current implementation deliberately uses deterministic arithmetic-style prompts across these
categories. Therefore, the category framework and scoring pipeline are present, but this is not
yet a deep domain-specific coding, tool-use, or long-context evaluation. The README describes
what is actually run rather than implying unsupported coverage.

## Automatic scoring

Yes. No manual grading is required. Each answer is compared with its generated expected answer
using objective exact matching:

- Correct answer: 10/10
- Incorrect, empty, or unavailable answer: 0/10
- Category score: percentage of points earned in that category
- Quality score: average across completed test scores, normalized to 0–100
- Performance score: derived from measured request throughput
- Overall score: `quality × 0.70 + performance × 0.30`

Every answer, score, status, and failure is retained in the JSON result. Demo results are clearly
marked with `demo: true` and `DEMO DATA`; they must not be interpreted as model performance.

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
4K, 8K, 16K, 32K, 64K, 128K, and (when supported) 256K target prompt sizes. The accumulated variant deliberately reuses the
conversation history so context degradation can be observed; it is not mixed into the primary
fresh-context quality score.

The suite never blindly exceeds the model's advertised context limit. If metadata reports a
smaller limit, larger sizes are skipped and recorded in `context_skipped`. If the limit is
unknown, the conservative 128K ceiling is used; 256K is only attempted when explicitly
advertised. This makes large-context results transparent rather than turning an overflow into a
misleading quality failure.

Performance runs perform three warm-ups, discard their outputs, then collect repeated latency,
TTFT, generation time, token throughput, P50/P95 latency, standard deviation, and concurrency
samples at levels 1, 2, and 4. These are controlled measurements and should not be treated as
guarantees for unrelated real-world workloads.

Runs can be cancelled through the backend API. Resume state is written atomically, and corrupted
resume files are rejected instead of being executed. Final JSON results are also written through
an atomic replacement.

## Test

```bash
python -m unittest discover -s tests -v
```

See [tests/README.md](tests/README.md) for the test-by-test coverage and known gaps.
