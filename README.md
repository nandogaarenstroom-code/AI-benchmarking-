# Local AI Benchmark

An offline, zero-interaction benchmark dashboard for local LLM servers. It discovers Ollama,
LM Studio, and common OpenAI-compatible endpoints, then runs deterministic tests, scores them,
and writes JSON, CSV, PDF, and SVG chart artifacts.

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

The current dependency-free engine uses deterministic exact-match tasks. It provides a stable
adapter/result contract so additional local adapters and sandboxed coding evaluators can be added
without changing the UI. Controlled benchmark measurements are not universal real-world guarantees.

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
will always behave this way in a real application. Context accumulation is itself an important
property, so a future context-scaling profile should explicitly add controlled 4K/8K/16K/etc.
prompts and report accuracy, latency, and failure rate by context size. Those measurements should
be kept separate from fresh-context quality scores rather than mixed into them.

## Test

```bash
python -m unittest discover -s tests -v
```
