# Benchmark Test Documentation

This document describes the tests currently implemented by the benchmark engine. It documents
what is actually executed; category names do not imply deeper coverage than the prompts and
evaluators described below.

## Test record

Every generated test contains:

- `id`
- `category`
- `subcategory`
- `difficulty`
- `prompt`
- `expected_answer` or expected terms/schema
- `evaluation_method`
- `max_score` (`10`)

Tests are generated deterministically from a seed. Prompt fingerprints are checked so duplicate
prompts are avoided. Every profile includes **Easy**, **Medium**, **Hard**, and **Expert** tiers,
including `QUICK`. Tiers are operational descriptions of task construction, not a standardized,
independently calibrated difficulty scale. Mathematics progresses from single-digit addition to
grouped multi-step arithmetic. Coding progresses from a basic operation to preserving order while
deduplicating values with edge-case checks.

Each result records the rationale for an individual test's tier. Compare difficulty primarily within
the same category: task types vary, so an Expert knowledge-recall item is not necessarily as hard as
an Expert coding item.

## Profiles

| Profile | Number of primary tests |
| --- | ---: |
| `QUICK` | 25 |
| `STANDARD` | 100 |
| `FULL` | 252 |
| `PERFORMANCE` | 25 |
| `QUALITY` | 100 |
| `REASONING` | 40 |
| `CODING` | 40 |
| `LONG CONTEXT` | 30 |

The primary tests are fresh-context tests: each request contains only the current prompt.

## Primary test categories

The generator rotates through these categories. The task family and current evaluation behavior
are documented exactly below.

### Mathematics

**Prompt:** Seeded arithmetic varies by tier: single-digit addition (Easy), two-digit
multiplication (Medium), multiplication plus addition (Hard), and grouped multi-operation
arithmetic (Expert).

**Evaluation:** Extract the first numeric value from the response and compare it with the generated
answer using zero tolerance.

### Reasoning

**Prompt:** Easy items apply one rule; higher tiers chain category rules and exclusions.

**Evaluation:** The response must contain every displayed required term. This checks the stated
conclusion words, not a free-form proof.

### Coding

**Prompt:** Tasks progress from a basic arithmetic function to collection processing and stable
deduplication.

**Evaluation:** The returned source is combined with the test's displayed sandbox assertions. For
example:

```python
assert add(2, 3) == 5
assert add(-1, 1) == 0
```

The code evaluator returns a pass only when all assertions complete successfully. A failed run says
one or more assertions failed; process output is suppressed, so the UI does not claim to identify
which assertion failed. Code execution is bounded by source length and timeout and occurs only in a
network-disabled, read-only Docker container with CPU, memory, and process limits. If Docker is
unavailable, the test is marked **NOT RUN**, not a model failure. Generated code should not be treated
as safe for hostile workloads unless container isolation is available and verified.

### Structured Output

**Prompt:** Ask for JSON with required keys and value types; higher tiers require more fields.

**Evaluation:** Parse the response as JSON, require an object, and verify required keys and primitive
value types. This is not full JSON Schema validation.

### Knowledge

**Prompt:** Seeded fact recall varies by tier and includes questions such as:

> What is the capital of France? Answer concisely.

**Evaluation:** The response must contain the expected fact. This small question set is not a
calibrated measure of broad knowledge or category-wide difficulty.

### Hallucination Resistance

**Prompt:** Ask about a seeded fictional country and request an explicit uncertainty response.

**Evaluation:** The response must contain `unknown`; higher tiers also require words indicating the
premise is fictional and cannot be verified. This is a controlled probe, not a general hallucination
score.

### Multilingual

**Prompt:** Translate seeded words or sentences into Spanish; higher tiers include clauses and
conditional grammar.

**Evaluation:** The response must contain the configured target translation phrase. Other valid
translations may be rejected; this evaluator is not a human language-quality judgment.

### Instruction Following

**Prompt:** Describe a seeded subject using an exact word count (3–6 words); Expert tasks also have
an additional required term.

**Evaluation:** Check exact word count and required terms. Semantic relevance is not evaluated.

### Summarization

**Prompt:** Summarize a seeded statement; harder tiers require preserving additional keywords.

**Evaluation:** Check required keywords only. This does not evaluate whether the summary is
semantically correct or complete.

### Creative Generation

**Prompt:** Write a slogan with an exact word count and required term(s).

**Evaluation:** Check exact word count and required term(s), not originality, quality, or relevance.

### Consistency

**Prompt:** Verify seeded addition, multiplication, or multi-operation expressions and answer yes/no.

**Evaluation:** The first answer token must be the expected `yes` or `no`.

### Context Retrieval

**Prompt:** Retrieve seeded markers among distractors; harder items request multiple values by
attribute.

**Evaluation:** Check that all requested marker strings occur in the answer. This does not validate
their order.

## Automatic scoring

Each passing primary test receives `10/10`; a completed failed test receives `0/10`. A test whose
required evaluator cannot run or whose model never returns an answer is **NOT RUN**, receives no
score, and is excluded from quality averages. Categories without completed applicable tests are
recorded as `null`/not tested rather than zero.

The result also records:

- answer text
- pass/fail status
- evaluator method
- retry attempts
- failure code, when available
- prompt-token estimate
- output-token count
- latency
- whether the test used a fresh context
- difficulty rationale, prompt, and evaluation criteria
- an explanation of the observable pass/fail check and each attempt's answer

The dashboard exposes these records with difficulty and outcome filters. It shows model outputs and
deterministic evaluator evidence, not hidden chain-of-thought. If a required evaluator is unavailable
or all model requests fail, the test is marked **NOT RUN** and excluded from quality scoring rather
than counted as an incorrect answer.

The fresh-context quality score is separate from context-behavior and performance measurements.
Demo results are marked `demo: true` and are not real model measurements.

## Context tests

The context suite is separate from the primary quality score. It targets:

- 4K
- 8K
- 16K
- 32K
- 64K
- 128K
- 256K when explicitly supported

Each context test:

1. Creates a seeded marker.
2. Adds deterministic distractor text.
3. Places the marker at the beginning, middle, or end.
4. Estimates prompt tokens from prompt length.
5. Asks the model to retrieve the marker.
6. Records latency, estimated prompt tokens, answer, and pass/fail status.

Two variants are recorded:

- **Fresh:** every context test is an independent request.
- **Accumulated:** prior user and assistant messages are included in the next request.

The result reports fresh score, accumulated score, and degradation between them. Context sizes
larger than the model's reported limit are skipped and listed in `context_skipped`. If the limit
is unknown, the implementation uses a conservative 128K ceiling. These are controlled context
experiments, not universal predictions of application behavior.

## Performance measurements

Before measured work, the engine sends three warm-up requests and discards their outputs. It then
records:

- tokens per second
- time to first token (`ttft_s`) for streaming-compatible real adapters
- generation time
- mean latency
- P50 latency
- P95 latency
- latency standard deviation
- concurrency samples at 1, 2, and 4 workers

For real Ollama and OpenAI-compatible adapters, streaming is now enabled and `ttft_s` is measured
from the first non-empty streamed content chunk. Demo mode remains deterministic and does not
claim real streaming performance. Docker code isolation is opt-in with
`BENCHMARK_USE_DOCKER=1`; otherwise the bounded fallback evaluator is used.

The current performance score is derived from measured throughput and capped at 100. These values
are controlled local measurements and should not be presented as universal real-world guarantees.

## Retries and failures

Each test can have an initial attempt plus two automatic retries. All attempts are preserved.
Known failure codes include:

- `TIMEOUT`
- `CONTEXT_OVERFLOW`
- `CONNECTION_ERROR`
- `SERVER_ERROR`
- `MODEL_ERROR`

Failures continue through the remaining tests and appear in the final JSON result.
When Docker is not installed, code tests report `EVALUATOR_UNAVAILABLE` instead of silently
implying that the model's code was incorrect.

## Evaluator limitations

The current objective evaluators are intentionally simple:

- numeric extraction with tolerance
- first-letter multiple-choice matching
- JSON parsing and required-key validation
- required-term matching
- word-count rubric checks
- assertion-based Python execution

There is no configured local evaluator-model rubric yet. Subjective categories therefore must not
be interpreted as human-quality judgments. The benchmark reports controlled measurements and
limitations rather than claiming universal model capability.
