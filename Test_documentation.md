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
prompts are avoided. The difficulty labels cycle through **Easy**, **Medium**, **Hard**, and
**Expert**.

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

**Prompt:** A parameterized multiplication problem such as:

> Calculate `a × b`. Return only the number.

`a` and `b` are seeded random integers from 2 through 99.

**Evaluation:** Extract the first numeric value from the response and compare it with the expected
product. The current multiplication tests use zero numeric tolerance.

### Reasoning

**Prompt:** A simple categorical deduction such as:

> If all zibs are blue and item `a` is a zib, what color is item `a`?

**Evaluation:** The response must contain the required answer term, currently `blue`.

### Coding

**Prompt:** Ask the model to write Python defining `add(a, b)` and return code only.

**Evaluation:** The returned source is combined with assertions such as:

```python
assert add(2, 3) == 5
assert add(-1, 1) == 0
```

The code evaluator returns a pass only when the assertions complete successfully. Code execution
is bounded by source length and timeout. If Docker is available and enabled, execution prefers a
network-disabled, read-only container with CPU, memory, process, and timeout limits. This is a
defense-in-depth measure; generated code should not be treated as safe for hostile workloads
unless Docker isolation is available and verified.

### Structured Output

**Prompt:** Ask for JSON with the keys `answer` and `valid`.

**Evaluation:** Parse the response as JSON, require a JSON object, and verify that all required
keys exist. Values are not currently validated against a full JSON Schema.

### Knowledge

**Prompt:** Ask:

> What is the capital of France? Answer concisely.

**Evaluation:** The response must contain `Paris`.

### Hallucination Resistance

**Prompt:** Ask about the fictional country Zorbland and instruct the model to say it is unknown.

**Evaluation:** The response must contain `unknown`. This is a controlled refusal/factuality probe,
not a general hallucination score.

### Multilingual

**Prompt:** Translate `good morning` into Spanish.

**Evaluation:** The response must contain both `buenos` and `días`.

### Instruction Following

**Prompt:** Ask for exactly three words describing rain.

**Evaluation:** The current rubric checks only that the response has at least the configured
minimum number of words. It does not yet enforce exactly three words or semantic relevance.

### Summarization

**Prompt:** Summarize a statement about local systems and mention local privacy.

**Evaluation:** The response must contain both `local` and `privacy`. This checks required facts,
not semantic summary quality.

### Creative Generation

**Prompt:** Ask for a four-word slogan about careful testing.

**Evaluation:** The current rubric checks only a minimum word count. It does not objectively score
originality, quality, or relevance.

### Consistency

**Prompt:** Ask whether `2 + 2` equals `4`.

**Evaluation:** The response must contain `yes`.

### Context Retrieval

**Prompt:** Include a seeded marker such as `BLUE-1234` and ask the model to return it.

**Evaluation:** The response must contain the exact marker.

## Automatic scoring

Each passing primary test receives `10/10`; each failed, empty, invalid, or unavailable response
receives `0/10`. Category scores are normalized to 0–100. Categories without applicable tests
are recorded as `null`/not tested rather than zero.

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
- time to first response (`ttft_s`; true streaming TTFT is not yet available)
- generation time
- mean latency
- P50 latency
- P95 latency
- latency standard deviation
- concurrency samples at 1, 2, and 4 workers

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
