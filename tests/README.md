# Automated Test Suite

This file documents the repository's automated regression tests. These tests verify the benchmark
software itself; they are separate from the prompts used to evaluate AI models, which are documented
in [Benchmark Test Documentation](../Test_documentation.md).

## Run the suite

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The suite uses Python's standard library and the built-in demo model. It does not require an AI
server, network access, Docker, or third-party packages.

## What each test verifies

Tests are in `tests/test_benchmark.py` and `tests/test_calibration.py`.

### Calibration analysis

The calibration-tool tests validate CSV schema and duplicate rejection, cohort/version separation,
respondent-cluster intervals, missing outcomes, reviewer agreement, and repeatability calculations
using synthetic fixtures. These fixtures validate the analyzer only; they are not evidence about
model or human difficulty. Task-generation tests also verify seeded item fingerprints distinguish
exact generated prompt/check variants. See [`CALIBRATION_STUDY.md`](../CALIBRATION_STUDY.md) for
the data-collection protocol and analysis limits.

### Generation settings

Verifies generation settings are normalized to supported ranges and mapped into requests sent to
Ollama and OpenAI-compatible adapters. The demo export test also verifies the selected setting is
retained in result configuration. It does not replace integration testing against live servers.
Additional coverage verifies repeated-run score metadata and that runtime details are included only
when reported by the local server.

Hardware inventory/recommendation tests mock OS/vendor output, missing utilities, and memory tiers;
they do not certify model fit or GPU offload on real Windows/macOS/Linux machines. Adapter coverage
checks backend timing/usage counters and estimate labels, live callbacks, truncated output, and
single-chunk responses that cannot establish a decode rate. HTTP tests verify the hardware API,
advanced controls, and sandbox headers for downloadable voxel HTML reports. Export tests cover
per-model folders and legacy flat results. Live hardware/server compatibility still needs validation
on the target workstation.

Streaming adapter tests cover first-content TTFT measurement, timeout error mapping, malformed event
handling, size-bounded discovery responses, accumulated history in both Ollama and OpenAI-compatible
requests, and a loopback streaming round-trip. These tests exercise local protocols only and do not
establish compatibility with live model servers. Additional adapter tests cover fragmented UTF-8/SSE
events, finish markers, empty and incomplete streams, mid-stream disconnects, HTTP status errors, and
the rule that adapter failures are recorded as unscored `NOT RUN` outcomes. Result comparisons warn
about version/configuration mismatches and missing measurements; export checks assert demo provenance,
PDF literal escaping, CSV formula neutralization, and SVG text escaping. Evaluator checks cover strict
JSON constants, keyword-only rubric answers, JSON value types, exact-response contradictions, and
word-count rubrics. Benchmark tests verify category-weighted scoring, missing-repeat accounting,
performance calculations use dedicated generation-time
samples, distinguish item bootstrap resampling from repeated model runs, and checkpoint/resume after
cancellation during context tests. Difficulty-generation coverage checks that all four tiers appear
in QUICK and that every primary category spans all tiers in FULL. It also checks varied
reasoning/summarization items, two coding task families, and generated long-context tasks.
Knowledge generation tests check seeded domain-family variety; context generation checks estimated
target sizes and rotating marker positions. Performance checks verify reported concurrency outcomes.
Test-result coverage verifies
prompt/evaluator explanations, answer traces, and that unavailable evaluators produce unscored
NOT RUN outcomes.

Managed llama.cpp launcher tests mock Windows process creation and verify locating executables in an
already-installed WinGet package; they do not run a real `llama-server.exe`, exercise Windows Job
Objects, or validate actual GPU/context capacity. Live Windows testing is still required for those
behaviors.

HTTP tests cover all implemented POST routes' CSRF/origin rejection, explicit model selection,
request-target and body limits, and traversal/internal-file download rejection. Checkpoint tests cover
HMAC tampering, malformed/non-object and incompatible state, generated-suite validation, and
cancellation during warm-up, primary, context, and performance phases. Docker command tests verify
resource flags, digest pinning, and forced cleanup attempts on timeout/failure; they do not exercise a
real Docker daemon or prove the container boundary.

### `test_domain_evaluators`

Verifies representative evaluator behavior:

- numeric matching accepts the expected number
- required-key JSON evaluation accepts a valid object
- Python code evaluation accepts a correct `add` implementation
- numeric matching rejects an incorrect number

This is representative coverage, not exhaustive coverage of every evaluator input or security
boundary.

### `test_deterministic_test_generation`

Verifies that:

- the same profile and seed generate identical tests
- the `FULL` profile generates 252 primary tests
- generated `FULL` prompts are unique
- the default context suite includes 4K through 128K targets
- a 32K model limit excludes larger context targets

It does not assess whether the generated prompts are sufficiently diverse or representative of
real-world work.

### `test_demo_run_exports_json_csv_pdf`

Runs the `QUICK` profile end to end with the deterministic demo model and verifies that:

- JSON, CSV, and PDF files are created
- the JSON result reports a perfect demo quality score and is marked as demo data
- primary tests report fresh-context execution
- fresh and accumulated context sections are present
- result comparison identifies the supplied result
- CSV headers include failure diagnostics; the PDF includes the not-run summary; the SVG has a root element

These are representative export checks, not complete schema, rendering, or accessibility validation.

## What is not currently covered

The automated suite does not currently verify:

- live Ollama, LM Studio, or OpenAI-compatible servers
- live streaming behavior across real servers or every adapter failure mode (the loopback fixture
  checks only the local adapter protocol); run-specific llama.cpp and Ollama integration behavior
  still needs validation against supported real server versions
- all HTTP status codes and malformed request combinations (route protections and key size/traversal
  boundaries are covered; exhaustive protocol fuzzing is not)
- real browser rendering and user interactions (a static regression check guards against `innerHTML`;
  it does not replace browser execution)
- cancellation responsiveness during every model request and concurrent lifecycle behavior beyond the
  tested start/resume/cancel serialization paths
- full retry coverage and all evaluator edge cases
- accuracy of real-server performance metrics or concurrency behavior beyond the reported sample
- output report contents beyond the newly asserted representative PDF/CSV/SVG fields
- live Docker isolation or adversarial generated code
- abnormal Windows process termination and cross-platform behavior
- saved-history interactions, keyboard chart navigation, or live model-server compatibility

Accordingly, a passing suite shows that the covered core paths still behave as expected. It does not
prove model quality, benchmark validity, security against hostile code, real-server compatibility,
or production readiness.
