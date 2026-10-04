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

## Test

```bash
python -m unittest discover -s tests -v
```
