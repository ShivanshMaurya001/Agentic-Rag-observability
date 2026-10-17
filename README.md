# Agentic RAG Observability & Self-Correction System

## KNOWN_ISSUES

- **Judge latency:** `judge_latency_ms` is recorded as `0` in SQLite on `main`; a callback-based attempt exists on branch `wip/callback-timing` and is unverified.
- **Cost fields:** Cost values are currently unmeasured placeholders (`0`). Do not present them as measured costs or display them in the dashboard unless actual cost measurement is implemented and verified.

## M3 Test Results

- Command: `python -m pytest -v --basetemp=.pytest_tmp`
- Result: **13 passed, 2 warnings**
- Warnings: FastAPI `on_event` deprecation warnings.
