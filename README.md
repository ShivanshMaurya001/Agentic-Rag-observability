# Agentic RAG Observability & Self-Correction System

## KNOWN_ISSUES

- **Judge latency:** `judge_latency_ms` is recorded as `0` in SQLite on `main`; a callback-based attempt exists on branch `wip/callback-timing` and is unverified.
- **Cost fields:** Cost values are currently unmeasured placeholders (`0`). Do not present them as measured costs or display them in the dashboard unless actual cost measurement is implemented and verified.

## M3 Test Results

- Command: `python -m pytest -v --basetemp=.pytest_tmp`
- Result: **13 passed, 2 warnings**
- Warnings: FastAPI `on_event` deprecation warnings.

## M4 Evaluation Metrics

- **Answer Correctness:** Offline key-term coverage against `expected_answer`.
- **Scoring:** Normalize text to lowercase, remove punctuation and stopwords, then calculate the fraction of expected-answer key terms present in the generated answer.
- **Correctness threshold:** `0.6`, configured through `ANSWER_CORRECTNESS_THRESHOLD`.
- **Limitation:** This is an offline key-term coverage metric, not semantic scoring. It may miss valid paraphrases or overestimate correctness when key terms appear without the intended meaning.
- **Abstained answerable queries:** Counted as incorrect for answer correctness.
- **Adversarial queries:** Answer correctness is not scored for these queries.

Baseline and corrected evaluation report hallucination rate, False Answer Rate, Correct Abstention Rate, and general abstention rate, alongside the additional retrieval and retry metrics implemented by the evaluation runner.

**Metric interpretation:** Cost fields are placeholders until actual cost measurement is implemented and verified. Judge latency is also a known issue documented under `KNOWN_ISSUES`; do not treat its current SQLite value as measured latency.
