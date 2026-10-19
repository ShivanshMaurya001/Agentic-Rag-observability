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

## M4 Baseline Evaluation Reuse

- **Baseline reuse:** The evaluation runner can reuse saved initial retrieval and generation results when they are available, avoiding unnecessary repeated generator API calls during corrected evaluation.
- **Resume behavior:** With `--resume`, previously saved evaluation pairs are reused, and missing pairs are evaluated.
- **Result persistence:** Evaluation results are written to `eval/baseline_results.json`. Keep this file and its backups out of implementation checkpoint commits unless explicitly required.
- **Validation:** The focused baseline evaluation tests passed (**14 passed**). The full test suite passed (**30 passed, 2 warnings**).
- **Limitation:** Reusing saved results depends on the required initial retrieval and answer data being present. These results should not be interpreted as proof that every historical result can be reused.
