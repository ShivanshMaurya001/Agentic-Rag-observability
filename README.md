# Agentic RAG Observability & Self-Correction System

## KNOWN_ISSUES

- **Judge latency:** `judge_latency_ms` is recorded as `0` in SQLite on `main`; a callback-based attempt exists on branch `wip/callback-timing` and is unverified.
 **Cost fields:** Cost values are currently unmeasured placeholders (`0`). Do not present them as measured costs or display them in the dashboard unless actual cost measurement is implemented and verified.

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

**Hallucination Rate (owner decision):** An answered query is one for which `is_abstained()` returns false. Every answered adversarial query counts as a hallucination; an answered non-adversarial query counts only when its saved `answer_correct` value is false. The rate is hallucinated answered queries divided by all answered queries. It is `null` when there are no answered queries or when an answered non-adversarial row lacks a boolean `answer_correct` value; the status identifies the zero-answer or incomplete-data case. This is a ground-truth proxy based on answer correctness, not a separate judge-verdict metric.

`answer_correct` uses offline expected-answer key-term coverage with a `0.6` threshold. This proxy inherits the limitation above: it is not semantic scoring and may miss valid paraphrases or overestimate correctness when terms appear without the intended meaning. Baseline and corrected evaluation report this rate alongside False Answer Rate, Correct Abstention Rate, general abstention rate, and the additional retrieval and retry metrics implemented by the evaluation runner.

**Metric interpretation:** Cost fields are placeholders until actual cost measurement is implemented and verified. Judge latency is also a known issue documented under `KNOWN_ISSUES`; do not treat its current SQLite value as measured latency.

## M4 Baseline Evaluation Reuse

- **Baseline reuse:** The evaluation runner can reuse saved initial retrieval and generation results when they are available, avoiding unnecessary repeated generator API calls during corrected evaluation.
- **Resume behavior:** With `--resume`, previously saved evaluation pairs are reused, and missing pairs are evaluated.
- **Result persistence:** Evaluation results are written to `eval/baseline_results.json`. Keep this file and its backups out of implementation checkpoint commits unless explicitly required.
- **Validation:** The focused baseline evaluation tests passed (**14 passed**). The full test suite passed (**30 passed, 2 warnings**).
- **Limitation:** Reusing saved results depends on the required initial retrieval and answer data being present. These results should not be interpreted as proof that every historical result can be reused.

## Decisions

- **Approved provider exception:** Answer generation, query rewriting, regeneration, and structured judging use the OpenAI API. Gemini remains responsible for embeddings. The locked PRD and Implementation Blueprint are unchanged.

## M4 Evaluation Findings

The M4 evaluation compares the baseline RAG pipeline with the corrective retry pipeline across 60 queries, including 13 adversarial queries.

### Key Results

| Metric | Baseline | Corrected |
|---|---: |---:|
| False Answer Rate | 0/13 = 0% | 0/13 = 0% |
| Correct Abstention Rate | 13/13 = 100% | 13/13 = 100% |
| Hallucination Rate | 26/46 = 56.52% | 27/47 = 57.45% |
| Answer Correctness | 20/47 = 42.55% | 20/47 = 42.55% |
| General Abstention Rate | 14/60 = 23.33% | 13/60 = 21.67% |
| Recall@k | 72.34% | 72.34% |
| Hit Rate@k | 74.47% | 74.47% |
| Judge-approved after Retry | n/a (no baseline retries) | 12/13 = 92.31% |

### Key Observations

- The original abstention detector missed eight reviewed natural-language refusals in corrected-mode evaluation rows. The deterministic detector correction recognizes those eight cases. This changes their classification from answered to abstained; it does not change the generated answers.
- The corrected False Answer Rate is now 0%, but this is a classification correction, not evidence of improved generation quality. Both modes have identical measured Answer Correctness (20/47 = 42.55%); self-correction did not improve this measure.
- Hallucination Rate is an offline ground-truth/key-term coverage proxy, not an objective claim that 57% of answers are wrong. `answer_correct` uses expected-answer key-term coverage with a 0.6 threshold. It is not semantic scoring and may miss valid paraphrases or overestimate correctness when terms appear without the intended meaning.
- The adversarial sample is small (13 queries), so these results should be interpreted cautiously.
- A separate SQLite trace investigation found that 12 of 13 corrected adversarial runs had first attempts marked `retrieval`, even though the judge reasons acknowledged that the generated refusal correctly stated the information was missing. All 12 were followed by another refusal; 11 ended as `answered_after_retry` and one as `abstained_judge_rejected`. This suggests the retry loop can spend additional calls retrying an already-correct refusal. These SQLite runs are not proven to be the same population as the 13 offline evaluation rows, and this inspection does not establish exact cost savings or row-level matching.
- The 92.31% **Judge-approved after Retry** metric represents answers approved by the judge after retry. It does not establish factual correctness.
- Recall@k and Hit Rate@k are identical across both modes. Therefore, this evaluation does not demonstrate retrieval improvement.

### Evaluation Interpretation

The corrective pipeline demonstrates the retry workflow, but the current evaluation does not establish that it improves answer correctness or retrieval. The abstention correction changes classification of the reviewed refusals; it does not establish improved generation quality or reduced hallucinations. The results expose limitations in the current judge and corrective process that should remain visible in the baseline comparison.
