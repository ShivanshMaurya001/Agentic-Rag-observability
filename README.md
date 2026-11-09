# Agentic RAG Observability & Self-Correction System
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM_API-412991?logo=openai&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

An observable Retrieval-Augmented Generation (RAG) system with structured LLM judging, corrective retry workflows, abstention handling, baseline comparison, and an evaluation dashboard.

## 🚀 Live Demo

**Dashboard:** https://agentic-rag-observability.onrender.com

## Project Overview

This project explores how a RAG pipeline can evaluate its own generated answers, identify retrieval or generation failures, and attempt corrective actions through a controlled retry workflow.

The system includes:

- Dense retrieval using ChromaDB.
- Structured answer evaluation using a judge.
- Failure classification: `retrieval`, `generation`, or `none`.
- Corrective query rewriting and answer regeneration.
- Retry handling using LangGraph.
- Abstention when the retry process cannot obtain an accepted answer.
- Execution trace logging and baseline comparison.
- A read-only Streamlit observability dashboard.
- Offline evaluation of answer correctness, retrieval performance, hallucination proxy, and abstention behavior.

**Important:** Self-correction is not assumed to improve factual accuracy. The evaluation compares the baseline and corrected pipelines to identify actual improvements, unchanged metrics, and limitations.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ShivanshMaurya001/Agentic-Rag-observability
cd Agentic-Rag-observability
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 3. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add Documents and Run Ingestion

Place PDF documents inside `test_data/` and run:

```bash
python ingest_pdfs.py
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

### 7. Run Evaluation

**Note:** Evaluation makes real OpenAI API calls and may incur API costs.

```bash
python scripts/run_baseline_eval.py
```

To resume an interrupted evaluation:

```bash
python scripts/run_baseline_eval.py --resume
```

### 8. Launch the Dashboard

```bash
python -m streamlit run dashboard/streamlit_app.py
```

## Dashboard Preview

### Dashboard Overview
![Dashboard Overview](screenshots/dashboard.png)

### Execution and Retry Analysis
![Execution and Retry Analysis](screenshots/retry.png)

### Decision Trace and Failure Cases
![Decision Trace and Failure Cases](screenshots/retry11.png)

## Key Findings

- **No aggregate accuracy improvement:** Answer correctness remained 20/47 (42.55%) in both baseline and corrected evaluation.
- **Correct refusals triggered retries:** 12 adversarial queries triggered retries because insufficient retrieval context was classified as a retrieval failure.
- **Judge approval does not guarantee correctness:** `answered_after_retry` indicates judge acceptance, not necessarily factual correctness.
- **Abstention detection bug fixed:** The initial detector missed natural-language refusals. After fixing it, the corrected false-answer rate changed from 61.5% to 0%.
---

## System Architecture

```mermaid
flowchart TD
    A[User Query] --> B[Dense Retrieval]
    B --> C[Answer Generation]
    C --> D[Structured Judge]

    D --> E{route_after_attempt}

    E -->|failure_type = none| F[Accept Answer]

    E -->|Retrieval failure and retry allowed| G[Rewrite Query]
    G --> B

    E -->|Generation failure and retry allowed| H[Regenerate Answer]
    H --> D

    E -->|Retry limit reached| I[Abstain]

    F --> J[Trace Logging]
    I --> J
    J --> L[Observability Dashboard]
```

The judge evaluates the answer against the retrieved context and produces a structured verdict.

| Judge field | Purpose |
|---|---|
| `grounded` | Indicates whether the answer is grounded in the retrieved context. |
| `retrieval_quality` | Evaluates the quality of retrieved context. |
| `answer_quality` | Evaluates the generated answer. |
| `failure_type` | Identifies retrieval, generation, or no failure. |
| `reason` | Explains the judge's decision. |
| `is_abstention` | Indicates whether the generated response is an abstention. |

The retry workflow distinguishes between two failure paths:

- **Retrieval failure:** Rewrite the query, retrieve again, generate a new answer, and judge it.
- **Generation failure:** Regenerate the answer using the same retrieved context and judge it again.

The system allows a maximum of two retry attempts.

---

## Test Results

| Test detail | Result |
|---|---:|
| Test command | `python -m pytest -v --basetemp=.pytest_tmp` |
| Tests passed | 74 |
| Warnings | 2 |
| Warning type | FastAPI `on_event` deprecation warnings |

---

## Evaluation Methodology

The evaluation compares two execution modes:

1. **Baseline:** The original RAG pipeline without corrective retries.
2. **Corrected:** The RAG pipeline with structured judging and corrective retry handling.

### Evaluation Dataset

| Dataset property | Value |
|---|---:|
| Total queries | 60 |
| Adversarial queries | 13 |
| Non-adversarial queries | 47 |
| Baseline evaluation rows | 60 |
| Corrected evaluation rows | 60 |

**Corpus:** 15 PDFs on LLM and RAG topics (`test_data/`).

The adversarial queries test whether the system can avoid providing unsupported answers when the requested information is unavailable or insufficiently supported by the documents.

### Answer Correctness Metric

Answer correctness is evaluated using offline key-term coverage against `expected_answer`.

The evaluation process:

1. Normalize text to lowercase.
2. Remove punctuation and stopwords.
3. Identify expected-answer key terms.
4. Calculate the fraction of expected-answer key terms present in the generated answer.

**Correctness threshold:** `0.6`, configured through `ANSWER_CORRECTNESS_THRESHOLD`.

This is a key-term coverage metric, not semantic answer evaluation. It can miss valid paraphrases or overestimate correctness when expected terms appear without the intended meaning.

Additional rules:

- Abstained answerable queries are counted as incorrect.
- Adversarial queries are not scored for answer correctness.

### Answered-but-wrong rate (proxy)

The approved offline answered-but-wrong rate proxy uses the following logic:

- Every answered adversarial query counts as an answered-but-wrong response.
- An answered non-adversarial query counts as an answered-but-wrong response when its saved `answer_correct` value is false.

This is a ground-truth evaluation proxy, not a separate semantic answered-but-wrong detector or a direct measure of all unsupported claims.

### Other Evaluation Metrics

| Metric | What it measures |
|---|---|
| False Answer Rate | Unsupported answers provided for adversarial queries. |
| Correct Abstention Rate | Correct abstentions on adversarial queries. |
| General Abstention Rate | Total queries classified as abstentions. |
| Recall@K | Retrieval recall against the relevant chunks. |
| Hit Rate@K | Whether at least one relevant chunk was retrieved. |
| Answer Correctness | Expected-answer key-term coverage. |
| Answered-but-wrong rate (proxy) | Offline ground-truth/key-term coverage proxy. |
| Judge-approved after Retry | Query-level judge acceptance after corrective retries. |

---

## Baseline vs Corrected Results

The following table contains the saved evaluation results.

### Key Metrics Comparison

| Metric | Baseline | Corrected |
|---|---:|---:|
| False Answer Rate | 0/13 = **0%** | 0/13 = **0%** |
| Correct Abstention Rate | 13/13 = **100%** | 13/13 = **100%** |
| Answered-but-wrong rate (proxy) | 26/46 = **56.52%** | 27/47 = **57.45%** |
| Answer Correctness | 20/47 = **42.55%** | 20/47 = **42.55%** |
| General Abstention Rate | 14/60 = **23.33%** | 13/60 = **21.67%** |
| Recall@K | **72.34%** | **72.34%** |
| Hit Rate@K | **74.47%** | **74.47%** |
| Judge-approved after Retry | N/A | 12/13 = **92.31%** |

### Visual Comparison

The following charts visualize the baseline and corrected evaluation metrics.

```mermaid
xychart-beta
    title "Baseline vs Corrected Metrics (%)"
    x-axis ["Correctness", "Recall@K", "Hit Rate@K", "Answered-but-wrong"]
    y-axis "Percentage" 0 --> 100
    bar [42.55, 72.34, 74.47, 56.52]
    bar [42.55, 72.34, 74.47, 57.45]
```

**Series order:** First bar = Baseline; second bar = Corrected.

```mermaid
xychart-beta
    title "General Abstention Rate (%)"
    x-axis ["Baseline", "Corrected"]
    y-axis "Percentage" 0 --> 30
    bar [23.33, 21.67]
```

### Metric Interpretation

The results show:

- Answer Correctness remained unchanged at 42.55%.
- Recall@K remained unchanged at 72.34%.
- Hit Rate@K remained unchanged at 74.47%.
- False Answer Rate remained at 0% in both modes.
- Correct Abstention Rate remained at 100% in both modes.
- The corrected answered-but-wrong rate proxy increased slightly from 56.52% to 57.45%.
- General Abstention Rate decreased from 23.33% to 21.67%.

**Conclusion:** The current evaluation does not demonstrate an improvement in aggregate answer correctness or retrieval performance from the corrective retry pipeline.

This does not prove that the retry workflow never helps. Its behavior and limitations need to be interpreted alongside the attempt-level results.

---

## Corrective Retry Analysis

### Retry Statistics

In the saved evaluation (60 queries), **13 corrected-mode queries triggered a retry: 12 adversarial and 1 answerable.**

Some queries required more than one retry round.

| Retry statistic | Result |
|---|---:|
| Total evaluation queries | 60 |
| Queries that triggered retries | 13 |
| Adversarial queries retried | 12 |
| Answerable queries retried | 1 |
| Total retry rounds | 16 |
| Queries with two retry rounds | 3 |

### Retry Type Breakdown

The attempt-level saved evaluation results show:

| Retry type | Retry rounds | LLM calls per round | Additional LLM calls |
|---|---:|---:|---:|
| Retrieval retry | 15 | 3 | 45 |
| Generation retry | 1 | 2 | 2 |
| **Total** | **16** | | **47** |

The graph implementation explains the call counts:

**Retrieval retry:**

1. Query rewriting using the rewrite model.
2. Answer generation using the generator model.
3. Structured judge evaluation.

**Generation retry:**

1. Answer regeneration using the generator model.
2. Structured judge evaluation.

These are graph-level LLM-call invocations based on the implemented function flow. They are not independently verified provider billing records.

### Retry Distribution

```mermaid
xychart-beta
    title "Retry Rounds by Failure Type"
    x-axis ["Retrieval", "Generation"]
    y-axis "Retry Rounds" 0 --> 16
    bar [15, 1]
```

### Answerable Query: Possible Recovery Example

One answerable query triggered a generation retry:

**Query:**

"What is the purpose of a reward model in RLHF?"

**Attempt 0:**

The initial generated response was:

"The answer was not found in the document."

The judge classified this as a generation failure.

**Attempt 1:**

The retry produced a substantive answer explaining that a reward model learns from human preference data and guides fine-tuning toward outputs considered helpful, honest, and harmless.

| Evaluation detail | Result |
|---|---:|
| Initial attempt | Abstention |
| Retry type | Generation |
| Retrieved context | Same chunks |
| Final key-term score | 0.5833 |
| Correctness threshold | 0.6 |
| Final `answer_correct` | False |

This is a possible recovery example: the retry produced a substantive answer that appears consistent with the document's explanation, but the offline key-term proxy did not classify it as correct because its score remained just below the threshold.

It should not be presented as a confirmed improvement in answer correctness.

### Why Aggregate Correctness Did Not Change

Only 1 of the 47 answerable queries triggered a retry.

The other 46 answerable queries did not require a retry according to the judge's initial verdict.

The single retried answerable query moved from an initial abstention to a substantive answer, but its final key-term score remained below the correctness threshold.

Therefore, the aggregate Answer Correctness remained 20/47 (42.55%).

The unchanged Recall@K and Hit Rate@K also do not establish that the retry loop failed. The evaluation does not show a retrieval improvement, and the single retried answerable query used the same retrieved chunk IDs across both attempts.

## Limitations

- **Small evaluation dataset:** 60 queries, including only 13 adversarial queries. Results are from a single evaluation run, with no confidence intervals.
- **LLM non-determinism:** Model outputs may vary between runs.
- **Same model for all LLM roles:** Generator, query rewriter and judge use the same model, which may introduce judge self-evaluation bias.
- **Proxy-based correctness:** Answer correctness uses key-term coverage rather than semantic evaluation. The judge accepted 46 of 47 first attempts on answerable queries, while the offline metric classified only 20 of 47 as correct.
- **Shared baseline attempts:** Baseline results reuse the corrected pipeline's first attempt, so both modes share the same initial retrieval and generation.
- **Outcome classification limitation:** `answered_after_retry` can represent a judge-approved refusal. Offline evaluation classifies abstentions separately.
- **Dashboard comparison limitation:** The dashboard reads recorded SQLite traces across runs. Its aggregate baseline and corrected rates are not a matched comparison. API cost is not recorded.
- **Unresolved design consideration:** Retry routing does not currently check `is_abstention`. This behavior remains documented under PRD Open Decision C.

## Design Decisions

- **Dense retrieval:** ChromaDB is used as the v1 vector retrieval backend.
- **Retrieval interface:** All retrieval-time reads use the common `retrieve()` interface.
- **Corrective workflow:** LangGraph manages retrieval failures, generation failures, retries, and abstention.
- **Observability:** SQLite stores execution traces, which are displayed through a read-only Streamlit dashboard.
- **LLM providers:** Gemini handles embeddings; OpenAI is used for generation, query rewriting, regeneration, and judging.
- **Scope control:** Hybrid search is not implemented in v1. Open Decisions A, B, and C remain unresolved.

## Future Work

- Resolve Open Decision C before changing judge-only routing.
- Investigate disagreement between judge acceptance and offline answer correctness.
- Evaluate answerability and abstention behavior on a larger dataset.
- Implement and verify actual API cost measurement before reporting costs.
- Re-evaluate the corrective workflow after any approved changes.
- **Optional:** Validate the custom judge using RAGAS.