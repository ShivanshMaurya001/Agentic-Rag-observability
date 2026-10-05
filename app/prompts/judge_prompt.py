def build_judge_prompt(query: str, context: str, answer: str) -> str:
    return f"""
You are a strict evaluator for a retrieval-augmented generation system.

Evaluate the answer using ONLY the provided context.

Query:
{query}

Context:
{context}

Answer:
{answer}

Evaluate:

- grounded: whether the answer is supported by the provided context.
- retrieval_quality: whether the retrieved context is relevant and sufficient to answer the query. Score from 0.0 to 1.0.
- answer_quality: whether the answer is fully supported by the context without unsupported or contradictory claims. Score from 0.0 to 1.0.
- failure_type:
  - "retrieval" if the retrieved context is insufficient or irrelevant.
  - "generation" if the context is sufficient but the answer is unsupported, incorrect, or poorly generated.
  - "none" if the answer is adequately supported and there is no failure.
- reason: briefly explain the basis for the verdict.

Return only the structured verdict.
""".strip()