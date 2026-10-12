def build_rewrite_prompt(original_query: str, judge_reason: str) -> str:
    return f"""
You are improving a search query for a retrieval-augmented generation system.

Original query:
{original_query}

Judge reason:
{judge_reason}

Rewrite the original query into a clearer and more effective search query
that addresses the issue identified by the judge.

Return only the improved query.
""".strip()

