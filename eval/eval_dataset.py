import json
from pathlib import Path


EVAL_QUERIES_PATH = Path(__file__).with_name("eval_queries.json")


def load_eval_queries() -> list[dict]:
    """Load evaluation queries from the JSON dataset."""
    with EVAL_QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError("Eval dataset must be a JSON list.")

    return queries
