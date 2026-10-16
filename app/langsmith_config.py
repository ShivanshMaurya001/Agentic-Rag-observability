import os

from dotenv import load_dotenv


load_dotenv()


def configure_langsmith() -> None:
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError(
            "LANGSMITH_API_KEY is missing. Add it to your .env file."
        )

    os.environ.setdefault(
        "LANGSMITH_PROJECT",
        "Agentic-RAG-Observability",
    )