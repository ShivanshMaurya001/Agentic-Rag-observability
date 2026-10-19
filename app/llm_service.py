"""Shared OpenAI text and structured-output client."""

import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel


MAX_TRANSPORT_RETRIES = 2

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_client: OpenAI | None = None


class LLMServiceError(RuntimeError):
    """Sanitized failure raised for unsuccessful OpenAI operations."""


class LLMRefusalError(LLMServiceError):
    """The model refused the request."""


class LLMIncompleteResponseError(LLMServiceError):
    """The model response did not complete successfully."""


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMServiceError("OPENAI_API_KEY is not configured.")

        _client = OpenAI(
            api_key=api_key,
            max_retries=MAX_TRANSPORT_RETRIES,
        )

    return _client


def _is_refusal(response: object) -> bool:
    for output_item in getattr(response, "output", ()) or ():
        for content_item in getattr(output_item, "content", ()) or ():
            if getattr(content_item, "type", None) == "refusal":
                return True
    return False


def _raise_request_error(operation: str, error: Exception) -> None:
    # Do not include provider error text: it may contain request or credential data.
    error_type = type(error).__name__
    raise LLMServiceError(
        f"OpenAI {operation} failed ({error_type})."
    ) from None


def generate_text(prompt: str, model: str) -> str:
    """Generate plain text, returning only completed non-empty output."""
    client = _get_client()
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
    except Exception as error:
        _raise_request_error("text generation", error)

    if getattr(response, "status", None) != "completed":
        raise LLMIncompleteResponseError(
            "OpenAI text generation returned an incomplete response."
        )

    output = getattr(response, "output_text", None)
    if not isinstance(output, str) or not output.strip():
        if _is_refusal(response):
            raise LLMRefusalError("OpenAI text generation was refused.")
        raise LLMServiceError("OpenAI text generation returned no text.")

    return output


def generate_structured(
    prompt: str,
    model: str,
    schema: type[SchemaT],
) -> SchemaT:
    """Generate and validate a Pydantic object with Responses Structured Outputs."""
    client = _get_client()
    try:
        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=schema,
        )
    except Exception as error:
        _raise_request_error("structured generation", error)

    if getattr(response, "status", None) != "completed":
        raise LLMIncompleteResponseError(
            "OpenAI structured generation returned an incomplete response."
        )

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        if _is_refusal(response):
            raise LLMRefusalError("OpenAI structured generation was refused.")
        raise LLMServiceError(
            "OpenAI structured generation returned no parsed value."
        )

    try:
        return schema.model_validate(parsed)
    except Exception:
        raise LLMServiceError(
            "OpenAI structured generation did not match the required schema."
        ) from None
