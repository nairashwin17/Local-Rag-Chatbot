"""Small provider adapters for embeddings and chat completions."""

from __future__ import annotations

from config import Settings


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected provider has no configured API key."""


def _require_key(settings: Settings) -> str:
    if not settings.active_api_key:
        variable = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "GEMINI_API_KEY"
        raise ProviderConfigurationError(f"Set {variable} in .env before using {settings.llm_provider}.")
    return settings.active_api_key


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed text with the official SDK for the selected provider."""
    key = _require_key(settings)
    if settings.llm_provider == "openai":
        from openai import OpenAI

        response = OpenAI(api_key=key).embeddings.create(model=settings.active_embedding_model, input=texts)
        return [item.embedding for item in response.data]
    from google import genai

    # Keep a strong reference until the blocking SDK request completes. Creating the
    # client inline can trigger Client.__del__ and close its HTTP client too early.
    client = genai.Client(api_key=key)
    try:
        response = client.models.embed_content(model=settings.active_embedding_model, contents=texts)
        return [item.values for item in response.embeddings]
    finally:
        client.close()


def generate_text(prompt: str, settings: Settings) -> str:
    """Generate a response using the selected provider's official SDK."""
    key = _require_key(settings)
    if settings.llm_provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=key).responses.create(model=settings.active_model, input=prompt).output_text
    from google import genai

    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(model=settings.active_model, contents=prompt)
        return response.text or "I could not generate an answer from the retrieved evidence."
    finally:
        client.close()
