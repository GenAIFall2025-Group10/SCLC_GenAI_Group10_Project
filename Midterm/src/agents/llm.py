"""
ll.py — Unified LLM loader for OpenAI-compatible endpoints
"""

from langchain_openai import ChatOpenAI
import os


def get_llm(
    model_name: str = None,
    temperature: float = 0.3,
    base_url: str = None,
    api_key: str = None
):
    """
    Returns a configured ChatOpenAI model compatible with OpenAI or custom APIs.

    Args:
        model_name: The model name (e.g. "gpt-4o", "llama3", etc.)
        temperature: Creativity level.
        base_url: Custom endpoint (e.g. http://localhost:8000/v1).
        api_key: API key (defaults to OPENAI_API_KEY env var).

    Usage:
        llm = get_llm(model_name="gpt-4o")
        llm.invoke("Hello!")
    """

    # Use environment vars if not provided
    model_name = model_name or os.getenv("OPENAI_MODEL", "llama-4-scout")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://llm.jetstream-cloud.org/api/")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        base_url=base_url,
    )

    return llm
