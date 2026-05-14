"""Shared LLM factory for all agents.

Uses OpenAI's API with gpt-4o-mini model.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenAI."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )