from __future__ import annotations

from typing import Any
from cais.config import get_llm_client
from langchain_core.language_models import BaseChatModel




class LLMClient:
    """Thin adapter that wraps a BaseChatModel"""
    def __init__(self, config: Any = None, use_cache: bool = False) -> None:
        if isinstance(config, dict):
            self._llm = get_llm_client(
                provider=config.get("provider"),
                model_name=config.get("model"),
            )
        else:
            self._llm = get_llm_client()

    def invoke(self, prompt: str) -> str:
        """Delegate to the underlying LangChain model and return string content."""
        response = self._llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)