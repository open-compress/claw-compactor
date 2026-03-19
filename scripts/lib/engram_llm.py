"""
engram_llm.py — LLM client for Engram Observer/Reflector calls.

Supports Anthropic Messages API and OpenAI-compatible chat completions.
Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import logging
from typing import Optional

from claw_compactor.engram_http import http_post

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class EngramLLMClient:
    """
    LLM client that routes calls to Anthropic or OpenAI-compatible endpoints.

    Args:
        model:             LLM model identifier.
        max_tokens:        Max tokens the LLM may produce per call.
        anthropic_api_key: Anthropic API key (empty string = disabled).
        openai_api_key:    OpenAI API key (empty string = disabled).
        openai_base_url:   OpenAI-compatible base URL.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        openai_base_url: str = "https://api.openai.com",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url

    def call(self, system: str, user: str) -> str:
        """
        Call LLM API. Prefers Anthropic if key available, else OpenAI-compatible.

        Args:
            system: System prompt.
            user:   User message content.

        Returns:
            Assistant response text.

        Raises:
            RuntimeError: If no API key is configured.
        """
        if self.anthropic_api_key:
            return self._call_anthropic(system, user)
        if self.openai_api_key:
            return self._call_openai_compatible(system, user)
        raise RuntimeError(
            "EngramEngine: no API key configured. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
        )

    def _call_anthropic(self, system: str, user: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = http_post(url, headers, body)
        content = data.get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        raise ValueError(f"Engram: no text content in Anthropic response: {data}")

    def _call_openai_compatible(self, system: str, user: str) -> str:
        base = self.openai_base_url.rstrip("/")
        url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = http_post(url, headers, body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Engram: unexpected OpenAI response structure: {data}"
            ) from exc
