"""LLMSummarizer — API-based conversation summarization.

Inspired by Claude Code's AutoCompact which calls the LLM to generate a 20K
token summary of compacted conversation history. This module provides an
optional LLM-powered summarizer that produces higher-quality summaries than
the deterministic extractor in conversation_summarizer.py.

Usage::

    from claw_compactor.fusion.llm_summarizer import LLMSummarizer

    summarizer = LLMSummarizer(api_key="sk-...", model="claude-sonnet-4-20250514")
    messages, stats = summarizer.summarize(messages, token_budget=200_000)

Falls back to deterministic summarization if:
  - No API key is provided
  - The LLM call fails
  - The LLM response is empty or too large

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional, Protocol

from claw_compactor.tokens import estimate_tokens
from claw_compactor.fusion.conversation_summarizer import (
    summarize_conversation as _deterministic_summarize,
    _split_messages,
    DEFAULT_PRESERVE_RECENT_TURNS,
    DEFAULT_TRIGGER_PCT,
    MAX_SUMMARY_TOKENS,
)

logger = logging.getLogger(__name__)

# Default model for summarization.
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# System prompt for the summarizer LLM call.
_SUMMARIZER_SYSTEM_PROMPT = """\
You are a conversation summarizer. Summarize the conversation history into a \
structured, concise summary that preserves:
1. All user instructions and requirements
2. Key decisions made
3. Files and functions referenced
4. Errors encountered and their resolutions
5. Current state of the task

Output format: Use markdown headers for each section. Be concise but complete.
Do NOT include any preamble or explanation — output only the summary.
Target length: {target_tokens} tokens maximum."""

# Budget for the summarizer LLM call itself.
SUMMARIZER_MAX_INPUT_TOKENS = 100_000
SUMMARIZER_MAX_OUTPUT_TOKENS = 20_000


class LLMClient(Protocol):
    """Protocol for LLM API clients."""

    def create_message(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Send a chat completion request and return the text response."""
        ...


class SimpleLLMClient:
    """Minimal LLM client that calls an HTTP API.

    Supports both Anthropic and OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        provider: str = "anthropic",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.provider = provider

    def create_message(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Call the LLM API and return the text response."""
        import urllib.request
        import urllib.error

        if self.provider == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
        else:
            # OpenAI-compatible
            url = f"{self.base_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}] + messages,
            }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

        if self.provider == "anthropic":
            content_blocks = result.get("content", [])
            return "".join(
                block.get("text", "") for block in content_blocks
                if block.get("type") == "text"
            )
        else:
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""


class LLMSummarizer:
    """LLM-powered conversation summarizer with deterministic fallback.

    Parameters
    ----------
    api_key:
        API key for the LLM service. If None, always falls back to
        deterministic summarization.
    model:
        Model to use for summarization.
    base_url:
        API base URL.
    provider:
        "anthropic" or "openai".
    client:
        Optional pre-built LLM client (overrides api_key/base_url/provider).
    fallback_to_deterministic:
        If True (default), falls back to deterministic summarization on failure.
    max_output_tokens:
        Maximum tokens for the LLM response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.anthropic.com",
        provider: str = "anthropic",
        client: Optional[LLMClient] = None,
        fallback_to_deterministic: bool = True,
        max_output_tokens: int = SUMMARIZER_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.model = model
        self.fallback_to_deterministic = fallback_to_deterministic
        self.max_output_tokens = max_output_tokens

        if client is not None:
            self._client: Optional[LLMClient] = client
        elif api_key:
            self._client = SimpleLLMClient(
                api_key=api_key, base_url=base_url, provider=provider
            )
        else:
            self._client = None

    @property
    def has_llm(self) -> bool:
        """Whether an LLM client is configured."""
        return self._client is not None

    def summarize(
        self,
        messages: list[dict[str, Any]],
        token_budget: int = 200_000,
        trigger_pct: float = DEFAULT_TRIGGER_PCT,
        preserve_recent_turns: int = DEFAULT_PRESERVE_RECENT_TURNS,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Summarize conversation using LLM, with deterministic fallback.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        token_budget:
            Context window size in tokens.
        trigger_pct:
            Fraction of token_budget at which summarization activates.
        preserve_recent_turns:
            Number of recent turn pairs to keep verbatim.

        Returns
        -------
        (new_messages, stats)
        """
        total_tokens = sum(
            estimate_tokens(
                m.get("content", "") if isinstance(m.get("content"), str)
                else str(m.get("content", ""))
            )
            for m in messages
        )
        threshold = int(token_budget * trigger_pct)

        if total_tokens < threshold:
            return messages, {
                "method": "none",
                "triggered": False,
                "total_tokens_before": total_tokens,
                "total_tokens_after": total_tokens,
            }

        # Split messages.
        system_msgs, body_msgs, recent_msgs = _split_messages(
            messages, preserve_recent_turns
        )

        if len(body_msgs) < 2:
            return messages, {
                "method": "none",
                "triggered": False,
                "reason": "too_few_messages",
                "total_tokens_before": total_tokens,
                "total_tokens_after": total_tokens,
            }

        # Try LLM summarization first.
        if self._client is not None:
            try:
                summary, llm_stats = self._llm_summarize(body_msgs)
                boundary_msg = self._make_boundary(
                    summary, len(body_msgs), total_tokens, method="llm"
                )
                new_messages = system_msgs + [boundary_msg] + recent_msgs
                new_total = sum(
                    estimate_tokens(
                        m.get("content", "") if isinstance(m.get("content"), str)
                        else str(m.get("content", ""))
                    )
                    for m in new_messages
                )
                return new_messages, {
                    "method": "llm",
                    "model": self.model,
                    "triggered": True,
                    "turns_summarized": len(body_msgs),
                    "total_tokens_before": total_tokens,
                    "total_tokens_after": new_total,
                    **llm_stats,
                }
            except Exception as exc:
                logger.warning("LLM summarization failed: %s", exc)
                if not self.fallback_to_deterministic:
                    return messages, {
                        "method": "llm_failed",
                        "triggered": True,
                        "error": str(exc),
                        "total_tokens_before": total_tokens,
                        "total_tokens_after": total_tokens,
                    }
                # Fall through to deterministic.

        # Deterministic fallback.
        return _deterministic_summarize(
            messages,
            token_budget=token_budget,
            trigger_pct=trigger_pct,
            preserve_recent_turns=preserve_recent_turns,
        )

    def _llm_summarize(
        self, body_msgs: list[dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
        """Call the LLM to summarize conversation body messages."""
        assert self._client is not None

        # Build the conversation text for the LLM.
        conversation_parts: list[str] = []
        total_input_tokens = 0

        for msg in body_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            line = f"[{role}]: {content}"
            line_tokens = estimate_tokens(line)
            if total_input_tokens + line_tokens > SUMMARIZER_MAX_INPUT_TOKENS:
                conversation_parts.append("[...older messages truncated...]")
                break
            conversation_parts.append(line)
            total_input_tokens += line_tokens

        system_prompt = _SUMMARIZER_SYSTEM_PROMPT.format(
            target_tokens=self.max_output_tokens
        )

        t0 = time.monotonic()
        summary = self._client.create_message(
            messages=[{"role": "user", "content": "\n\n".join(conversation_parts)}],
            system=system_prompt,
            model=self.model,
            max_tokens=self.max_output_tokens,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Enforce MAX_SUMMARY_TOKENS.
        summary_tokens = estimate_tokens(summary)
        if summary_tokens > MAX_SUMMARY_TOKENS:
            lines = summary.split("\n")
            while estimate_tokens("\n".join(lines)) > MAX_SUMMARY_TOKENS and len(lines) > 5:
                lines.pop()
            summary = "\n".join(lines) + "\n[...truncated]"

        stats = {
            "llm_input_tokens": total_input_tokens,
            "llm_output_tokens": estimate_tokens(summary),
            "llm_latency_ms": round(elapsed_ms, 2),
        }
        return summary, stats

    def _make_boundary(
        self,
        summary: str,
        turns_summarized: int,
        original_tokens: int,
        method: str = "llm",
    ) -> dict[str, Any]:
        """Create a compact_boundary system message."""
        return {
            "role": "system",
            "content": json.dumps({
                "type": "system",
                "subtype": "compact_boundary",
                "summary": summary,
                "compactMetadata": {
                    "turnsSummarized": turns_summarized,
                    "originalTokens": original_tokens,
                    "compressedTokens": estimate_tokens(summary),
                    "preservedSegment": True,
                    "method": method,
                },
            }),
        }
