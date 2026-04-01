"""CachePrefix — prompt cache prefix management for compaction.

Inspired by Claude Code's cache prefix reuse: the compaction loop and the
main conversation loop share a common prompt cache prefix. This avoids
re-processing the system prompt and early conversation turns that remain
unchanged after compaction.

The cache prefix is the longest common prefix of system messages and
early conversation turns that hasn't changed between compaction rounds.

Usage::

    from claw_compactor.fusion.cache_prefix import CachePrefixManager

    manager = CachePrefixManager()
    prefix_info = manager.compute_prefix(messages)
    # Use prefix_info['prefix_hash'] and prefix_info['prefix_tokens']
    # to enable API-level prompt caching.

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from claw_compactor.tokens import estimate_tokens


# Maximum prefix length in tokens.
MAX_PREFIX_TOKENS = 50_000


class CachePrefixManager:
    """Manages prompt cache prefix computation and reuse.

    Tracks the stable prefix of a conversation — system messages and early
    turns that don't change between compaction rounds — so that API-level
    prompt caching can skip re-processing them.
    """

    def __init__(self) -> None:
        self._last_prefix_hash: Optional[str] = None
        self._last_prefix_length: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def compute_prefix(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = MAX_PREFIX_TOKENS,
    ) -> dict[str, Any]:
        """Compute the cacheable prefix of a message list.

        The prefix includes all leading system messages plus any messages
        that appear before the first user message (or the first N messages
        that fit within max_tokens).

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        max_tokens:
            Maximum tokens for the prefix.

        Returns
        -------
        dict with:
            prefix_messages  — the messages that form the prefix
            prefix_tokens    — total tokens in the prefix
            prefix_hash      — hash of the prefix content (for cache key)
            prefix_length    — number of messages in the prefix
            cache_hit        — whether the prefix matches the last computation
        """
        prefix_messages: list[dict[str, Any]] = []
        total_tokens = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Include system messages and compact_boundary messages.
            if role == "system":
                msg_tokens = estimate_tokens(
                    content if isinstance(content, str) else str(content)
                )
                if total_tokens + msg_tokens > max_tokens:
                    break
                prefix_messages.append(msg)
                total_tokens += msg_tokens
            else:
                # Stop at first non-system message (user/assistant/tool).
                break

        # Compute hash for cache key.
        prefix_content = json.dumps(
            [
                {"role": m.get("role"), "content": m.get("content")}
                for m in prefix_messages
            ],
            sort_keys=True,
            ensure_ascii=False,
        )
        prefix_hash = hashlib.sha256(prefix_content.encode("utf-8")).hexdigest()[:16]

        # Check cache hit.
        cache_hit = prefix_hash == self._last_prefix_hash
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        self._last_prefix_hash = prefix_hash
        self._last_prefix_length = len(prefix_messages)

        return {
            "prefix_messages": prefix_messages,
            "prefix_tokens": total_tokens,
            "prefix_hash": prefix_hash,
            "prefix_length": len(prefix_messages),
            "cache_hit": cache_hit,
        }

    def annotate_messages_for_caching(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = MAX_PREFIX_TOKENS,
    ) -> list[dict[str, Any]]:
        """Annotate messages with cache_control markers for API caching.

        Adds ``cache_control: {"type": "ephemeral"}`` to the last message
        in the stable prefix, following the Anthropic prompt caching format.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        max_tokens:
            Maximum tokens for the prefix.

        Returns
        -------
        New message list with cache_control annotations.
        """
        prefix_info = self.compute_prefix(messages, max_tokens)
        prefix_length = prefix_info["prefix_length"]

        if prefix_length == 0:
            return list(messages)

        result = list(messages)
        # Mark the last prefix message with cache_control.
        last_prefix_idx = prefix_length - 1
        msg = dict(result[last_prefix_idx])

        # Handle multipart content.
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list):
            # Add cache_control to the last text block.
            new_content = list(content)
            for i in range(len(new_content) - 1, -1, -1):
                if isinstance(new_content[i], dict) and new_content[i].get("type") == "text":
                    new_content[i] = {
                        **new_content[i],
                        "cache_control": {"type": "ephemeral"},
                    }
                    break
            msg["content"] = new_content

        result[last_prefix_idx] = msg
        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 3) if total > 0 else 0.0,
            "last_prefix_hash": self._last_prefix_hash,
            "last_prefix_length": self._last_prefix_length,
        }

    def reset(self) -> None:
        """Reset cache state."""
        self._last_prefix_hash = None
        self._last_prefix_length = 0
        self._cache_hits = 0
        self._cache_misses = 0
