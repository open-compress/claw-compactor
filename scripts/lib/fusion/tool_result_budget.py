"""ToolResultBudget — age-based tool result truncation for chat messages.

Inspired by Claude Code's MicroCompact layer: old tool results are truncated
to a short summary, keeping only the N most recent tool results intact.

This operates at the message-list level (not per-message text compression).
It modifies the ``content`` of tool-role messages that exceed the age threshold.

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import re
from typing import Any

from claw_compactor.tokens import estimate_tokens


# Tools whose results should NEVER be truncated (analogous to Claude Code's
# exemption of MCP/Agent/custom tool results from microcompact).
EXEMPT_TOOLS: frozenset[str] = frozenset({
    "mcp",           # MCP tool results
    "agent",         # Sub-agent outputs
    "memory",        # Memory retrieval
    "rewind",        # Rewind store lookups
    "file_search",   # File search results (user may re-reference)
})

# Default: keep the 5 most recent tool results untruncated.
DEFAULT_KEEP_RECENT = 5

# When truncating, replace tool content with a short summary of this form.
_TRUNCATION_TEMPLATE = "[tool result truncated — was {tokens} tokens, {chars} chars]"

# Maximum tokens allowed per tool result (results larger than this are
# candidates for truncation even within the keep-recent window).
MAX_TOOL_RESULT_TOKENS = 8000


def budget_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int = DEFAULT_KEEP_RECENT,
    max_tokens_per_result: int = MAX_TOOL_RESULT_TOKENS,
    exempt_tools: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Truncate old/oversized tool results in a message list.

    Parameters
    ----------
    messages:
        List of OpenAI-format chat messages.
    keep_recent:
        Number of most-recent tool results to keep untruncated.
    max_tokens_per_result:
        Hard cap per tool result — even recent results exceeding this
        are truncated to a summary + first 200 chars.
    exempt_tools:
        Tool names whose results are never truncated.

    Returns
    -------
    (new_messages, stats) where stats reports truncation counts and savings.
    """
    if exempt_tools is None:
        exempt_tools = EXEMPT_TOOLS

    # Identify all tool-result message indices (reverse order = most recent first).
    tool_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_indices.append(i)

    # The last `keep_recent` tool messages are "recent" and protected.
    recent_set = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()

    result_messages = list(messages)  # shallow copy
    stats = {
        "tool_results_total": len(tool_indices),
        "tool_results_truncated": 0,
        "tool_results_oversized": 0,
        "tokens_saved": 0,
        "chars_saved": 0,
    }

    for idx in tool_indices:
        msg = result_messages[idx]
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            continue

        tool_name = msg.get("name", msg.get("tool_call_id", "")).lower()
        # Check exemption.
        if any(exempt in tool_name for exempt in exempt_tools):
            continue

        original_tokens = estimate_tokens(content)
        is_recent = idx in recent_set

        # Case 1: Recent but oversized → trim to first 200 chars + summary.
        if is_recent and original_tokens > max_tokens_per_result:
            preview = content[:200].rstrip()
            truncated = (
                f"{preview}...\n\n"
                f"[truncated from {original_tokens} tokens — result too large]"
            )
            new_tokens = estimate_tokens(truncated)
            result_messages[idx] = {**msg, "content": truncated}
            stats["tool_results_oversized"] += 1
            stats["tokens_saved"] += original_tokens - new_tokens
            stats["chars_saved"] += len(content) - len(truncated)
            continue

        # Case 2: Old tool result → full truncation to summary line.
        if not is_recent:
            truncated = _TRUNCATION_TEMPLATE.format(
                tokens=original_tokens, chars=len(content)
            )
            new_tokens = estimate_tokens(truncated)
            result_messages[idx] = {**msg, "content": truncated}
            stats["tool_results_truncated"] += 1
            stats["tokens_saved"] += original_tokens - new_tokens
            stats["chars_saved"] += len(content) - len(truncated)

    return result_messages, stats
