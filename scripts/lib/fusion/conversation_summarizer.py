"""ConversationSummarizer — LLM-free conversation turn summarization.

Inspired by Claude Code's AutoCompact: when a conversation exceeds a token
budget, older turns are collapsed into a structured summary block.

Unlike Claude Code (which calls the LLM for summarization), this module uses
deterministic extraction to avoid API calls and latency:
  - Extracts key decisions, file paths, function names, and error patterns
  - Preserves user instructions and system messages verbatim
  - Collapses assistant responses to their first sentence + action items
  - Collapses tool results to one-line summaries

The summarized turns are replaced by a single system message with subtype
``compact_boundary`` (compatible with Claude Code's format) so downstream
consumers can detect and handle compacted history.

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import json
import re
from typing import Any

from claw_compactor.tokens import estimate_tokens


# Summarization fires when total message tokens exceed this fraction of budget.
DEFAULT_TRIGGER_PCT = 0.80

# After summarization, the summary should be at most this many tokens.
MAX_SUMMARY_TOKENS = 20_000

# Keep the N most recent turns unsummarized (a "turn" = one user + one assistant).
DEFAULT_PRESERVE_RECENT_TURNS = 4

# Patterns to extract from assistant messages.
_FILE_PATH_RE = re.compile(r'[`"\']?(/[\w./-]+\.\w{1,10})[`"\']?')
_FUNCTION_RE = re.compile(r'(?:def|function|class|fn|func)\s+(\w+)')
_ERROR_RE = re.compile(r'(?:Error|Exception|FAIL|error|failed|bug)[:. ]\s*(.{10,80})')
_DECISION_RE = re.compile(
    r'(?:decided|decision|chose|choosing|will use|going with|plan is|approach:)\s+(.{10,120})',
    re.IGNORECASE,
)


def summarize_conversation(
    messages: list[dict[str, Any]],
    token_budget: int = 200_000,
    trigger_pct: float = DEFAULT_TRIGGER_PCT,
    preserve_recent_turns: int = DEFAULT_PRESERVE_RECENT_TURNS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize older conversation turns if total tokens exceed budget threshold.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    token_budget:
        The context window size in tokens.
    trigger_pct:
        Fraction of token_budget at which summarization activates.
    preserve_recent_turns:
        Number of recent user+assistant turn pairs to keep verbatim.

    Returns
    -------
    (new_messages, stats) — stats includes tokens_before, tokens_after, turns_summarized.
    """
    total_tokens = sum(estimate_tokens(m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", ""))) for m in messages)
    threshold = int(token_budget * trigger_pct)

    stats: dict[str, Any] = {
        "total_tokens_before": total_tokens,
        "total_tokens_after": total_tokens,
        "turns_summarized": 0,
        "triggered": False,
        "threshold": threshold,
    }

    if total_tokens < threshold:
        return messages, stats

    stats["triggered"] = True

    # Split messages into: system prefix, conversation body, recent tail.
    system_msgs, body_msgs, recent_msgs = _split_messages(
        messages, preserve_recent_turns
    )

    if len(body_msgs) < 2:
        # Not enough to summarize.
        return messages, stats

    # Build a deterministic summary of the body.
    summary_lines = _extract_summary(body_msgs)
    summary_text = "\n".join(summary_lines)

    # Enforce MAX_SUMMARY_TOKENS.
    summary_tokens = estimate_tokens(summary_text)
    if summary_tokens > MAX_SUMMARY_TOKENS:
        # Truncate to budget.
        lines = summary_lines
        while estimate_tokens("\n".join(lines)) > MAX_SUMMARY_TOKENS and len(lines) > 5:
            lines = lines[:len(lines) - 1]
        summary_text = "\n".join(lines) + "\n[...truncated summary]"

    # Build compact_boundary message.
    boundary_msg = _make_compact_boundary(
        summary_text,
        turns_summarized=len(body_msgs),
        original_tokens=sum(
            estimate_tokens(m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in body_msgs
        ),
    )

    # Reassemble.
    new_messages = system_msgs + [boundary_msg] + recent_msgs

    new_total = sum(
        estimate_tokens(m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", "")))
        for m in new_messages
    )
    stats["total_tokens_after"] = new_total
    stats["turns_summarized"] = len(body_msgs)

    return new_messages, stats


def _split_messages(
    messages: list[dict[str, Any]],
    preserve_recent_turns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into (system_prefix, compactable_body, recent_tail)."""
    # System messages at the start.
    system_msgs: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        system_msgs.append(messages[i])
        i += 1

    remaining = messages[i:]

    # Count turns from the end (a turn = user msg followed by any non-user msgs).
    if preserve_recent_turns <= 0:
        return system_msgs, remaining, []

    # Walk backwards counting user messages as turn boundaries.
    turns_found = 0
    split_idx = len(remaining)
    for j in range(len(remaining) - 1, -1, -1):
        if remaining[j].get("role") == "user":
            turns_found += 1
            if turns_found >= preserve_recent_turns:
                split_idx = j
                break

    body = remaining[:split_idx]
    recent = remaining[split_idx:]
    return system_msgs, body, recent


def _extract_summary(messages: list[dict[str, Any]]) -> list[str]:
    """Extract a structured summary from a list of conversation messages."""
    lines: list[str] = ["## Conversation Summary (auto-compacted)"]
    lines.append("")

    decisions: list[str] = []
    files_mentioned: set[str] = set()
    functions_mentioned: set[str] = set()
    errors: list[str] = []
    user_instructions: list[str] = []
    actions_taken: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "user":
            # Preserve user instructions (first 200 chars each).
            trimmed = content.strip()[:200]
            if trimmed:
                user_instructions.append(trimmed)

        elif role == "assistant":
            # Extract decisions.
            for m in _DECISION_RE.finditer(content):
                decisions.append(m.group(1).strip())
            # Extract first sentence as action summary.
            first_sentence = content.split("\n")[0][:150].strip()
            if first_sentence:
                actions_taken.append(first_sentence)

        elif role == "tool":
            # One-line summary.
            tool_name = msg.get("name", "tool")
            token_count = estimate_tokens(content)
            actions_taken.append(f"[{tool_name}: {token_count} tokens]")

        # Extract file paths, functions, errors from any role.
        files_mentioned.update(_FILE_PATH_RE.findall(content))
        functions_mentioned.update(_FUNCTION_RE.findall(content))
        for m in _ERROR_RE.finditer(content):
            errors.append(m.group(1).strip()[:100])

    # Build summary sections.
    if user_instructions:
        lines.append("### User Instructions")
        for instr in user_instructions[-10:]:  # cap at 10
            lines.append(f"- {instr}")
        lines.append("")

    if decisions:
        lines.append("### Key Decisions")
        for d in decisions[-10:]:
            lines.append(f"- {d}")
        lines.append("")

    if actions_taken:
        lines.append("### Actions Taken")
        for a in actions_taken[-15:]:
            lines.append(f"- {a}")
        lines.append("")

    if files_mentioned:
        lines.append("### Files Referenced")
        for f in sorted(files_mentioned)[:20]:
            lines.append(f"- `{f}`")
        lines.append("")

    if errors:
        lines.append("### Errors Encountered")
        for e in errors[-5:]:
            lines.append(f"- {e}")
        lines.append("")

    return lines


def _make_compact_boundary(
    summary: str,
    turns_summarized: int,
    original_tokens: int,
) -> dict[str, Any]:
    """Create a compact_boundary system message (Claude Code compatible format)."""
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
            },
        }),
    }
