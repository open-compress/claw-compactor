"""
engram_utils.py — Utility functions for Engram message processing.

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from claw_compactor.tokens import estimate_tokens


def now_utc() -> str:
    """Return current UTC timestamp as a formatted string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def count_messages_tokens(messages: List[dict]) -> int:
    """Estimate token count for a list of message dicts."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text", ""))
                    total += estimate_tokens(str(block.get("input", "")))
        else:
            total += estimate_tokens(str(content))
        total += 4  # per-message overhead
    return total


def messages_to_text(messages: List[dict]) -> str:
    """Serialise a list of message dicts into a human-readable text block."""
    lines: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        ts = msg.get("timestamp", "")
        ts_str = f" [{ts}]" if ts else ""
        content = msg.get("content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        parts.append(
                            f"[tool_call: {block.get('name')} "
                            f"input={json.dumps(block.get('input', {}), ensure_ascii=False)[:200]}]"
                        )
                    elif btype == "tool_result":
                        raw = block.get("content", "")
                        if isinstance(raw, list):
                            raw = " ".join(
                                b.get("text", "") for b in raw if isinstance(b, dict)
                            )
                        parts.append(f"[tool_result: {str(raw)[:500]}]")
                    else:
                        parts.append(str(block))
            content_str = "\n".join(parts)
        else:
            content_str = str(content)

        lines.append(f"[{i + 1}] {role}{ts_str}:\n{content_str}\n")

    return "\n".join(lines)
