"""Tests for ToolResultBudget — age-based tool result truncation."""
from __future__ import annotations

import pytest
from claw_compactor.fusion.tool_result_budget import (
    budget_tool_results,
    DEFAULT_KEEP_RECENT,
    EXEMPT_TOOLS,
    MAX_TOOL_RESULT_TOKENS,
)


def _make_messages(n_tool: int = 10, tool_content_size: int = 500) -> list[dict]:
    """Helper: generate a conversation with tool results."""
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_tool):
        msgs.append({"role": "user", "content": f"Do task {i}"})
        msgs.append({"role": "assistant", "content": f"Running tool for task {i}"})
        msgs.append({
            "role": "tool",
            "name": f"bash_{i}",
            "content": f"Result line\n" * tool_content_size,
        })
    return msgs


class TestBasicTruncation:
    def test_old_tool_results_are_truncated(self):
        msgs = _make_messages(n_tool=10, tool_content_size=100)
        result, stats = budget_tool_results(msgs, keep_recent=3)
        # 10 tool msgs total, 3 kept recent, 7 should be truncated.
        assert stats["tool_results_truncated"] == 7
        assert stats["tokens_saved"] > 0

    def test_recent_tool_results_preserved(self):
        msgs = _make_messages(n_tool=5, tool_content_size=100)
        result, stats = budget_tool_results(msgs, keep_recent=5)
        # All 5 are recent, none truncated.
        assert stats["tool_results_truncated"] == 0

    def test_empty_messages(self):
        result, stats = budget_tool_results([])
        assert result == []
        assert stats["tool_results_total"] == 0

    def test_no_tool_messages(self):
        msgs = [
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result, stats = budget_tool_results(msgs)
        assert result == msgs
        assert stats["tool_results_total"] == 0


class TestExemptTools:
    def test_exempt_tools_not_truncated(self):
        msgs = [
            {"role": "tool", "name": "mcp_search", "content": "x" * 5000},
            {"role": "tool", "name": "agent_result", "content": "y" * 5000},
            {"role": "tool", "name": "bash_run", "content": "z" * 5000},
        ]
        result, stats = budget_tool_results(msgs, keep_recent=0)
        # mcp and agent are exempt, only bash should be truncated.
        assert stats["tool_results_truncated"] == 1
        # Check mcp content preserved.
        assert result[0]["content"] == "x" * 5000

    def test_custom_exempt_set(self):
        msgs = [
            {"role": "tool", "name": "my_special_tool", "content": "data " * 500},
        ]
        result, stats = budget_tool_results(
            msgs, keep_recent=0, exempt_tools=frozenset({"my_special"})
        )
        assert stats["tool_results_truncated"] == 0


class TestOversizedRecent:
    def test_oversized_recent_result_is_trimmed(self):
        big_content = "x" * 100_000  # Way over MAX_TOOL_RESULT_TOKENS
        msgs = [{"role": "tool", "name": "bash", "content": big_content}]
        result, stats = budget_tool_results(msgs, keep_recent=5)
        # It's recent but oversized — should be trimmed, not fully truncated.
        assert stats["tool_results_oversized"] == 1
        assert stats["tool_results_truncated"] == 0
        assert len(result[0]["content"]) < len(big_content)
        assert "truncated" in result[0]["content"]


class TestKeepRecentZero:
    def test_keep_zero_truncates_all(self):
        msgs = _make_messages(n_tool=3, tool_content_size=50)
        result, stats = budget_tool_results(msgs, keep_recent=0)
        assert stats["tool_results_truncated"] == 3


class TestIdempotency:
    def test_double_budget_is_idempotent(self):
        msgs = _make_messages(n_tool=8, tool_content_size=100)
        result1, stats1 = budget_tool_results(msgs, keep_recent=3)
        result2, stats2 = budget_tool_results(result1, keep_recent=3)
        # Second pass should find nothing more to truncate (already truncated).
        assert stats2["tokens_saved"] <= 10  # Minor rounding from re-estimating truncation summaries


class TestOriginalUnmodified:
    def test_original_messages_not_mutated(self):
        msgs = _make_messages(n_tool=5, tool_content_size=100)
        original_contents = [m.get("content", "") for m in msgs]
        budget_tool_results(msgs, keep_recent=2)
        # Original should be unchanged.
        for i, msg in enumerate(msgs):
            assert msg.get("content", "") == original_contents[i]
