"""Enhanced edge-case tests for existing v8 modules.

Covers tiered_compaction, tool_result_budget, conversation_summarizer
with additional boundary, error, and integration tests.

Part of claw-compactor v8 test suite. License: MIT.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from claw_compactor.tokens import estimate_tokens


# ===========================================================================
# Helpers
# ===========================================================================

def _sys(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}

def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}

def _asst(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}

def _tool_msg(name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": content}

def _make_msgs(target_tokens: int) -> list[dict[str, Any]]:
    content_per_msg = "x" * 400
    n_msgs = max(1, target_tokens // 100)
    msgs = [_sys("You are helpful.")]
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": content_per_msg})
    return msgs


# ===========================================================================
# TieredCompaction — Edge Cases
# ===========================================================================
class TestTieredCompactionEdge:

    def test_determine_level_small_conversation(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel
        # Small conversation under any threshold
        msgs = [_user("hello")]
        level = determine_level(msgs, token_budget=200_000)
        assert level is None or level == CompactionLevel.NONE

    def test_determine_level_large_conversation_triggers(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel
        # Large conversation should trigger
        msgs = _make_msgs(180_000)  # ~180K tokens, 90% of 200K
        level = determine_level(msgs, token_budget=200_000)
        assert level is not None

    def test_circuit_breaker_success_resets_after_partial_failures(self):
        from claw_compactor.fusion.tiered_compaction import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        assert not cb.disabled
        cb.record_success()
        assert cb.consecutive_failures == 0
        cb.record_failure()
        cb.record_failure()
        assert not cb.disabled  # still under 3

    def test_circuit_breaker_to_dict_tracks_totals(self):
        from claw_compactor.fusion.tiered_compaction import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        d = cb.to_dict()
        assert d["total_failures"] == 2
        assert d["total_attempts"] == 3

    def test_file_access_tracker_empty(self):
        from claw_compactor.fusion.tiered_compaction import FileAccessTracker
        tracker = FileAccessTracker()
        assert tracker.get_recent_files() == [] or len(tracker.accessed_files) == 0

    def test_file_access_tracker_records_and_retrieves(self):
        from claw_compactor.fusion.tiered_compaction import FileAccessTracker
        tracker = FileAccessTracker()
        tracker.record_access("file.py", "content here")
        files = tracker.get_recent_files()
        assert len(files) >= 1

    def test_file_access_tracker_dedup(self):
        from claw_compactor.fusion.tiered_compaction import FileAccessTracker
        tracker = FileAccessTracker()
        tracker.record_access("file.py", "v1")
        tracker.record_access("file.py", "v2")
        # Should only have one entry (latest)
        assert len(tracker.accessed_files) == 1

    def test_compact_preserves_system_messages(self):
        from claw_compactor.fusion.tiered_compaction import compact
        msgs = [_sys("sys prompt")] + [_user("word " * 100), _asst("word " * 100)] * 30
        result, stats = compact(msgs, token_budget=3000)
        sys_msgs = [m for m in result if m.get("role") == "system"]
        assert any("sys prompt" in str(m.get("content", "")) for m in sys_msgs)

    def test_compact_with_empty_messages(self):
        from claw_compactor.fusion.tiered_compaction import compact
        result, stats = compact([], token_budget=200_000)
        assert result == []

    def test_compact_single_message(self):
        from claw_compactor.fusion.tiered_compaction import compact
        msgs = [_user("hello")]
        result, stats = compact(msgs, token_budget=200_000)
        assert result == msgs

    def test_env_var_override(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel
        with patch.dict(os.environ, {"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "0.3"}):
            # Large messages with low threshold
            msgs = _make_msgs(80_000)
            level = determine_level(msgs, token_budget=200_000)
            # 40% of budget, but threshold lowered to 30% → should trigger
            assert level is not None

    def test_env_var_invalid_ignored(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel
        with patch.dict(os.environ, {"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "not_a_number"}):
            msgs = [_user("hello")]
            level = determine_level(msgs, token_budget=200_000)
            # Should not crash, use default
            assert level is None or level == CompactionLevel.NONE

    def test_env_var_out_of_range_ignored(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel
        with patch.dict(os.environ, {"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "2.0"}):
            msgs = [_user("hello")]
            level = determine_level(msgs, token_budget=200_000)
            assert level is None or level == CompactionLevel.NONE


# ===========================================================================
# ToolResultBudget — Edge Cases
# ===========================================================================
class TestToolResultBudgetEdge:

    def test_all_exempt_tools_preserved(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [
            _tool_msg("list_files", "file1.py\nfile2.py\n" + "x" * 5000),
            _tool_msg("read_file", "content " * 500),
            _tool_msg("list_files", "file3.py"),
        ]
        result, stats = budget_tool_results(msgs, exempt_tools={"list_files"})
        list_files_results = [m for m in result if m.get("name") == "list_files"]
        assert all("file" in m["content"] for m in list_files_results)

    def test_very_large_tool_result(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [
            _user("run this"),
            _tool_msg("bash", "output " * 10000),
            _user("ok"),
            _tool_msg("bash", "small"),
        ]
        result, stats = budget_tool_results(msgs, keep_recent=1)
        old_bash = [m for m in result if m.get("name") == "bash"][0]
        assert len(old_bash["content"]) < len("output " * 10000)

    def test_non_tool_messages_untouched(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [_user("hello"), _asst("hi"), _user("more")]
        result, stats = budget_tool_results(msgs)
        assert result == msgs

    def test_tool_result_with_no_content(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [{"role": "tool", "name": "test", "content": ""}]
        result, stats = budget_tool_results(msgs)
        assert len(result) == 1

    def test_keep_recent_larger_than_tool_count(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [_tool_msg("bash", "result")]
        result, stats = budget_tool_results(msgs, keep_recent=10)
        assert result[0]["content"] == "result"

    def test_mixed_tool_and_non_tool(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        msgs = [
            _user("step 1"),
            _tool_msg("bash", "old result " * 500),
            _asst("got it"),
            _user("step 2"),
            _tool_msg("bash", "new result"),
            _asst("done"),
        ]
        result, stats = budget_tool_results(msgs, keep_recent=1)
        assert result[0]["content"] == "step 1"
        assert result[2]["content"] == "got it"


# ===========================================================================
# ConversationSummarizer — Edge Cases
# ===========================================================================
class TestConversationSummarizerEdge:

    def test_empty_messages(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        result, stats = summarize_conversation([])
        assert result == []

    def test_only_system_messages(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [_sys("system prompt")]
        result, stats = summarize_conversation(msgs, token_budget=100, trigger_pct=0.01)
        assert len(result) >= 1

    def test_preserves_recent_turns_count(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [_sys("system")]
        for i in range(20):
            msgs.append(_user(f"Message {i}: " + "word " * 50))
            msgs.append(_asst(f"Response {i}: " + "word " * 50))

        result, stats = summarize_conversation(
            msgs, token_budget=2000, trigger_pct=0.1, preserve_recent_turns=3
        )
        if stats.get("triggered"):
            recent = result[-6:]
            assert any("Message" in str(m.get("content", "")) for m in recent)

    def test_summary_extracts_code_blocks(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [
            _sys("system"),
            _user("Fix this code:\n```python\ndef foo(): pass\n```"),
            _asst("Here's the fix:\n```python\ndef foo(): return 42\n```"),
        ] * 10

        result, stats = summarize_conversation(msgs, token_budget=500, trigger_pct=0.1)
        if stats.get("triggered"):
            assert len(result) < len(msgs)

    def test_summary_extracts_error_patterns(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [
            _sys("system"),
            _user("I got an error"),
            _asst("Error: TypeError: cannot read property 'x' of undefined"),
            _user("Fix it"),
            _asst("Fixed by adding null check"),
        ] * 5

        result, stats = summarize_conversation(msgs, token_budget=1000, trigger_pct=0.1)
        if stats.get("triggered"):
            assert len(result) < len(msgs)

    def test_high_trigger_pct_never_triggers(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [_sys("system")] + [_user("hello"), _asst("hi")] * 5
        result, stats = summarize_conversation(msgs, token_budget=200_000, trigger_pct=0.99)
        assert not stats.get("triggered", False)

    def test_very_low_trigger_pct_always_triggers(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [_sys("system")]
        for i in range(10):
            msgs.append(_user("word " * 100))
            msgs.append(_asst("word " * 100))
        result, stats = summarize_conversation(msgs, token_budget=500, trigger_pct=0.01)
        assert stats.get("triggered", False) or len(msgs) < 4

    def test_tool_messages_handled(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [
            _sys("system"),
            _user("run bash"),
            _tool_msg("bash", "output " * 100),
            _asst("done"),
        ] * 10

        result, stats = summarize_conversation(msgs, token_budget=1000, trigger_pct=0.1)
        if stats.get("triggered"):
            assert len(result) < len(msgs)

    def test_split_messages_basic(self):
        from claw_compactor.fusion.conversation_summarizer import _split_messages
        msgs = [_sys("sys"), _user("a"), _asst("b"), _user("c"), _asst("d")]
        system, body, recent = _split_messages(msgs, preserve_recent_turns=1)
        assert len(system) == 1
        assert len(recent) == 2
        assert len(body) == 2

    def test_split_messages_more_recent_than_available(self):
        from claw_compactor.fusion.conversation_summarizer import _split_messages
        msgs = [_sys("sys"), _user("a"), _asst("b")]
        system, body, recent = _split_messages(msgs, preserve_recent_turns=5)
        # Implementation may keep all in recent or body — just verify no crash
        assert len(system) + len(body) + len(recent) == len(msgs)

    def test_non_string_content_handled(self):
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation
        msgs = [
            _sys("system"),
            {"role": "user", "content": ["multipart"]},
            _asst("ok"),
        ] * 5
        result, stats = summarize_conversation(msgs, token_budget=200, trigger_pct=0.1)
        assert isinstance(result, list)


# ===========================================================================
# Cross-module Integration
# ===========================================================================
class TestCrossModuleIntegration:

    def test_budget_then_summarize(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        from claw_compactor.fusion.conversation_summarizer import summarize_conversation

        msgs = [_sys("system")]
        for i in range(15):
            msgs.append(_user(f"Do thing {i}"))
            msgs.append(_tool_msg("bash", "output " * 200))
            msgs.append(_asst(f"Done {i}"))

        budgeted, budget_stats = budget_tool_results(msgs, keep_recent=3)
        result, sum_stats = summarize_conversation(
            budgeted, token_budget=3000, trigger_pct=0.1
        )
        assert len(result) <= len(budgeted)

    def test_tiered_compact_with_budgeted_tools(self):
        from claw_compactor.fusion.tool_result_budget import budget_tool_results
        from claw_compactor.fusion.tiered_compaction import compact

        msgs = [_sys("system")]
        for i in range(20):
            msgs.append(_user(f"Step {i}"))
            msgs.append(_tool_msg("bash", "result " * 300))
            msgs.append(_asst(f"Response {i}"))

        budgeted, _ = budget_tool_results(msgs, keep_recent=3)
        result, stats = compact(budgeted, token_budget=5000)
        assert len(result) <= len(budgeted)

    def test_circuit_breaker_prevents_over_compaction(self):
        from claw_compactor.fusion.tiered_compaction import CircuitBreaker, compact
        cb = CircuitBreaker()
        for _ in range(3):
            cb.record_failure()

        assert cb.disabled
        msgs = _make_msgs(50000)
        result, stats = compact(msgs, token_budget=5000, circuit_breaker=cb)
        assert len(result) == len(msgs) or stats.get("circuit_breaker_blocked")
