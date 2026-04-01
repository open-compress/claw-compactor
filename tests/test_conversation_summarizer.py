"""Tests for ConversationSummarizer — deterministic conversation summarization."""
from __future__ import annotations

import json
import pytest
from claw_compactor.fusion.conversation_summarizer import (
    summarize_conversation,
    _split_messages,
    _extract_summary,
    DEFAULT_TRIGGER_PCT,
)
from claw_compactor.tokens import estimate_tokens


def _make_long_conversation(n_turns: int = 20, content_size: int = 500) -> list[dict]:
    """Generate a conversation that will exceed a small token budget."""
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        msgs.append({
            "role": "user",
            "content": f"Please fix the bug in /src/app/module_{i}.py\n" + "context " * content_size,
        })
        msgs.append({
            "role": "assistant",
            "content": f"I decided to use the refactoring approach for module_{i}. "
                       f"def fix_bug_{i}(): pass\n"
                       f"Error: NullPointerException in handler_{i}\n" + "analysis " * content_size,
        })
    return msgs


class TestSummarizationTrigger:
    def test_no_trigger_under_budget(self):
        msgs = [
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result, stats = summarize_conversation(msgs, token_budget=200_000)
        assert not stats["triggered"]
        assert result == msgs

    def test_triggers_when_over_threshold(self):
        msgs = _make_long_conversation(n_turns=20, content_size=500)
        # Use a very small budget to force triggering.
        result, stats = summarize_conversation(msgs, token_budget=1000)
        assert stats["triggered"]
        assert stats["turns_summarized"] > 0
        assert stats["total_tokens_after"] < stats["total_tokens_before"]


class TestCompactBoundary:
    def test_compact_boundary_message_created(self):
        msgs = _make_long_conversation(n_turns=10, content_size=200)
        result, stats = summarize_conversation(msgs, token_budget=500)
        # Find the compact_boundary message.
        boundaries = [m for m in result if m.get("role") == "system"
                      and "compact_boundary" in m.get("content", "")]
        assert len(boundaries) == 1
        boundary = json.loads(boundaries[0]["content"])
        assert boundary["type"] == "system"
        assert boundary["subtype"] == "compact_boundary"
        assert boundary["compactMetadata"]["preservedSegment"] is True
        assert boundary["compactMetadata"]["turnsSummarized"] > 0

    def test_summary_contains_user_instructions(self):
        msgs = _make_long_conversation(n_turns=10, content_size=200)
        result, stats = summarize_conversation(msgs, token_budget=500)
        boundaries = [m for m in result if "compact_boundary" in m.get("content", "")]
        boundary = json.loads(boundaries[0]["content"])
        assert "User Instructions" in boundary["summary"]

    def test_summary_contains_file_paths(self):
        msgs = _make_long_conversation(n_turns=10, content_size=200)
        result, stats = summarize_conversation(msgs, token_budget=500)
        boundaries = [m for m in result if "compact_boundary" in m.get("content", "")]
        boundary = json.loads(boundaries[0]["content"])
        assert "Files Referenced" in boundary["summary"]
        assert "/src/app/module_" in boundary["summary"]


class TestRecentTurnsPreserved:
    def test_recent_turns_kept_verbatim(self):
        msgs = _make_long_conversation(n_turns=10, content_size=200)
        result, stats = summarize_conversation(
            msgs, token_budget=500, preserve_recent_turns=3
        )
        # The last 3 user messages should still be in result.
        user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(user_msgs) >= 3

    def test_system_messages_preserved(self):
        msgs = _make_long_conversation(n_turns=10, content_size=200)
        result, _ = summarize_conversation(msgs, token_budget=500)
        # First system message should still be there.
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."


class TestSplitMessages:
    def test_split_basic(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        system, body, recent = _split_messages(msgs, preserve_recent_turns=2)
        assert len(system) == 1
        assert system[0]["content"] == "sys"
        # Recent should have last 2 turns (u2,a2,u3,a3).
        recent_roles = [m["role"] for m in recent]
        assert "u2" in [m["content"] for m in recent]
        assert len(body) > 0


class TestExtractSummary:
    def test_extracts_decisions(self):
        msgs = [
            {"role": "assistant", "content": "I decided to use async/await pattern."},
        ]
        lines = _extract_summary(msgs)
        text = "\n".join(lines)
        assert "Decisions" in text or "async" in text.lower()

    def test_extracts_errors(self):
        msgs = [
            {"role": "assistant", "content": "Error: connection refused on port 8080"},
        ]
        lines = _extract_summary(msgs)
        text = "\n".join(lines)
        assert "Error" in text

    def test_extracts_file_paths(self):
        msgs = [
            {"role": "user", "content": "Fix /src/main.py and /lib/utils.ts"},
        ]
        lines = _extract_summary(msgs)
        text = "\n".join(lines)
        assert "/src/main.py" in text


class TestEdgeCases:
    def test_all_system_messages(self):
        msgs = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ]
        result, stats = summarize_conversation(msgs, token_budget=10)
        # Not enough body to summarize.
        assert not stats["triggered"] or stats["turns_summarized"] == 0

    def test_single_turn(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result, stats = summarize_conversation(msgs, token_budget=5)
        # Only 2 messages — might trigger but split yields no body.
        assert len(result) >= 1

    def test_tool_messages_in_summary(self):
        msgs = [
            {"role": "user", "content": "run tests"},
            {"role": "tool", "name": "bash", "content": "PASS 100 tests in 2.3s"},
            {"role": "assistant", "content": "All tests pass."},
            {"role": "user", "content": "great"},
            {"role": "assistant", "content": "Anything else?"},
        ]
        result, stats = summarize_conversation(msgs, token_budget=10)
        if stats["triggered"]:
            boundaries = [m for m in result if "compact_boundary" in m.get("content", "")]
            if boundaries:
                boundary = json.loads(boundaries[0]["content"])
                assert "Actions" in boundary["summary"] or "bash" in boundary["summary"]
