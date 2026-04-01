"""Tests for TieredCompaction — three-level compaction strategy."""
from __future__ import annotations

import pytest
from claw_compactor.fusion.tiered_compaction import (
    CompactionLevel,
    CircuitBreaker,
    FileAccessTracker,
    compact,
    determine_level,
    MICRO_THRESHOLD,
    AUTO_THRESHOLD,
    FULL_THRESHOLD,
    MAX_CONSECUTIVE_FAILURES,
)


def _make_messages_with_tokens(target_tokens: int) -> list[dict]:
    """Generate messages that total approximately target_tokens."""
    # ~4 chars per token (rough estimate).
    content_per_msg = "x" * 400  # ~100 tokens per message
    n_msgs = max(1, target_tokens // 100)
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": content_per_msg})
    return msgs


class TestDetermineLevel:
    def test_none_when_under_micro(self):
        msgs = _make_messages_with_tokens(100)
        level = determine_level(msgs, token_budget=200_000)
        assert level == CompactionLevel.NONE

    def test_micro_level(self):
        # 60-80% of budget.
        msgs = _make_messages_with_tokens(140_000)
        level = determine_level(msgs, token_budget=200_000)
        assert level == CompactionLevel.MICRO

    def test_auto_level(self):
        # 80-95% of budget.
        msgs = _make_messages_with_tokens(170_000)
        level = determine_level(msgs, token_budget=200_000)
        assert level == CompactionLevel.AUTO

    def test_full_level(self):
        # >95% of budget.
        msgs = _make_messages_with_tokens(195_000)
        level = determine_level(msgs, token_budget=200_000)
        assert level == CompactionLevel.FULL


class TestCircuitBreaker:
    def test_not_disabled_initially(self):
        cb = CircuitBreaker()
        assert not cb.disabled

    def test_disabled_after_max_failures(self):
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.disabled
        cb.record_failure()
        assert cb.disabled

    def test_success_resets_consecutive(self):
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.consecutive_failures == 0
        assert not cb.disabled

    def test_reset_clears_disabled(self):
        cb = CircuitBreaker(max_failures=1)
        cb.record_failure()
        assert cb.disabled
        cb.reset()
        assert not cb.disabled

    def test_to_dict(self):
        cb = CircuitBreaker()
        cb.record_failure()
        d = cb.to_dict()
        assert d["consecutive_failures"] == 1
        assert d["disabled"] is False
        assert d["total_failures"] == 1


class TestFileAccessTracker:
    def test_record_and_retrieve(self):
        tracker = FileAccessTracker()
        tracker.record_access("/src/main.py", "def main(): pass")
        tracker.record_access("/src/utils.py", "def util(): pass")
        files = tracker.get_recent_files()
        assert len(files) == 2
        # Most recent first.
        assert files[0]["path"] == "/src/utils.py"

    def test_budget_enforcement(self):
        tracker = FileAccessTracker()
        big_content = "x\n" * 50_000  # Very large file
        tracker.record_access("/big.py", big_content)
        files = tracker.get_recent_files(per_file_budget=100, total_budget=200)
        assert len(files) == 1
        assert files[0]["tokens"] <= 200

    def test_duplicate_access_updates(self):
        tracker = FileAccessTracker()
        tracker.record_access("/f.py", "v1")
        tracker.record_access("/g.py", "other")
        tracker.record_access("/f.py", "v2")
        files = tracker.get_recent_files()
        # /f.py should be most recent with v2 content.
        assert files[0]["path"] == "/f.py"
        assert files[0]["content"] == "v2"

    def test_total_budget_limits_files(self):
        tracker = FileAccessTracker()
        for i in range(20):
            tracker.record_access(f"/file_{i}.py", "content " * 100)
        files = tracker.get_recent_files(per_file_budget=5000, total_budget=500)
        # Should get far fewer than 20 files due to total budget.
        assert len(files) < 20


class TestCompactFunction:
    def test_compact_returns_unchanged_when_no_pressure(self):
        msgs = [
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
        result, stats = compact(msgs, token_budget=200_000)
        assert stats["level"] == "none"
        assert result == msgs

    def test_compact_micro_truncates_tool_results(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 400})
            msgs.append({"role": "assistant", "content": "y" * 400})
            msgs.append({"role": "tool", "name": f"bash_{i}", "content": "z" * 2000})
        result, stats = compact(
            msgs, token_budget=200_000, level_override=CompactionLevel.MICRO
        )
        assert stats["level"] == "micro"
        assert "tool_budget" in stats

    def test_compact_auto_summarizes(self):
        msgs = _make_messages_with_tokens(170_000)
        result, stats = compact(
            msgs, token_budget=200_000, level_override=CompactionLevel.AUTO
        )
        assert stats["level"] == "auto"
        assert "summarization" in stats

    def test_circuit_breaker_blocks_compaction(self):
        cb = CircuitBreaker(max_failures=1)
        cb.record_failure()  # Trip the breaker.
        assert cb.disabled
        msgs = _make_messages_with_tokens(195_000)
        result, stats = compact(msgs, token_budget=200_000, circuit_breaker=cb)
        assert stats["level"] == "none"
        assert stats["reason"] == "circuit_breaker_disabled"

    def test_compact_with_level_override(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result, stats = compact(
            msgs, token_budget=200_000, level_override=CompactionLevel.MICRO
        )
        assert stats["level"] == "micro"


class TestCompactIdempotency:
    def test_double_compact_stable(self):
        msgs = _make_messages_with_tokens(5000)
        result1, _ = compact(msgs, token_budget=200_000, level_override=CompactionLevel.AUTO)
        result2, stats2 = compact(result1, token_budget=200_000, level_override=CompactionLevel.AUTO)
        # Second pass should not significantly change token count.
        assert abs(stats2.get("tokens_saved", 0)) < 100


class TestCompactWithFileReinjection:
    def test_full_compact_reinjects_files(self):
        tracker = FileAccessTracker()
        tracker.record_access("/src/app.py", "def main(): pass\n" * 10)
        msgs = _make_messages_with_tokens(5000)
        from claw_compactor.fusion.engine import FusionEngine
        engine = FusionEngine(enable_rewind=False)
        result, stats = compact(
            msgs,
            token_budget=200_000,
            file_tracker=tracker,
            fusion_engine=engine,
            level_override=CompactionLevel.FULL,
        )
        assert stats.get("files_reinjected", 0) >= 1
        # Check that a system message with file content was added.
        system_msgs = [m for m in result if m.get("role") == "system"]
        file_injection = [m for m in system_msgs if "Recently Accessed Files" in m.get("content", "")]
        assert len(file_injection) == 1
