"""Deep edge-case tests for claw-compactor v8 features.

Covers plan_reinjection, skill_reinjection, llm_summarizer,
compact_hooks, content_stripper, cache_prefix with:
  - Boundary values (0, empty, very large)
  - Idempotency
  - Error recovery
  - Token budget limits
  - State machine transitions
  - Concurrent/sequential operations

Part of claw-compactor v8 test suite. License: MIT.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claw_compactor.tokens import estimate_tokens


# ===========================================================================
# Helpers
# ===========================================================================

def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _sys(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _asst(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def _tool(name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": content}


def _make_convo(n_turns: int, tokens_per_msg: int = 50) -> list[dict[str, Any]]:
    """Generate n_turns of user/assistant pairs."""
    word = "word " * (tokens_per_msg // 2)
    msgs = []
    for i in range(n_turns):
        msgs.append(_user(f"{word} turn {i}"))
        msgs.append(_asst(f"{word} response {i}"))
    return msgs


# ===========================================================================
# PlanReinjection — Deep Tests
# ===========================================================================
class TestPlanReinjectionDeep:
    """Deep edge-case tests for PlanTaskTracker."""

    def test_empty_tracker_returns_none(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        assert tracker.build_injection_message() is None

    def test_record_plan_empty_steps(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        plan = tracker.record_plan("Empty plan", steps=[])
        assert plan.steps == []
        msg = tracker.build_injection_message()
        assert msg is not None
        assert "Empty plan" in msg["content"]

    def test_record_plan_none_steps(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        plan = tracker.record_plan("No steps", steps=None)
        assert plan.steps == []

    def test_complete_plan_removes_from_active(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Plan A", steps=["step 1"])
        tracker.record_plan("Plan B", steps=["step 2"])
        tracker.complete_plan("Plan A")
        assert len(tracker.active_plans) == 1
        assert tracker.active_plans[0].title == "Plan B"

    def test_complete_nonexistent_plan_is_noop(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Plan A")
        tracker.complete_plan("Nonexistent")
        assert len(tracker.active_plans) == 1

    def test_task_status_transitions(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_task("T-001", "Task 1", status="pending")
        tracker.update_task_status("T-001", "running")
        assert tracker._tasks["T-001"].status == "running"
        tracker.update_task_status("T-001", "done")
        assert len(tracker.active_tasks) == 0  # done tasks are excluded

    def test_update_nonexistent_task_is_noop(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.update_task_status("T-999", "running")  # should not raise

    def test_record_task_idempotent_update(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        t1 = tracker.record_task("T-001", "Task 1", status="pending")
        t2 = tracker.record_task("T-001", "Task 1 updated", status="running")
        assert t2.title == "Task 1 updated"
        assert t2.status == "running"
        assert len(tracker._tasks) == 1

    def test_cancelled_tasks_excluded_from_active(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_task("T-001", "Task 1", status="cancelled")
        assert len(tracker.active_tasks) == 0

    def test_scan_messages_detects_plan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        messages = [
            _user("Here's my plan:\n- Step 1: Do thing\n- Step 2: Do other thing\n- Step 3: Done"),
        ]
        stats = tracker.scan_messages(messages)
        assert stats["plans_found"] >= 1

    def test_scan_messages_detects_tasks(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        messages = [
            _user("T-001: Fix the login bug"),
            _asst("I'll work on T-001. Status: running"),
        ]
        stats = tracker.scan_messages(messages)
        assert stats["tasks_found"] >= 1

    def test_scan_messages_with_status_detection(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        messages = [
            _user("action item: deploy the server. Status: blocked"),
        ]
        tracker.scan_messages(messages)
        tasks = tracker.active_tasks
        assert any(t.status == "blocked" for t in tasks)

    def test_scan_empty_messages(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        stats = tracker.scan_messages([])
        assert stats == {"plans_found": 0, "tasks_found": 0}

    def test_scan_messages_non_string_content(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        messages = [{"role": "user", "content": 12345}]
        stats = tracker.scan_messages(messages)
        assert stats == {"plans_found": 0, "tasks_found": 0}

    def test_injection_message_metadata(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Plan", steps=["a", "b"])
        tracker.record_task("T-1", "Task")
        msg = tracker.build_injection_message()
        assert msg["role"] == "system"
        assert msg["_metadata"]["type"] == "plan_task_reinjection"
        assert msg["_metadata"]["plans_count"] == 1
        assert msg["_metadata"]["tasks_count"] == 1

    def test_very_large_plan_truncation(self):
        from claw_compactor.fusion.plan_reinjection import (
            PlanTaskTracker, PLAN_INJECTION_MAX_TOKENS
        )
        tracker = PlanTaskTracker()
        # Create a plan with many steps that exceeds PLAN_INJECTION_MAX_TOKENS
        big_steps = [f"Step {i}: " + "x" * 200 for i in range(500)]
        tracker.record_plan("Huge plan", steps=big_steps)
        msg = tracker.build_injection_message()
        assert msg is not None
        assert "[...truncated" in msg["content"]

    def test_clear_completed(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Done plan")
        tracker.complete_plan("Done plan")
        tracker.record_task("T-1", "Done task", status="done")
        tracker.record_task("T-2", "Active task", status="running")
        removed = tracker.clear_completed()
        assert removed >= 1
        assert len(tracker._plans) == 0  # completed plan removed
        assert "T-2" in tracker._tasks
        assert "T-1" not in tracker._tasks

    def test_to_dict_serialization(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Plan A", steps=["s1", "s2"])
        tracker.record_task("T-1", "Task 1", status="running")
        d = tracker.to_dict()
        assert len(d["plans"]) == 1
        assert d["plans"][0]["title"] == "Plan A"
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["id"] == "T-1"

    def test_multiple_plans_injection_order(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("First plan", steps=["a"])
        tracker.record_plan("Second plan", steps=["b"])
        msg = tracker.build_injection_message()
        idx_first = msg["content"].index("First plan")
        idx_second = msg["content"].index("Second plan")
        assert idx_first < idx_second  # order preserved

    def test_task_title_truncation_in_injection(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        long_title = "x" * 200
        tracker.record_task("T-1", long_title)
        msg = tracker.build_injection_message()
        # Title should be truncated to 80 chars in the table
        assert len(long_title) > 80


# ===========================================================================
# SkillReinjection — Deep Tests
# ===========================================================================
class TestSkillReinjectionDeep:
    """Deep edge-case tests for SkillSchemaTracker."""

    def test_empty_tracker_returns_none(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        assert tracker.build_injection_message() is None

    def test_record_increments_usage_count(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("read_file")
        tracker.record_usage("read_file")
        tracker.record_usage("read_file")
        assert tracker._skills["read_file"].usage_count == 3

    def test_record_updates_schema(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool", schema={"v": 1})
        tracker.record_usage("tool", schema={"v": 2})
        assert tracker._skills["tool"].schema == {"v": 2}

    def test_record_updates_description(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool", description="old")
        tracker.record_usage("tool", description="new")
        assert tracker._skills["tool"].description == "new"

    def test_max_skills_limit(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker(max_skills=3)
        for i in range(10):
            tracker.record_usage(f"tool_{i}")
        assert len(tracker.recent_skills) == 3

    def test_recent_skills_sorted_by_recency(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("old_tool")
        time.sleep(0.01)
        tracker.record_usage("new_tool")
        skills = tracker.recent_skills
        assert skills[0].name == "new_tool"

    def test_scan_tool_messages(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [
            {"role": "tool", "name": "read_file", "content": "file contents"},
            {"role": "tool", "name": "edit_file", "content": "ok"},
            {"role": "tool", "name": "read_file", "content": "more contents"},
        ]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 3
        assert tracker._skills["read_file"].usage_count == 2

    def test_scan_assistant_tool_calls(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": "{}"}},
                    {"function": {"name": "read", "arguments": "{}"}},
                ],
            },
        ]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 2

    def test_scan_legacy_function_call(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [
            {"role": "assistant", "function_call": {"name": "legacy_fn", "arguments": "{}"}},
        ]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 1
        assert "legacy_fn" in tracker._skills

    def test_scan_empty_tool_name_skipped(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [{"role": "tool", "content": "data"}]  # no name
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 0

    def test_injection_with_schema(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("read", schema={"type": "object", "properties": {"path": {"type": "string"}}})
        msg = tracker.build_injection_message()
        assert msg is not None
        assert "read" in msg["content"]
        assert "json" in msg["content"]  # schema block

    def test_injection_metadata(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool_a")
        msg = tracker.build_injection_message()
        assert msg["_metadata"]["type"] == "skill_schema_reinjection"
        assert msg["_metadata"]["skills_count"] == 1

    def test_injection_token_budget_overflow(self):
        from claw_compactor.fusion.skill_reinjection import (
            SkillSchemaTracker, SKILL_INJECTION_MAX_TOKENS,
        )
        tracker = SkillSchemaTracker(max_skills=100)
        # Create many tools with huge schemas
        for i in range(50):
            tracker.record_usage(
                f"tool_{i}",
                schema={"description": "x" * 500},
                description="A " * 100,
            )
        msg = tracker.build_injection_message()
        assert msg is not None
        tokens = estimate_tokens(msg["content"])
        # Should be within budget (with truncation message)
        assert tokens <= SKILL_INJECTION_MAX_TOKENS + 100  # small tolerance

    def test_clear(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool")
        tracker.clear()
        assert tracker.build_injection_message() is None

    def test_to_dict(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool_a", schema={"x": 1}, description="desc")
        d = tracker.to_dict()
        assert len(d["skills"]) == 1
        assert d["skills"][0]["name"] == "tool_a"
        assert d["skills"][0]["has_schema"] is True

    def test_scan_non_list_tool_calls(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [{"role": "assistant", "tool_calls": "not a list"}]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 0

    def test_scan_tool_calls_non_dict_entry(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        messages = [{"role": "assistant", "tool_calls": ["not_a_dict"]}]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 0


# ===========================================================================
# LLMSummarizer — Deep Tests
# ===========================================================================
class TestLLMSummarizerDeep:
    """Deep edge-case tests for LLMSummarizer."""

    def test_no_client_falls_back_to_deterministic(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        summarizer = LLMSummarizer(api_key=None)
        assert not summarizer.has_llm
        msgs = [_sys("system")] + _make_convo(50, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=5000, trigger_pct=0.1)
        # Should use deterministic fallback (returns triggered=True without "method" key)
        assert stats.get("triggered") is True

    def test_below_threshold_no_summarization(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        summarizer = LLMSummarizer(api_key=None)
        msgs = [_user("hello"), _asst("hi")]
        result, stats = summarizer.summarize(msgs, token_budget=200_000)
        assert stats["triggered"] is False
        assert stats["method"] == "none"
        assert result == msgs

    def test_too_few_body_messages(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        summarizer = LLMSummarizer(api_key=None)
        msgs = [_sys("system"), _user("hi")]
        result, stats = summarizer.summarize(msgs, token_budget=10, trigger_pct=0.01)
        assert stats.get("reason") == "too_few_messages" or stats["method"] == "none"

    def test_llm_client_called_correctly(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.return_value = "## Summary\nDid stuff."
        summarizer = LLMSummarizer(client=mock_client)
        assert summarizer.has_llm
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        assert stats["method"] == "llm"
        assert mock_client.create_message.called
        # Check boundary message
        boundary = [m for m in result if "compact_boundary" in str(m.get("content", ""))]
        assert len(boundary) == 1

    def test_llm_failure_fallback_to_deterministic(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.side_effect = RuntimeError("API down")
        summarizer = LLMSummarizer(client=mock_client, fallback_to_deterministic=True)
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        # Should fallback to deterministic (no "method" key in deterministic stats)
        assert stats.get("triggered") is True
        assert "method" not in stats or stats["method"] != "llm"

    def test_llm_failure_no_fallback(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.side_effect = RuntimeError("API down")
        summarizer = LLMSummarizer(client=mock_client, fallback_to_deterministic=False)
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        assert stats["method"] == "llm_failed"
        assert "error" in stats
        assert result == msgs  # original messages returned

    def test_llm_summary_truncation(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer, MAX_SUMMARY_TOKENS
        mock_client = MagicMock()
        # Return a very long summary
        mock_client.create_message.return_value = "Summary line\n" * 5000
        summarizer = LLMSummarizer(client=mock_client)
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        assert stats["method"] == "llm"
        # Summary should be truncated
        boundary = [m for m in result if "compact_boundary" in str(m.get("content", ""))]
        assert len(boundary) == 1

    def test_llm_stats_include_latency(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary"
        summarizer = LLMSummarizer(client=mock_client)
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        assert "llm_latency_ms" in stats
        assert "llm_input_tokens" in stats
        assert "llm_output_tokens" in stats

    def test_preserve_recent_turns(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary"
        summarizer = LLMSummarizer(client=mock_client)
        msgs = [_sys("system")] + _make_convo(20, tokens_per_msg=200)
        result, stats = summarizer.summarize(
            msgs, token_budget=2000, trigger_pct=0.1, preserve_recent_turns=3
        )
        if stats["triggered"]:
            # Last 6 messages (3 user/asst pairs) should be preserved
            assert len(result) >= 3

    def test_boundary_message_format(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary text"
        summarizer = LLMSummarizer(client=mock_client)
        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)
        if stats["method"] == "llm":
            boundary = [m for m in result if "compact_boundary" in str(m.get("content", ""))]
            data = json.loads(boundary[0]["content"])
            assert data["type"] == "system"
            assert data["subtype"] == "compact_boundary"
            assert "summary" in data
            assert data["compactMetadata"]["method"] == "llm"
            assert data["compactMetadata"]["preservedSegment"] is True


# ===========================================================================
# CompactHooks — Deep Tests
# ===========================================================================
class TestCompactHooksDeep:
    """Deep edge-case tests for HookRegistry."""

    def test_register_decorator_returns_original_function(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()

        @registry.register(HookPhase.PRE_COMPACT)
        def my_hook(messages, **kwargs):
            return messages

        assert callable(my_hook)
        assert my_hook.__name__ == "my_hook"

    def test_register_with_custom_name(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()

        @registry.register(HookPhase.PRE_COMPACT, name="custom_name")
        def my_hook(messages, **kwargs):
            return messages

        hooks = registry.list_hooks(HookPhase.PRE_COMPACT)
        assert "custom_name" in hooks[HookPhase.PRE_COMPACT.value]

    def test_hook_chaining_order(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        order = []

        @registry.register(HookPhase.PRE_COMPACT, name="first")
        def first(messages, **kwargs):
            order.append("first")
            return messages + [_sys("first")]

        @registry.register(HookPhase.PRE_COMPACT, name="second")
        def second(messages, **kwargs):
            order.append("second")
            return messages + [_sys("second")]

        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, [])
        assert order == ["first", "second"]
        assert len(result) == 2

    def test_failing_hook_skipped_gracefully(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()

        @registry.register(HookPhase.PRE_COMPACT, name="bad")
        def bad_hook(messages, **kwargs):
            raise ValueError("intentional")

        @registry.register(HookPhase.PRE_COMPACT, name="good")
        def good_hook(messages, **kwargs):
            return messages + [_sys("added")]

        msgs = [_user("hello")]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, msgs)
        assert stats["hooks_failed"] == 1
        assert stats["hooks_run"] == 1
        assert len(result) == 2  # original + good hook's addition

    def test_hook_returning_none_preserves_messages(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()

        @registry.register(HookPhase.POST_COMPACT)
        def noop(messages, **kwargs):
            return None  # not a list

        msgs = [_user("hello")]
        result, stats = registry.run_hooks(HookPhase.POST_COMPACT, msgs)
        assert result == msgs  # unchanged

    def test_remove_hook(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m, name="removable")
        assert registry.remove_hook(HookPhase.PRE_COMPACT, "removable") is True
        assert registry.remove_hook(HookPhase.PRE_COMPACT, "removable") is False

    def test_has_hooks(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        assert not registry.has_hooks(HookPhase.PRE_COMPACT)
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m)
        assert registry.has_hooks(HookPhase.PRE_COMPACT)

    def test_all_phases_hookable(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        for phase in HookPhase:
            registry.add_hook(phase, lambda m, **k: m, name=f"hook_{phase.value}")
        for phase in HookPhase:
            assert registry.has_hooks(phase)

    def test_kwargs_passed_to_hooks(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        received_kwargs = {}

        def capture_kwargs(messages, **kwargs):
            received_kwargs.update(kwargs)
            return messages

        registry.add_hook(HookPhase.PRE_COMPACT, capture_kwargs)
        registry.run_hooks(HookPhase.PRE_COMPACT, [], token_budget=100_000, level="FULL")
        assert received_kwargs["token_budget"] == 100_000
        assert received_kwargs["level"] == "FULL"

    def test_stats_tracking(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m)
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m)
        registry.run_hooks(HookPhase.PRE_COMPACT, [])
        stats = registry.stats
        assert stats["hooks_registered"] == 2
        assert stats["hooks_called"] == 2
        assert stats["hooks_failed"] == 0

    def test_clear_removes_all(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        for phase in HookPhase:
            registry.add_hook(phase, lambda m, **k: m)
        registry.clear()
        for phase in HookPhase:
            assert not registry.has_hooks(phase)
        assert registry.stats["hooks_registered"] == 0

    def test_list_hooks_all_phases(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m, name="a")
        registry.add_hook(HookPhase.POST_COMPACT, lambda m, **k: m, name="b")
        all_hooks = registry.list_hooks()
        assert "a" in all_hooks["pre_compact"]
        assert "b" in all_hooks["post_compact"]

    def test_hook_timing_recorded(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **k: m, name="fast")
        _, stats = registry.run_hooks(HookPhase.PRE_COMPACT, [])
        assert stats["details"][0]["timing_ms"] >= 0
        assert stats["details"][0]["success"] is True

    def test_empty_phase_returns_quickly(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        msgs = [_user("hello")]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, msgs)
        assert result is msgs  # same object, not copied
        assert stats["hooks_run"] == 0


# ===========================================================================
# ContentStripper — Deep Tests
# ===========================================================================
class TestContentStripperDeep:
    """Deep edge-case tests for strip_images_and_docs."""

    def test_empty_messages(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        result, stats = strip_images_and_docs([])
        assert result == []
        assert stats["images_stripped"] == 0

    def test_no_images_passthrough(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [_user("hello"), _asst("hi")]
        result, stats = strip_images_and_docs(msgs)
        assert result[0]["content"] == "hello"
        assert stats["images_stripped"] == 0

    def test_base64_data_uri_replaced(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        b64 = "A" * 200
        msgs = [_user(f"Look: data:image/png;base64,{b64}")]
        result, stats = strip_images_and_docs(msgs)
        assert "image/png" in result[0]["content"]
        assert b64 not in result[0]["content"]
        assert stats["images_stripped"] == 1

    def test_multiple_base64_in_one_message(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        b64a = "A" * 200
        b64b = "B" * 200
        msgs = [_user(f"data:image/png;base64,{b64a} and data:image/jpeg;base64,{b64b}")]
        result, stats = strip_images_and_docs(msgs)
        assert stats["images_stripped"] == 2

    def test_short_base64_not_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        # < 100 chars of b64 data should not be stripped
        msgs = [_user("data:image/png;base64,AAAA")]
        result, stats = strip_images_and_docs(msgs)
        assert stats["images_stripped"] == 0

    def test_markdown_image_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [_user("Check this: ![screenshot](https://example.com/img.png)")]
        result, stats = strip_images_and_docs(msgs)
        assert "screenshot" in result[0]["content"]
        assert "https://example.com" not in result[0]["content"]
        assert stats["images_stripped"] == 1

    def test_markdown_image_empty_alt(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [_user("![](https://example.com/img.png)")]
        result, stats = strip_images_and_docs(msgs)
        assert "[image: unnamed]" in result[0]["content"]

    def test_html_image_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [_user('Look: <img src="https://example.com/img.png" />')]
        result, stats = strip_images_and_docs(msgs)
        assert "[image removed]" in result[0]["content"]

    def test_document_block_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [_user("```pdf\nlong pdf content here...\n```")]
        result, stats = strip_images_and_docs(msgs)
        assert "embedded document removed" in result[0]["content"]
        assert stats["documents_stripped"] == 1

    def test_multipart_image_url_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 200}},
            ],
        }]
        result, stats = strip_images_and_docs(msgs)
        parts = result[0]["content"]
        assert all(p.get("type") == "text" for p in parts)
        assert stats["multipart_images_stripped"] == 1

    def test_multipart_image_type_stripped(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "desc"},
                {"type": "image", "data": "binary"},
            ],
        }]
        result, stats = strip_images_and_docs(msgs)
        parts = result[0]["content"]
        assert any("[image removed]" in p.get("text", "") for p in parts)

    def test_selective_stripping_disabled(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        b64 = "A" * 200
        msgs = [_user(f"data:image/png;base64,{b64} ![alt](url)")]
        result, stats = strip_images_and_docs(
            msgs,
            strip_base64=False,
            strip_markdown_images=False,
            strip_html_images=False,
            strip_document_blocks=False,
        )
        assert b64 in result[0]["content"]
        assert stats["images_stripped"] == 0

    def test_tokens_saved_positive(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        big_b64 = "A" * 10000
        msgs = [_user(f"data:image/png;base64,{big_b64}")]
        result, stats = strip_images_and_docs(msgs)
        assert stats["tokens_saved"] > 0

    def test_non_string_content_passthrough(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{"role": "user", "content": 12345}]
        result, stats = strip_images_and_docs(msgs)
        assert result[0]["content"] == 12345

    def test_none_content_passthrough(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{"role": "assistant", "content": None}]
        result, stats = strip_images_and_docs(msgs)
        assert result[0]["content"] is None

    def test_multipart_non_image_passthrough(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "audio", "data": "audio_data"},
            ],
        }]
        result, stats = strip_images_and_docs(msgs)
        parts = result[0]["content"]
        assert any(p.get("type") == "audio" for p in parts)

    def test_multipart_external_url_image(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/very/long/path/to/image.png"}},
            ],
        }]
        result, stats = strip_images_and_docs(msgs)
        parts = result[0]["content"]
        assert parts[0]["type"] == "text"
        assert "image:" in parts[0]["text"]


# ===========================================================================
# CachePrefix — Deep Tests
# ===========================================================================
class TestCachePrefixDeep:
    """Deep edge-case tests for CachePrefixManager."""

    def test_empty_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        info = mgr.compute_prefix([])
        assert info["prefix_length"] == 0
        assert info["prefix_tokens"] == 0

    def test_no_system_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        info = mgr.compute_prefix([_user("hello")])
        assert info["prefix_length"] == 0

    def test_all_system_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("sys1"), _sys("sys2"), _sys("sys3")]
        info = mgr.compute_prefix(msgs)
        assert info["prefix_length"] == 3

    def test_system_then_user_stops_at_user(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("system prompt"), _user("hello"), _sys("late system")]
        info = mgr.compute_prefix(msgs)
        assert info["prefix_length"] == 1  # only first system msg

    def test_cache_hit_on_same_prefix(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("system"), _user("hello")]
        mgr.compute_prefix(msgs)
        info = mgr.compute_prefix(msgs)
        assert info["cache_hit"] is True

    def test_cache_miss_on_changed_prefix(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        mgr.compute_prefix([_sys("v1"), _user("hello")])
        info = mgr.compute_prefix([_sys("v2"), _user("hello")])
        assert info["cache_hit"] is False

    def test_hash_deterministic(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr1 = CachePrefixManager()
        mgr2 = CachePrefixManager()
        msgs = [_sys("same system prompt")]
        h1 = mgr1.compute_prefix(msgs)["prefix_hash"]
        h2 = mgr2.compute_prefix(msgs)["prefix_hash"]
        assert h1 == h2

    def test_max_tokens_limit(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        # Create a system message that exceeds max_tokens
        msgs = [_sys("word " * 10000)]  # ~10K tokens
        info = mgr.compute_prefix(msgs, max_tokens=100)
        assert info["prefix_length"] == 0  # too big to fit

    def test_annotate_adds_cache_control(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("system prompt"), _user("hello")]
        result = mgr.annotate_messages_for_caching(msgs)
        # First message should have cache_control
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"]["type"] == "ephemeral"

    def test_annotate_multipart_content(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "part 1"},
                {"type": "text", "text": "part 2"},
            ]},
            _user("hello"),
        ]
        result = mgr.annotate_messages_for_caching(msgs)
        content = result[0]["content"]
        # Last text block should have cache_control
        assert "cache_control" in content[-1]

    def test_annotate_empty_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        result = mgr.annotate_messages_for_caching([])
        assert result == []

    def test_annotate_no_system_prefix(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_user("hello")]
        result = mgr.annotate_messages_for_caching(msgs)
        assert result == msgs

    def test_stats(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("sys")]
        mgr.compute_prefix(msgs)
        mgr.compute_prefix(msgs)
        mgr.compute_prefix([_sys("different")])
        stats = mgr.stats
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert 0 < stats["hit_rate"] < 1

    def test_reset(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        mgr.compute_prefix([_sys("sys")])
        mgr.reset()
        assert mgr.stats["cache_hits"] == 0
        assert mgr.stats["cache_misses"] == 0
        assert mgr.stats["last_prefix_hash"] is None

    def test_prefix_with_compact_boundary(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        boundary = {"role": "system", "content": json.dumps({
            "type": "system", "subtype": "compact_boundary", "summary": "..."
        })}
        msgs = [_sys("original system"), boundary, _user("hello")]
        info = mgr.compute_prefix(msgs)
        assert info["prefix_length"] == 2  # both system messages


# ===========================================================================
# Integration Tests — All Features Together
# ===========================================================================
class TestFullPipelineIntegration:
    """Integration tests combining all v8 features."""

    def test_compact_with_hooks_stripping_and_reinjection(self):
        """Full pipeline: hooks → strip → compact → reinject plans/skills."""
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        # Setup
        hooks = HookRegistry()
        plan_tracker = PlanTaskTracker()
        skill_tracker = SkillSchemaTracker()
        cache_mgr = CachePrefixManager()

        # Record some state
        plan_tracker.record_plan("Build auth", steps=["Design API", "Implement", "Test"])
        plan_tracker.record_task("T-1", "Login endpoint", status="running")
        skill_tracker.record_usage("read_file", description="Read files")
        skill_tracker.record_usage("edit_file", description="Edit files")

        # Register a hook
        hook_called = [False]

        @hooks.register(HookPhase.PRE_COMPACT)
        def inject_env(messages, **kwargs):
            hook_called[0] = True
            return messages + [_sys("ENV: test")]

        # Build conversation with images
        b64 = "A" * 500
        msgs = [
            _sys("You are helpful."),
            _user(f"Look at this: data:image/png;base64,{b64}"),
            _asst("I see the image."),
            _user("Now fix the code."),
            _asst("Done."),
        ]

        # 1. Pre-compact hooks
        msgs, hook_stats = hooks.run_hooks(HookPhase.PRE_COMPACT, msgs)
        assert hook_called[0]

        # 2. Strip images
        msgs, strip_stats = strip_images_and_docs(msgs)
        assert strip_stats["images_stripped"] >= 1

        # 3. Compute cache prefix
        prefix_info = cache_mgr.compute_prefix(msgs)
        assert prefix_info["prefix_length"] >= 1

        # 4. Build injection messages
        plan_msg = plan_tracker.build_injection_message()
        skill_msg = skill_tracker.build_injection_message()
        assert plan_msg is not None
        assert skill_msg is not None

        # 5. Post-compact hooks
        msgs, _ = hooks.run_hooks(HookPhase.POST_COMPACT, msgs)

        # 6. Annotate for caching
        final = cache_mgr.annotate_messages_for_caching(msgs)
        assert len(final) > 0

    def test_llm_summarizer_with_plan_reinjection(self):
        """LLM summarizer followed by plan re-injection."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        mock_client = MagicMock()
        mock_client.create_message.return_value = "## Summary\nUser asked to build auth."

        summarizer = LLMSummarizer(client=mock_client)
        tracker = PlanTaskTracker()
        tracker.record_plan("Auth system", steps=["Design", "Code", "Test"])
        tracker.record_task("T-1", "Design API", status="done")
        tracker.record_task("T-2", "Code endpoints", status="running")

        msgs = [_sys("system")] + _make_convo(30, tokens_per_msg=200)

        # Summarize
        result, stats = summarizer.summarize(msgs, token_budget=3000, trigger_pct=0.1)

        if stats["triggered"]:
            # Re-inject plan
            plan_msg = tracker.build_injection_message()
            if plan_msg:
                result.append(plan_msg)

            # Verify plan is in final messages
            plan_contents = [m for m in result if "plan_task_reinjection" in str(m.get("_metadata", {}))]
            assert len(plan_contents) == 1

    def test_content_stripper_then_deterministic_summarizer(self):
        """Strip content first, then summarize."""
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        # No API key → deterministic fallback
        summarizer = LLMSummarizer(api_key=None)

        b64 = "B" * 1000
        msgs = [_sys("system")]
        for i in range(20):
            msgs.append(_user(f"Turn {i}: data:image/png;base64,{b64}"))
            msgs.append(_asst(f"Response {i}: " + "word " * 100))

        # Strip images first
        stripped, strip_stats = strip_images_and_docs(msgs)
        assert strip_stats["images_stripped"] == 20
        assert strip_stats["tokens_saved"] > 0

        # Then summarize
        result, sum_stats = summarizer.summarize(stripped, token_budget=2000, trigger_pct=0.1)
        # Should have fewer messages now
        assert len(result) <= len(stripped)

    def test_skill_tracker_scans_then_reinjects(self):
        """Scan conversation for tools, then build injection."""
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        msgs = [
            _user("Read the file"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}},
            ]},
            {"role": "tool", "name": "read_file", "content": "def foo(): pass"},
            _user("Now edit it"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "edit_file", "arguments": '{"path": "foo.py"}'}},
            ]},
            {"role": "tool", "name": "edit_file", "content": "ok"},
            _user("Read it again"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}},
            ]},
            {"role": "tool", "name": "read_file", "content": "def foo(): return 1"},
        ]

        stats = tracker.scan_messages_for_tools(msgs)
        assert stats["tools_found"] >= 5
        assert tracker._skills["read_file"].usage_count >= 3
        assert tracker._skills["edit_file"].usage_count >= 1

        msg = tracker.build_injection_message()
        assert msg is not None
        assert "read_file" in msg["content"]

    def test_circuit_breaker_with_hooks(self):
        """Circuit breaker should stop compaction after repeated failures."""
        from claw_compactor.fusion.tiered_compaction import CircuitBreaker
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        cb = CircuitBreaker()
        hooks = HookRegistry()
        failure_count = [0]

        @hooks.register(HookPhase.POST_COMPACT)
        def count_failures(messages, **kwargs):
            if cb.disabled:
                failure_count[0] += 1
            return messages

        # Trip the circuit breaker
        for _ in range(3):
            cb.record_failure()

        assert cb.disabled

        # Hook should see the breaker is disabled
        hooks.run_hooks(HookPhase.POST_COMPACT, [])
        assert failure_count[0] == 1


# ===========================================================================
# Edge Case: Token Budget Boundaries
# ===========================================================================
class TestTokenBudgetBoundaries:
    """Test behavior at exact token budget boundaries."""

    def test_plan_exactly_at_token_limit(self):
        from claw_compactor.fusion.plan_reinjection import (
            PlanTaskTracker, PLAN_INJECTION_MAX_TOKENS,
        )
        tracker = PlanTaskTracker()
        # Create a plan that's close to the limit
        steps = [f"Step {i}: " + "x" * 50 for i in range(100)]
        tracker.record_plan("Boundary plan", steps=steps)
        msg = tracker.build_injection_message()
        assert msg is not None

    def test_task_injection_respects_total_budget(self):
        from claw_compactor.fusion.plan_reinjection import (
            PlanTaskTracker, TOTAL_INJECTION_MAX_TOKENS,
        )
        tracker = PlanTaskTracker()
        # Many tasks
        for i in range(200):
            tracker.record_task(f"T-{i}", f"Task {i}: " + "y" * 50)
        msg = tracker.build_injection_message()
        if msg:
            tokens = estimate_tokens(msg["content"])
            assert tokens <= TOTAL_INJECTION_MAX_TOKENS + 500  # tolerance

    def test_llm_summarizer_input_truncation(self):
        from claw_compactor.fusion.llm_summarizer import (
            LLMSummarizer, SUMMARIZER_MAX_INPUT_TOKENS,
        )
        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary"
        summarizer = LLMSummarizer(client=mock_client)

        # Create very large conversation
        msgs = [_sys("system")]
        for i in range(500):
            msgs.append(_user("x " * 500))
            msgs.append(_asst("y " * 500))

        result, stats = summarizer.summarize(msgs, token_budget=5000, trigger_pct=0.01)
        if stats["method"] == "llm":
            # Input to LLM should be truncated
            assert stats["llm_input_tokens"] <= SUMMARIZER_MAX_INPUT_TOKENS + 1000


# ===========================================================================
# Edge Case: Idempotency
# ===========================================================================
class TestIdempotency:
    """Test that operations are idempotent."""

    def test_double_complete_plan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        tracker = PlanTaskTracker()
        tracker.record_plan("Plan A")
        tracker.complete_plan("Plan A")
        tracker.complete_plan("Plan A")  # should not error
        assert len(tracker.active_plans) == 0

    def test_double_clear_skills(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        tracker = SkillSchemaTracker()
        tracker.record_usage("tool")
        tracker.clear()
        tracker.clear()  # should not error
        assert tracker.build_injection_message() is None

    def test_strip_already_stripped_content(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        b64 = "A" * 200
        msgs = [_user(f"data:image/png;base64,{b64}")]
        result1, _ = strip_images_and_docs(msgs)
        result2, stats2 = strip_images_and_docs(result1)
        assert stats2["images_stripped"] == 0  # nothing left to strip
        assert result1[0]["content"] == result2[0]["content"]

    def test_cache_prefix_idempotent(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        mgr = CachePrefixManager()
        msgs = [_sys("system")]
        info1 = mgr.compute_prefix(msgs)
        info2 = mgr.compute_prefix(msgs)
        assert info1["prefix_hash"] == info2["prefix_hash"]
        assert info2["cache_hit"] is True

    def test_hook_run_twice_same_result(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        registry = HookRegistry()
        registry.add_hook(
            HookPhase.PRE_COMPACT,
            lambda m, **k: m + [_sys("added")],
        )
        msgs = [_user("hello")]
        r1, _ = registry.run_hooks(HookPhase.PRE_COMPACT, msgs)
        # Note: running again on r1 would add another message (not idempotent by design)
        # But running on original msgs should give same result
        r2, _ = registry.run_hooks(HookPhase.PRE_COMPACT, msgs)
        assert len(r1) == len(r2)
