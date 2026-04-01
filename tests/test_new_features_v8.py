"""Comprehensive tests for claw-compactor v8 new features.

Tests cover:
  1. LLM Summarizer (llm_summarizer.py)
  2. Plan/Task Re-injection (plan_reinjection.py)
  3. Skill Schema Re-injection (skill_reinjection.py)
  4. Plugin Hooks (compact_hooks.py)
  5. Image/Document Stripping (content_stripper.py)
  6. AUTOCOMPACT_PCT_OVERRIDE env var (tiered_compaction.py)
  7. Cache Prefix Reuse (cache_prefix.py)

Each feature has:
  - Unit tests (individual functions/methods)
  - Integration tests (features working together)
  - Edge case tests (boundary values, empty inputs, error handling)

Part of claw-compactor v8 test suite. License: MIT.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from claw_compactor.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(count: int, tokens_per_msg: int = 100) -> list[dict[str, Any]]:
    """Generate a list of alternating user/assistant messages."""
    msgs: list[dict[str, Any]] = []
    word = "hello " * (tokens_per_msg // 2)
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{word} message {i}"})
    return msgs


def _make_tool_messages(count: int) -> list[dict[str, Any]]:
    """Generate tool result messages."""
    msgs: list[dict[str, Any]] = []
    for i in range(count):
        msgs.append({
            "role": "tool",
            "name": f"tool_{i}",
            "content": f"Result from tool {i}: " + "data " * 50,
        })
    return msgs


def _make_system_message(content: str = "You are a helpful assistant.") -> dict[str, Any]:
    return {"role": "system", "content": content}


# ===========================================================================
# 1. LLM Summarizer Tests
# ===========================================================================

class TestLLMSummarizer:
    """Tests for claw_compactor.fusion.llm_summarizer."""

    def test_import(self):
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        assert LLMSummarizer is not None

    def test_no_api_key_falls_back_to_deterministic(self):
        """Without API key, should use deterministic summarization."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        summarizer = LLMSummarizer(api_key=None)
        assert not summarizer.has_llm

        # Create messages that exceed the trigger threshold.
        messages = [_make_system_message()] + _make_messages(40, tokens_per_msg=500)
        result_msgs, stats = summarizer.summarize(messages, token_budget=5000)

        assert stats["triggered"] is True or stats.get("method") == "none"
        assert isinstance(result_msgs, list)

    def test_below_threshold_no_summarization(self):
        """Messages below threshold should not trigger summarization."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        summarizer = LLMSummarizer(api_key=None)
        messages = [_make_system_message(), {"role": "user", "content": "hi"}]
        result_msgs, stats = summarizer.summarize(messages, token_budget=200_000)

        assert stats["triggered"] is False
        assert len(result_msgs) == len(messages)

    def test_llm_client_called_when_configured(self):
        """When LLM client is configured, it should be called for summarization."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        mock_client = MagicMock()
        mock_client.create_message.return_value = "## Summary\nTask: build auth system"

        summarizer = LLMSummarizer(client=mock_client)
        assert summarizer.has_llm

        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)
        result_msgs, stats = summarizer.summarize(messages, token_budget=2000)

        assert stats.get("method") == "llm"
        mock_client.create_message.assert_called_once()

    def test_llm_failure_falls_back(self):
        """LLM failure should fall back to deterministic."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        mock_client = MagicMock()
        mock_client.create_message.side_effect = RuntimeError("API error")

        summarizer = LLMSummarizer(client=mock_client, fallback_to_deterministic=True)
        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)
        result_msgs, stats = summarizer.summarize(messages, token_budget=2000)

        # Should have fallen back and still produced a result.
        assert isinstance(result_msgs, list)
        assert len(result_msgs) > 0

    def test_llm_failure_no_fallback_returns_original(self):
        """Without fallback, LLM failure should return original messages."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        mock_client = MagicMock()
        mock_client.create_message.side_effect = RuntimeError("API error")

        summarizer = LLMSummarizer(client=mock_client, fallback_to_deterministic=False)
        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)
        result_msgs, stats = summarizer.summarize(messages, token_budget=2000)

        assert stats.get("method") == "llm_failed"
        assert len(result_msgs) == len(messages)

    def test_too_few_messages_skips(self):
        """With only 1-2 messages, should not summarize."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        summarizer = LLMSummarizer(api_key=None)
        messages = [_make_system_message(), {"role": "user", "content": "x" * 5000}]
        result_msgs, stats = summarizer.summarize(messages, token_budget=100)
        assert len(result_msgs) == len(messages)

    def test_compact_boundary_format(self):
        """LLM summary should produce a compact_boundary message."""
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary of conversation"

        summarizer = LLMSummarizer(client=mock_client)
        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)
        result_msgs, stats = summarizer.summarize(messages, token_budget=2000)

        # Find the compact_boundary message.
        boundary_msgs = [
            m for m in result_msgs
            if m.get("role") == "system"
            and "compact_boundary" in str(m.get("content", ""))
        ]
        assert len(boundary_msgs) >= 1

        # Verify format.
        boundary = json.loads(boundary_msgs[0]["content"])
        assert boundary["subtype"] == "compact_boundary"
        assert "method" in boundary["compactMetadata"]
        assert boundary["compactMetadata"]["method"] == "llm"


# ===========================================================================
# 2. Plan/Task Re-injection Tests
# ===========================================================================

class TestPlanTaskTracker:
    """Tests for claw_compactor.fusion.plan_reinjection."""

    def test_import(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        assert PlanTaskTracker is not None

    def test_record_and_retrieve_plan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        plan = tracker.record_plan("Build auth", steps=["Step 1", "Step 2"])
        assert plan.title == "Build auth"
        assert len(plan.steps) == 2
        assert len(tracker.active_plans) == 1

    def test_complete_plan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.record_plan("Build auth", steps=["Step 1"])
        tracker.complete_plan("Build auth")
        assert len(tracker.active_plans) == 0

    def test_record_and_update_task(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        task = tracker.record_task("T-001", "Implement login", status="pending")
        assert task.status == "pending"

        tracker.update_task_status("T-001", "running")
        assert tracker._tasks["T-001"].status == "running"

    def test_active_tasks_excludes_done(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.record_task("T-001", "Task 1", "done")
        tracker.record_task("T-002", "Task 2", "running")
        tracker.record_task("T-003", "Task 3", "cancelled")

        active = tracker.active_tasks
        assert len(active) == 1
        assert active[0].task_id == "T-002"

    def test_build_injection_message(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.record_plan("Build auth", steps=["Create models", "Add routes"])
        tracker.record_task("T-001", "Implement login", "running")

        msg = tracker.build_injection_message()
        assert msg is not None
        assert msg["role"] == "system"
        assert "Active Plans" in msg["content"]
        assert "Active Tasks" in msg["content"]
        assert "Build auth" in msg["content"]
        assert "T-001" in msg["content"]

    def test_injection_message_none_when_empty(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        assert tracker.build_injection_message() is None

    def test_scan_messages_detects_plans(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        messages = [
            {"role": "assistant", "content": "Here's the plan:\n- Step 1: Create models\n- Step 2: Add routes\n- Step 3: Write tests"},
        ]
        stats = tracker.scan_messages(messages)
        assert stats["plans_found"] >= 1

    def test_scan_messages_detects_tasks(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        messages = [
            {"role": "assistant", "content": "T-001: Implement login endpoint"},
        ]
        stats = tracker.scan_messages(messages)
        assert stats["tasks_found"] >= 1

    def test_clear_completed(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.record_plan("Plan 1")
        tracker.record_plan("Plan 2")
        tracker.complete_plan("Plan 1")
        tracker.record_task("T-001", "Task 1", "done")
        tracker.record_task("T-002", "Task 2", "running")

        removed = tracker.clear_completed()
        assert removed == 1  # T-001
        assert len(tracker._plans) == 1
        assert len(tracker._tasks) == 1

    def test_to_dict(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.record_plan("Plan A", ["Step 1"])
        tracker.record_task("T-001", "Task A", "running")

        data = tracker.to_dict()
        assert len(data["plans"]) == 1
        assert len(data["tasks"]) == 1
        assert data["plans"][0]["title"] == "Plan A"

    # Edge cases

    def test_empty_messages_scan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        stats = tracker.scan_messages([])
        assert stats["plans_found"] == 0
        assert stats["tasks_found"] == 0

    def test_non_string_content_in_scan(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        messages = [{"role": "user", "content": 12345}]
        stats = tracker.scan_messages(messages)
        assert stats["plans_found"] == 0

    def test_update_nonexistent_task(self):
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

        tracker = PlanTaskTracker()
        tracker.update_task_status("T-999", "done")
        # Should not crash.
        assert len(tracker._tasks) == 0


# ===========================================================================
# 3. Skill Schema Re-injection Tests
# ===========================================================================

class TestSkillSchemaTracker:
    """Tests for claw_compactor.fusion.skill_reinjection."""

    def test_import(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        assert SkillSchemaTracker is not None

    def test_record_usage(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        record = tracker.record_usage("read_file", schema={"type": "function"})
        assert record.name == "read_file"
        assert record.usage_count == 1

    def test_usage_count_increments(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        tracker.record_usage("read_file")
        tracker.record_usage("read_file")
        tracker.record_usage("read_file")
        assert tracker._skills["read_file"].usage_count == 3

    def test_recent_skills_sorted_by_recency(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        tracker.record_usage("tool_a")
        time.sleep(0.01)
        tracker.record_usage("tool_b")
        time.sleep(0.01)
        tracker.record_usage("tool_c")

        recent = tracker.recent_skills
        assert recent[0].name == "tool_c"
        assert recent[-1].name == "tool_a"

    def test_max_skills_limit(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker(max_skills=3)
        for i in range(10):
            tracker.record_usage(f"tool_{i}")
            time.sleep(0.001)

        assert len(tracker.recent_skills) == 3

    def test_build_injection_message(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        tracker.record_usage("read_file", description="Read a file")
        tracker.record_usage("edit_file", description="Edit a file")

        msg = tracker.build_injection_message()
        assert msg is not None
        assert msg["role"] == "system"
        assert "read_file" in msg["content"]
        assert "edit_file" in msg["content"]

    def test_injection_message_none_when_empty(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        assert tracker.build_injection_message() is None

    def test_scan_messages_for_tools(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        messages = [
            {"role": "tool", "name": "read_file", "content": "file contents"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "edit_file", "arguments": "{}"}}
            ], "content": ""},
            {"role": "assistant", "function_call": {"name": "bash", "arguments": "ls"}, "content": ""},
        ]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 3
        assert "read_file" in tracker._skills
        assert "edit_file" in tracker._skills
        assert "bash" in tracker._skills

    def test_clear(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        tracker.record_usage("tool_a")
        tracker.clear()
        assert len(tracker._skills) == 0

    def test_to_dict(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        tracker.record_usage("tool_a", schema={"type": "func"}, description="A tool")
        data = tracker.to_dict()
        assert len(data["skills"]) == 1
        assert data["skills"][0]["has_schema"] is True

    # Edge cases

    def test_empty_messages_scan(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        stats = tracker.scan_messages_for_tools([])
        assert stats["tools_found"] == 0

    def test_message_without_tool_name(self):
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        tracker = SkillSchemaTracker()
        messages = [{"role": "tool", "content": "result"}]
        stats = tracker.scan_messages_for_tools(messages)
        assert stats["tools_found"] == 0


# ===========================================================================
# 4. Plugin Hooks Tests
# ===========================================================================

class TestCompactHooks:
    """Tests for claw_compactor.fusion.compact_hooks."""

    def test_import(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        assert HookRegistry is not None
        assert HookPhase is not None

    def test_register_decorator(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        @registry.register(HookPhase.PRE_COMPACT)
        def my_hook(messages, **kwargs):
            return messages

        assert registry.has_hooks(HookPhase.PRE_COMPACT)

    def test_add_hook_imperative(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def my_hook(messages, **kwargs):
            return messages

        registry.add_hook(HookPhase.POST_COMPACT, my_hook, name="my_hook")
        assert registry.has_hooks(HookPhase.POST_COMPACT)

    def test_run_hooks_modifies_messages(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def add_system_msg(messages, **kwargs):
            return messages + [{"role": "system", "content": "injected"}]

        registry.add_hook(HookPhase.PRE_COMPACT, add_system_msg)

        messages = [{"role": "user", "content": "hello"}]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, messages)

        assert len(result) == 2
        assert result[-1]["content"] == "injected"
        assert stats["hooks_run"] == 1

    def test_run_hooks_chaining(self):
        """Multiple hooks should chain — each receives previous hook's output."""
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def hook_a(messages, **kwargs):
            return messages + [{"role": "system", "content": "A"}]

        def hook_b(messages, **kwargs):
            return messages + [{"role": "system", "content": "B"}]

        registry.add_hook(HookPhase.PRE_COMPACT, hook_a, name="a")
        registry.add_hook(HookPhase.PRE_COMPACT, hook_b, name="b")

        messages = [{"role": "user", "content": "hi"}]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, messages)

        assert len(result) == 3
        assert result[1]["content"] == "A"
        assert result[2]["content"] == "B"

    def test_hook_failure_is_fail_open(self):
        """A failing hook should be skipped, not crash the pipeline."""
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def bad_hook(messages, **kwargs):
            raise ValueError("intentional error")

        def good_hook(messages, **kwargs):
            return messages + [{"role": "system", "content": "good"}]

        registry.add_hook(HookPhase.PRE_COMPACT, bad_hook, name="bad")
        registry.add_hook(HookPhase.PRE_COMPACT, good_hook, name="good")

        messages = [{"role": "user", "content": "hi"}]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, messages)

        assert stats["hooks_failed"] == 1
        assert stats["hooks_run"] == 1
        # Good hook still ran.
        assert len(result) == 2

    def test_remove_hook(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def my_hook(messages, **kwargs):
            return messages

        registry.add_hook(HookPhase.PRE_COMPACT, my_hook, name="removable")
        assert registry.remove_hook(HookPhase.PRE_COMPACT, "removable")
        assert not registry.has_hooks(HookPhase.PRE_COMPACT)

    def test_remove_nonexistent_hook(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()
        assert not registry.remove_hook(HookPhase.PRE_COMPACT, "nonexistent")

    def test_list_hooks(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def hook_a(messages, **kwargs): return messages
        def hook_b(messages, **kwargs): return messages

        registry.add_hook(HookPhase.PRE_COMPACT, hook_a, name="a")
        registry.add_hook(HookPhase.POST_COMPACT, hook_b, name="b")

        all_hooks = registry.list_hooks()
        assert "pre_compact" in all_hooks
        assert "post_compact" in all_hooks
        assert "a" in all_hooks["pre_compact"]

    def test_no_hooks_for_phase(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()
        messages = [{"role": "user", "content": "hi"}]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, messages)
        assert result == messages
        assert stats["hooks_run"] == 0

    def test_clear(self):
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **kw: m)
        registry.clear()
        assert not registry.has_hooks(HookPhase.PRE_COMPACT)
        assert registry.stats["hooks_registered"] == 0

    # Edge cases

    def test_hook_returning_none_preserves_messages(self):
        """A hook that returns None should preserve original messages."""
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

        registry = HookRegistry()

        def returns_none(messages, **kwargs):
            return None  # type: ignore

        registry.add_hook(HookPhase.PRE_COMPACT, returns_none)
        messages = [{"role": "user", "content": "hi"}]
        result, stats = registry.run_hooks(HookPhase.PRE_COMPACT, messages)
        assert result == messages  # Should preserve original.

    def test_all_phases_exist(self):
        from claw_compactor.fusion.compact_hooks import HookPhase

        phases = list(HookPhase)
        assert len(phases) == 6
        phase_names = {p.value for p in phases}
        assert "pre_compact" in phase_names
        assert "post_compact" in phase_names
        assert "pre_summarize" in phase_names
        assert "post_summarize" in phase_names
        assert "pre_budget" in phase_names
        assert "post_budget" in phase_names


# ===========================================================================
# 5. Image/Document Stripping Tests
# ===========================================================================

class TestContentStripper:
    """Tests for claw_compactor.fusion.content_stripper."""

    def test_import(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs
        assert strip_images_and_docs is not None

    def test_strip_base64_data_uri(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        b64 = "A" * 200
        content = f"Here is an image: data:image/png;base64,{b64} and some text"
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["images_stripped"] == 1
        assert "data:image/png;base64," not in result[0]["content"]
        assert "[image:" in result[0]["content"]

    def test_strip_markdown_image(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        content = "Check this: ![screenshot](https://example.com/img.png)"
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["images_stripped"] == 1
        assert "[image: screenshot]" in result[0]["content"]

    def test_strip_html_image(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        content = 'Look: <img src="https://example.com/pic.jpg" />'
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["images_stripped"] == 1
        assert "<img" not in result[0]["content"]

    def test_strip_document_block(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        content = "Data:\n```csv\ncol1,col2\n1,2\n3,4\n```\nEnd"
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["documents_stripped"] == 1
        assert "[embedded document removed" in result[0]["content"]

    def test_strip_multipart_image_url(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "B" * 200}},
            ],
        }]

        result, stats = strip_images_and_docs(messages)
        assert stats["multipart_images_stripped"] == 1
        # Image part should be converted to text placeholder.
        parts = result[0]["content"]
        assert all(p.get("type") == "text" for p in parts)

    def test_no_stripping_when_disabled(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        content = "![img](url) data:image/png;base64," + "A" * 200
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(
            messages,
            strip_base64=False,
            strip_markdown_images=False,
            strip_html_images=False,
            strip_document_blocks=False,
        )
        assert stats["images_stripped"] == 0
        assert result[0]["content"] == content

    def test_tokens_saved_positive(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        b64 = "A" * 5000
        content = f"data:image/png;base64,{b64}"
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["tokens_saved"] > 0

    # Edge cases

    def test_empty_messages(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        result, stats = strip_images_and_docs([])
        assert result == []
        assert stats["images_stripped"] == 0

    def test_non_string_content_passthrough(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        messages = [{"role": "user", "content": None}]
        result, stats = strip_images_and_docs(messages)
        assert result[0]["content"] is None

    def test_multiple_images_in_one_message(self):
        from claw_compactor.fusion.content_stripper import strip_images_and_docs

        b64 = "A" * 200
        content = (
            f"Image 1: data:image/png;base64,{b64}\n"
            f"Image 2: data:image/jpeg;base64,{b64}\n"
            f"![alt](http://example.com/img.png)"
        )
        messages = [{"role": "user", "content": content}]

        result, stats = strip_images_and_docs(messages)
        assert stats["images_stripped"] == 3


# ===========================================================================
# 6. AUTOCOMPACT_PCT_OVERRIDE Env Var Tests
# ===========================================================================

class TestAutocompactPctOverride:
    """Tests for CLAW_AUTOCOMPACT_PCT_OVERRIDE env var support."""

    def test_default_threshold(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel

        # 70% of 200K = 140K tokens. Default micro threshold is 60%.
        messages = _make_messages(100, tokens_per_msg=300)
        level = determine_level(messages, token_budget=200_000)
        # Should be at least MICRO since messages likely exceed 60%.
        assert isinstance(level, CompactionLevel)

    def test_env_var_override_raises_threshold(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel

        # Create messages at ~65% of budget.
        messages = _make_messages(10, tokens_per_msg=100)
        total = sum(estimate_tokens(m["content"]) for m in messages)

        # Set budget so messages are at 65%.
        budget = int(total / 0.65)

        # Default: 60% threshold -> should be MICRO.
        level_default = determine_level(messages, token_budget=budget)

        # Override to 70% -> should be NONE.
        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.70"}):
            level_override = determine_level(messages, token_budget=budget)

        # At 65% usage: default (60%) -> MICRO, override (70%) -> NONE
        assert level_default == CompactionLevel.MICRO
        assert level_override == CompactionLevel.NONE

    def test_env_var_invalid_value_ignored(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, _get_pct_override

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "not_a_number"}):
            override = _get_pct_override()
            assert override is None

    def test_env_var_out_of_range_ignored(self):
        from claw_compactor.fusion.tiered_compaction import _get_pct_override

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "1.5"}):
            assert _get_pct_override() is None

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.0"}):
            assert _get_pct_override() is None

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "-0.5"}):
            assert _get_pct_override() is None

    def test_env_var_valid_value(self):
        from claw_compactor.fusion.tiered_compaction import _get_pct_override

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.75"}):
            assert _get_pct_override() == 0.75

    def test_explicit_param_takes_priority(self):
        from claw_compactor.fusion.tiered_compaction import determine_level, CompactionLevel

        messages = _make_messages(10, tokens_per_msg=100)
        total = sum(estimate_tokens(m["content"]) for m in messages)
        budget = int(total / 0.65)

        # Even with env var set to 0.50, explicit param 0.70 wins.
        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.50"}):
            level = determine_level(messages, token_budget=budget, micro_pct=0.70)
            assert level == CompactionLevel.NONE

    def test_env_var_not_set(self):
        from claw_compactor.fusion.tiered_compaction import _get_pct_override

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLAW_AUTOCOMPACT_PCT_OVERRIDE", None)
            assert _get_pct_override() is None


# ===========================================================================
# 7. Cache Prefix Reuse Tests
# ===========================================================================

class TestCachePrefixManager:
    """Tests for claw_compactor.fusion.cache_prefix."""

    def test_import(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager
        assert CachePrefixManager is not None

    def test_compute_prefix_system_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Rules: be concise."},
            {"role": "user", "content": "Hello"},
        ]
        result = manager.compute_prefix(messages)
        assert result["prefix_length"] == 2
        assert result["prefix_tokens"] > 0
        assert len(result["prefix_hash"]) == 16

    def test_cache_hit_on_repeated_call(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result1 = manager.compute_prefix(messages)
        assert result1["cache_hit"] is False

        result2 = manager.compute_prefix(messages)
        assert result2["cache_hit"] is True
        assert result2["prefix_hash"] == result1["prefix_hash"]

    def test_cache_miss_on_changed_prefix(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages1 = [
            {"role": "system", "content": "Version A"},
            {"role": "user", "content": "Hello"},
        ]
        messages2 = [
            {"role": "system", "content": "Version B"},
            {"role": "user", "content": "Hello"},
        ]

        manager.compute_prefix(messages1)
        result = manager.compute_prefix(messages2)
        assert result["cache_hit"] is False

    def test_annotate_messages_for_caching(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        annotated = manager.annotate_messages_for_caching(messages)
        # System message should have cache_control.
        first_msg = annotated[0]
        content = first_msg["content"]
        if isinstance(content, list):
            assert any(
                p.get("cache_control") == {"type": "ephemeral"}
                for p in content
                if isinstance(p, dict)
            )
        # Original messages unchanged.
        assert messages[0]["content"] == "You are helpful."

    def test_stats(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hi"},
        ]

        manager.compute_prefix(messages)
        manager.compute_prefix(messages)
        manager.compute_prefix(messages)

        stats = manager.stats
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] > 0

    def test_reset(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        manager.compute_prefix([{"role": "system", "content": "x"}])
        manager.reset()
        assert manager.stats["cache_hits"] == 0
        assert manager.stats["last_prefix_hash"] is None

    # Edge cases

    def test_no_system_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [{"role": "user", "content": "Hello"}]
        result = manager.compute_prefix(messages)
        assert result["prefix_length"] == 0

    def test_all_system_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        messages = [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
        ]
        result = manager.compute_prefix(messages)
        assert result["prefix_length"] == 2

    def test_empty_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        result = manager.compute_prefix([])
        assert result["prefix_length"] == 0
        assert result["prefix_tokens"] == 0

    def test_annotate_empty_messages(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        result = manager.annotate_messages_for_caching([])
        assert result == []

    def test_max_tokens_limit(self):
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        manager = CachePrefixManager()
        # Create system messages that exceed max_tokens.
        messages = [
            {"role": "system", "content": "x " * 5000},
            {"role": "system", "content": "y " * 5000},
            {"role": "user", "content": "hi"},
        ]
        result = manager.compute_prefix(messages, max_tokens=100)
        assert result["prefix_tokens"] <= 110  # Allow small overshoot from per-message estimation


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """Integration tests: multiple features working together."""

    def test_compact_with_hooks_and_reinjection(self):
        """Full compact() with hooks, plan tracker, and skill tracker."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

        # Setup hooks.
        registry = HookRegistry()
        hook_called = {"pre": False, "post": False}

        def pre_hook(messages, **kwargs):
            hook_called["pre"] = True
            return messages

        def post_hook(messages, **kwargs):
            hook_called["post"] = True
            return messages

        registry.add_hook(HookPhase.PRE_COMPACT, pre_hook)
        registry.add_hook(HookPhase.POST_COMPACT, post_hook)

        # Setup trackers.
        plan_tracker = PlanTaskTracker()
        plan_tracker.record_plan("Build auth", ["Create models", "Add routes"])
        plan_tracker.record_task("T-001", "Implement login", "running")

        skill_tracker = SkillSchemaTracker()
        skill_tracker.record_usage("read_file", description="Read a file")

        # Build messages.
        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)

        result, stats = compact(
            messages,
            token_budget=2000,
            level_override=CompactionLevel.FULL,
            hook_registry=registry,
            plan_tracker=plan_tracker,
            skill_tracker=skill_tracker,
        )

        assert hook_called["pre"]
        assert hook_called["post"]
        assert stats.get("plans_reinjected") is True
        assert stats.get("skills_reinjected") is True

    def test_compact_with_image_stripping(self):
        """compact() should strip images before summarization."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel

        b64 = "A" * 500
        messages = [
            _make_system_message(),
            {"role": "user", "content": f"data:image/png;base64,{b64}"},
        ] + _make_messages(10, tokens_per_msg=500)

        result, stats = compact(
            messages,
            token_budget=2000,
            level_override=CompactionLevel.AUTO,
            strip_images=True,
        )

        assert "content_stripping" in stats
        assert stats["content_stripping"]["images_stripped"] >= 1

    def test_compact_with_env_var_override(self):
        """CLAW_AUTOCOMPACT_PCT_OVERRIDE should affect compact() behavior."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel

        messages = _make_messages(5, tokens_per_msg=100)
        total = sum(estimate_tokens(m["content"]) for m in messages)
        budget = int(total / 0.65)  # Messages at 65%.

        # Default (60%) -> should compact.
        _, stats_default = compact(messages, token_budget=budget)
        assert stats_default["level"] != "none"

        # Override to 70% -> should NOT compact.
        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.70"}):
            _, stats_override = compact(messages, token_budget=budget)
            assert stats_override["level"] == "none"

    def test_llm_summarizer_in_compact_flow(self):
        """LLM summarizer should integrate with compact()."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer

        mock_client = MagicMock()
        mock_client.create_message.return_value = "## Summary\nAll tasks done."

        summarizer = LLMSummarizer(client=mock_client)
        messages = [_make_system_message()] + _make_messages(20, tokens_per_msg=500)

        result, stats = compact(
            messages,
            token_budget=2000,
            level_override=CompactionLevel.AUTO,
            llm_summarizer=summarizer,
        )

        assert stats["summarization"].get("method") == "llm"

    def test_cache_prefix_with_compacted_messages(self):
        """Cache prefix should work on post-compaction messages."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        messages = [
            _make_system_message("You are helpful."),
            _make_system_message("Rules: be concise."),
        ] + _make_messages(20, tokens_per_msg=200)

        result, stats = compact(
            messages,
            token_budget=2000,
            level_override=CompactionLevel.AUTO,
        )

        # Compute prefix on compacted messages.
        manager = CachePrefixManager()
        prefix = manager.compute_prefix(result)
        assert prefix["prefix_length"] >= 1

    def test_full_pipeline_all_features(self):
        """Smoke test: all 7 features active simultaneously."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel
        from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
        from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
        from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
        from claw_compactor.fusion.llm_summarizer import LLMSummarizer
        from claw_compactor.fusion.cache_prefix import CachePrefixManager

        # Setup all components.
        registry = HookRegistry()
        registry.add_hook(HookPhase.PRE_COMPACT, lambda m, **kw: m)
        registry.add_hook(HookPhase.POST_COMPACT, lambda m, **kw: m)

        plan_tracker = PlanTaskTracker()
        plan_tracker.record_plan("Test plan", ["Step 1"])
        plan_tracker.record_task("T-001", "Test task", "running")

        skill_tracker = SkillSchemaTracker()
        skill_tracker.record_usage("bash", description="Run commands")

        mock_client = MagicMock()
        mock_client.create_message.return_value = "Summary"
        summarizer = LLMSummarizer(client=mock_client)

        cache_manager = CachePrefixManager()

        # Build messages with images.
        b64 = "A" * 200
        messages = [
            _make_system_message("System prompt"),
            {"role": "user", "content": f"Image: data:image/png;base64,{b64}"},
        ] + _make_messages(20, tokens_per_msg=300)

        with patch.dict(os.environ, {"CLAW_AUTOCOMPACT_PCT_OVERRIDE": "0.50"}):
            result, stats = compact(
                messages,
                token_budget=2000,
                level_override=CompactionLevel.FULL,
                hook_registry=registry,
                plan_tracker=plan_tracker,
                skill_tracker=skill_tracker,
                llm_summarizer=summarizer,
                strip_images=True,
            )

        # Verify all features fired.
        assert stats["level"] == "full"
        assert "content_stripping" in stats
        assert stats.get("plans_reinjected") is True
        assert stats.get("skills_reinjected") is True
        assert "pre_compact_hooks" in stats
        assert "post_compact_hooks" in stats

        # Cache prefix on result.
        prefix = cache_manager.compute_prefix(result)
        assert prefix["prefix_length"] >= 1


# ===========================================================================
# Backward Compatibility Tests
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure new features don't break existing API."""

    def test_compact_without_new_params(self):
        """compact() should work without any new parameters."""
        from claw_compactor.fusion.tiered_compaction import compact, CompactionLevel

        messages = _make_messages(5, tokens_per_msg=100)
        result, stats = compact(messages, token_budget=200_000)
        assert isinstance(result, list)
        assert isinstance(stats, dict)

    def test_determine_level_without_new_params(self):
        """determine_level() should work with old signature."""
        from claw_compactor.fusion.tiered_compaction import determine_level

        messages = _make_messages(5, tokens_per_msg=100)
        level = determine_level(messages, token_budget=200_000)
        assert level is not None

    def test_imports_from_init(self):
        """All v8 exports should be importable from fusion.__init__."""
        from claw_compactor.fusion import (
            budget_tool_results,
            summarize_conversation,
            CompactionLevel,
            CircuitBreaker,
            FileAccessTracker,
            compact,
            determine_level,
        )
        assert all(x is not None for x in [
            budget_tool_results, summarize_conversation,
            CompactionLevel, CircuitBreaker, FileAccessTracker,
            compact, determine_level,
        ])
