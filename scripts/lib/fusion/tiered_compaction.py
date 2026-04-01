"""TieredCompaction — three-level compaction strategy for chat messages.

Inspired by Claude Code's three-tier system (MicroCompact / AutoCompact / Full),
this module provides a unified ``compact()`` entry point that selects the
appropriate compaction level based on context pressure.

Levels
------
  micro  — Fast, no API calls. Tool result truncation only.
           Fires when context is 60-80% full.
  auto   — Medium. Tool result truncation + conversation summarization.
           Fires when context is 80-95% full.
  full   — Aggressive. Everything above + per-message Fusion Pipeline
           compression + file re-injection + plan/task/skill re-injection.
           Fires when context is >95% full.

The module also includes a CircuitBreaker that disables compaction after
MAX_CONSECUTIVE_FAILURES consecutive failures (default 3), preventing
infinite retry loops (the same bug Claude Code discovered wasting 250K
API calls/day globally).

Environment Variables
---------------------
  CLAW_AUTOCOMPACT_PCT_OVERRIDE
      Override the micro compaction threshold (default 0.60).
      Set to a float between 0 and 1, e.g., ``0.70`` to trigger at 70%.

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from claw_compactor.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class CompactionLevel(Enum):
    """Compaction aggressiveness levels."""
    NONE = "none"
    MICRO = "micro"   # Tool result truncation only
    AUTO = "auto"     # + conversation summarization
    FULL = "full"     # + per-message pipeline compression + file re-injection


# Default thresholds (fraction of token_budget).
MICRO_THRESHOLD = 0.60
AUTO_THRESHOLD = 0.80
FULL_THRESHOLD = 0.95

# Circuit breaker: stop after this many consecutive failures.
MAX_CONSECUTIVE_FAILURES = 3

# File re-injection budget (tokens per file, total budget).
FILE_REINJECTION_PER_FILE = 5_000
FILE_REINJECTION_TOTAL = 30_000

# Post-full-compaction target budget.
POST_COMPACT_BUDGET = 50_000

# Environment variable to override compaction trigger percentage.
_ENV_OVERRIDE_KEY = "CLAW_AUTOCOMPACT_PCT_OVERRIDE"


def _get_pct_override() -> Optional[float]:
    """Read CLAW_AUTOCOMPACT_PCT_OVERRIDE from environment."""
    val = os.environ.get(_ENV_OVERRIDE_KEY)
    if val is not None:
        try:
            pct = float(val)
            if 0.0 < pct <= 1.0:
                return pct
            logger.warning(
                "%s must be between 0 and 1, got %s - ignoring",
                _ENV_OVERRIDE_KEY, val,
            )
        except ValueError:
            logger.warning(
                "%s is not a valid float: %s - ignoring",
                _ENV_OVERRIDE_KEY, val,
            )
    return None


@dataclass
class CircuitBreaker:
    """Tracks consecutive compaction failures and disables after threshold."""
    max_failures: int = MAX_CONSECUTIVE_FAILURES
    consecutive_failures: int = 0
    disabled: bool = False
    total_attempts: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_attempts += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_attempts += 1
        if self.consecutive_failures >= self.max_failures:
            self.disabled = True
            logger.warning(
                "CircuitBreaker tripped: %d consecutive failures, "
                "compaction disabled for this session",
                self.consecutive_failures,
            )

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.disabled = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "disabled": self.disabled,
            "consecutive_failures": self.consecutive_failures,
            "total_attempts": self.total_attempts,
            "total_failures": self.total_failures,
        }


@dataclass
class FileAccessTracker:
    """Tracks recently accessed files for re-injection after full compaction."""
    accessed_files: dict[str, str] = field(default_factory=dict)
    # Maps file_path -> content (last seen)
    access_order: list[str] = field(default_factory=list)

    def record_access(self, path: str, content: str) -> None:
        """Record a file access. Most recent access wins."""
        if path in self.accessed_files:
            self.access_order.remove(path)
        self.accessed_files[path] = content
        self.access_order.append(path)

    def get_recent_files(
        self,
        per_file_budget: int = FILE_REINJECTION_PER_FILE,
        total_budget: int = FILE_REINJECTION_TOTAL,
    ) -> list[dict[str, str]]:
        """Return recently accessed files, newest first, within budget."""
        results: list[dict[str, str]] = []
        total_tokens = 0
        # Walk most-recent first.
        for path in reversed(self.access_order):
            content = self.accessed_files.get(path, "")
            tokens = estimate_tokens(content)
            if tokens > per_file_budget:
                # Trim to budget.
                lines = content.split("\n")
                trimmed_lines: list[str] = []
                running = 0
                for line in lines:
                    lt = estimate_tokens(line)
                    if running + lt > per_file_budget:
                        break
                    trimmed_lines.append(line)
                    running += lt
                content = "\n".join(trimmed_lines) + f"\n[...truncated at {per_file_budget} token budget]"
                tokens = estimate_tokens(content)
            if total_tokens + tokens > total_budget:
                break
            results.append({"path": path, "content": content, "tokens": tokens})
            total_tokens += tokens
        return results


def determine_level(
    messages: list[dict[str, Any]],
    token_budget: int = 200_000,
    micro_pct: Optional[float] = None,
    auto_pct: float = AUTO_THRESHOLD,
    full_pct: float = FULL_THRESHOLD,
) -> CompactionLevel:
    """Determine the compaction level needed based on current token usage.

    The micro threshold can be overridden by:
    1. The ``micro_pct`` parameter (highest priority)
    2. The ``CLAW_AUTOCOMPACT_PCT_OVERRIDE`` environment variable
    3. The default ``MICRO_THRESHOLD`` (0.60)
    """
    if micro_pct is None:
        micro_pct = _get_pct_override() or MICRO_THRESHOLD

    total_tokens = _count_message_tokens(messages)
    ratio = total_tokens / token_budget if token_budget > 0 else 0

    if ratio >= full_pct:
        return CompactionLevel.FULL
    elif ratio >= auto_pct:
        return CompactionLevel.AUTO
    elif ratio >= micro_pct:
        return CompactionLevel.MICRO
    return CompactionLevel.NONE


def compact(
    messages: list[dict[str, Any]],
    token_budget: int = 200_000,
    circuit_breaker: Optional[CircuitBreaker] = None,
    file_tracker: Optional[FileAccessTracker] = None,
    fusion_engine: Any = None,
    level_override: Optional[CompactionLevel] = None,
    hook_registry: Any = None,
    plan_tracker: Any = None,
    skill_tracker: Any = None,
    llm_summarizer: Any = None,
    strip_images: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply tiered compaction to a message list.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    token_budget:
        Context window size in tokens.
    circuit_breaker:
        Optional CircuitBreaker instance. If tripped, returns messages unchanged.
    file_tracker:
        Optional FileAccessTracker for file re-injection in FULL mode.
    fusion_engine:
        Optional FusionEngine for per-message compression in FULL mode.
    level_override:
        Force a specific compaction level (for testing).
    hook_registry:
        Optional HookRegistry for pre/post compaction hooks.
    plan_tracker:
        Optional PlanTaskTracker for plan/task re-injection.
    skill_tracker:
        Optional SkillSchemaTracker for skill schema re-injection.
    llm_summarizer:
        Optional LLMSummarizer for API-based summarization.
    strip_images:
        Whether to strip images before summarization (default True).

    Returns
    -------
    (new_messages, stats)
    """
    # Import here to avoid circular imports.
    from claw_compactor.fusion.tool_result_budget import budget_tool_results
    from claw_compactor.fusion.conversation_summarizer import summarize_conversation
    from claw_compactor.fusion.content_stripper import strip_images_and_docs

    if circuit_breaker and circuit_breaker.disabled:
        return messages, {
            "level": "none",
            "reason": "circuit_breaker_disabled",
            "circuit_breaker": circuit_breaker.to_dict(),
        }

    level = level_override or determine_level(messages, token_budget)
    t0 = time.monotonic()

    stats: dict[str, Any] = {
        "level": level.value,
        "tokens_before": _count_message_tokens(messages),
    }

    try:
        result_messages = list(messages)

        # --- Pre-compact hooks ---
        if hook_registry is not None:
            from claw_compactor.fusion.compact_hooks import HookPhase
            result_messages, hook_stats = hook_registry.run_hooks(
                HookPhase.PRE_COMPACT, result_messages,
                token_budget=token_budget, level=level,
            )
            stats["pre_compact_hooks"] = hook_stats

        # Level 1: Micro — tool result truncation.
        if level in (CompactionLevel.MICRO, CompactionLevel.AUTO, CompactionLevel.FULL):
            if hook_registry is not None:
                result_messages, _ = hook_registry.run_hooks(
                    HookPhase.PRE_BUDGET, result_messages,
                )
            result_messages, tool_stats = budget_tool_results(result_messages)
            stats["tool_budget"] = tool_stats
            if hook_registry is not None:
                result_messages, _ = hook_registry.run_hooks(
                    HookPhase.POST_BUDGET, result_messages,
                )

        # Level 2: Auto — image stripping + conversation summarization.
        if level in (CompactionLevel.AUTO, CompactionLevel.FULL):
            # Strip images/docs before summarization.
            if strip_images:
                result_messages, strip_stats = strip_images_and_docs(result_messages)
                stats["content_stripping"] = strip_stats

            if hook_registry is not None:
                result_messages, _ = hook_registry.run_hooks(
                    HookPhase.PRE_SUMMARIZE, result_messages,
                )

            # Use LLM summarizer if available, otherwise deterministic.
            if llm_summarizer is not None:
                result_messages, summ_stats = llm_summarizer.summarize(
                    result_messages, token_budget=token_budget
                )
            else:
                result_messages, summ_stats = summarize_conversation(
                    result_messages, token_budget=token_budget
                )
            stats["summarization"] = summ_stats

            if hook_registry is not None:
                result_messages, _ = hook_registry.run_hooks(
                    HookPhase.POST_SUMMARIZE, result_messages,
                )

        # Level 3: Full — per-message Fusion Pipeline compression.
        if level == CompactionLevel.FULL:
            if fusion_engine is not None:
                result_messages, fusion_stats = _apply_fusion_compression(
                    result_messages, fusion_engine
                )
                stats["fusion"] = fusion_stats

            # File re-injection.
            if file_tracker:
                recent_files = file_tracker.get_recent_files()
                if recent_files:
                    injection_msg = _build_file_injection_message(recent_files)
                    result_messages.append(injection_msg)
                    stats["files_reinjected"] = len(recent_files)

            # Plan/task re-injection.
            if plan_tracker is not None:
                plan_msg = plan_tracker.build_injection_message()
                if plan_msg is not None:
                    result_messages.append(plan_msg)
                    stats["plans_reinjected"] = True

            # Skill schema re-injection.
            if skill_tracker is not None:
                skill_msg = skill_tracker.build_injection_message()
                if skill_msg is not None:
                    result_messages.append(skill_msg)
                    stats["skills_reinjected"] = True

        # --- Post-compact hooks ---
        if hook_registry is not None:
            result_messages, hook_stats = hook_registry.run_hooks(
                HookPhase.POST_COMPACT, result_messages,
                token_budget=token_budget, level=level,
            )
            stats["post_compact_hooks"] = hook_stats

        stats["tokens_after"] = _count_message_tokens(result_messages)
        stats["tokens_saved"] = stats["tokens_before"] - stats["tokens_after"]
        stats["timing_ms"] = round((time.monotonic() - t0) * 1000, 2)

        if circuit_breaker:
            circuit_breaker.record_success()
            stats["circuit_breaker"] = circuit_breaker.to_dict()

        return result_messages, stats

    except Exception as exc:
        logger.error("Compaction failed at level %s: %s", level.value, exc)
        if circuit_breaker:
            circuit_breaker.record_failure()
            stats["circuit_breaker"] = circuit_breaker.to_dict()
        stats["error"] = str(exc)
        stats["tokens_after"] = stats.get("tokens_before", 0)
        stats["tokens_saved"] = 0
        return messages, stats


def _count_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
        else:
            total += estimate_tokens(str(content))
    return total


def _apply_fusion_compression(
    messages: list[dict[str, Any]],
    fusion_engine: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compress each non-system message through the Fusion Pipeline."""
    result = fusion_engine.compress_messages(messages)
    stats = {
        "original_tokens": result["stats"]["original_tokens"],
        "compressed_tokens": result["stats"]["compressed_tokens"],
        "reduction_pct": result["stats"]["reduction_pct"],
    }
    return result["messages"], stats


def _build_file_injection_message(files: list[dict[str, str]]) -> dict[str, Any]:
    """Build a system message with recently accessed file contents."""
    parts: list[str] = ["## Recently Accessed Files (re-injected after compaction)\n"]
    for f in files:
        parts.append(f"### {f['path']} ({f['tokens']} tokens)")
        parts.append(f"```\n{f['content']}\n```\n")
    return {
        "role": "system",
        "content": "\n".join(parts),
    }
