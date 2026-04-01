"""CompactHooks — plugin hooks for pre/post compaction.

Inspired by Claude Code's plugin architecture: allows external code to
register callbacks that run before and after compaction. Use cases:
  - Inject custom context before compaction (e.g., environment state)
  - Log or audit compaction events
  - Modify messages before they enter the compaction pipeline
  - Post-process compacted messages (e.g., add watermarks)

Usage::

    from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase

    registry = HookRegistry()

    @registry.register(HookPhase.PRE_COMPACT)
    def inject_env_context(messages, **kwargs):
        messages.append({"role": "system", "content": "ENV: production"})
        return messages

    @registry.register(HookPhase.POST_COMPACT)
    def log_compaction(messages, **kwargs):
        print(f"Compacted to {len(messages)} messages")
        return messages

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HookPhase(Enum):
    """Phases where hooks can be registered."""
    PRE_COMPACT = "pre_compact"       # Before any compaction
    POST_COMPACT = "post_compact"     # After all compaction
    PRE_SUMMARIZE = "pre_summarize"   # Before conversation summarization
    POST_SUMMARIZE = "post_summarize" # After conversation summarization
    PRE_BUDGET = "pre_budget"         # Before tool result budgeting
    POST_BUDGET = "post_budget"       # After tool result budgeting


# Type for hook callbacks.
# Signature: (messages, **kwargs) -> messages
HookCallback = Callable[..., list[dict[str, Any]]]


class HookRegistry:
    """Registry for compaction lifecycle hooks.

    Hooks are called in registration order within each phase. Each hook
    receives the current message list and must return a (possibly modified)
    message list.

    If a hook raises an exception, it is logged and skipped (fail-open).
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPhase, list[tuple[str, HookCallback]]] = {
            phase: [] for phase in HookPhase
        }
        self._stats: dict[str, Any] = {
            "hooks_registered": 0,
            "hooks_called": 0,
            "hooks_failed": 0,
        }

    def register(
        self,
        phase: HookPhase,
        name: Optional[str] = None,
    ) -> Callable[[HookCallback], HookCallback]:
        """Decorator to register a hook for a specific phase.

        Parameters
        ----------
        phase:
            The compaction phase to hook into.
        name:
            Optional human-readable name for the hook (for logging).

        Returns
        -------
        Decorator that registers the function and returns it unchanged.
        """
        def decorator(func: HookCallback) -> HookCallback:
            hook_name = name or func.__name__
            self._hooks[phase].append((hook_name, func))
            self._stats["hooks_registered"] += 1
            return func
        return decorator

    def add_hook(
        self,
        phase: HookPhase,
        callback: HookCallback,
        name: Optional[str] = None,
    ) -> None:
        """Imperatively add a hook (non-decorator API)."""
        hook_name = name or getattr(callback, "__name__", "anonymous")
        self._hooks[phase].append((hook_name, callback))
        self._stats["hooks_registered"] += 1

    def remove_hook(self, phase: HookPhase, name: str) -> bool:
        """Remove a hook by name. Returns True if found and removed."""
        hooks = self._hooks[phase]
        for i, (hook_name, _) in enumerate(hooks):
            if hook_name == name:
                hooks.pop(i)
                return True
        return False

    def run_hooks(
        self,
        phase: HookPhase,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run all hooks for a phase, passing messages through each.

        Parameters
        ----------
        phase:
            The compaction phase.
        messages:
            Current message list.
        **kwargs:
            Additional context (e.g., token_budget, level, stats).

        Returns
        -------
        (modified_messages, hook_stats)
        """
        hooks = self._hooks.get(phase, [])
        if not hooks:
            return messages, {"phase": phase.value, "hooks_run": 0}

        hook_stats: dict[str, Any] = {
            "phase": phase.value,
            "hooks_run": 0,
            "hooks_failed": 0,
            "details": [],
        }

        current_messages = messages
        for hook_name, callback in hooks:
            t0 = time.monotonic()
            try:
                result = callback(current_messages, **kwargs)
                if isinstance(result, list):
                    current_messages = result
                elapsed_ms = (time.monotonic() - t0) * 1000
                hook_stats["details"].append({
                    "name": hook_name,
                    "success": True,
                    "timing_ms": round(elapsed_ms, 2),
                })
                hook_stats["hooks_run"] += 1
                self._stats["hooks_called"] += 1
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Hook %s (%s) failed: %s",
                    hook_name, phase.value, exc,
                )
                hook_stats["details"].append({
                    "name": hook_name,
                    "success": False,
                    "error": str(exc),
                    "timing_ms": round(elapsed_ms, 2),
                })
                hook_stats["hooks_failed"] += 1
                self._stats["hooks_failed"] += 1
                # Fail-open: continue with unmodified messages.

        return current_messages, hook_stats

    def has_hooks(self, phase: HookPhase) -> bool:
        """Check if any hooks are registered for a phase."""
        return bool(self._hooks.get(phase))

    def list_hooks(self, phase: Optional[HookPhase] = None) -> dict[str, list[str]]:
        """List registered hook names, optionally filtered by phase."""
        if phase is not None:
            return {phase.value: [name for name, _ in self._hooks.get(phase, [])]}
        return {
            p.value: [name for name, _ in hooks]
            for p, hooks in self._hooks.items()
            if hooks
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Return hook execution statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Remove all registered hooks."""
        for phase in HookPhase:
            self._hooks[phase] = []
        self._stats = {
            "hooks_registered": 0,
            "hooks_called": 0,
            "hooks_failed": 0,
        }
