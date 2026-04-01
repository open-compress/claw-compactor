"""PlanReinjection — re-inject active plans and tasks after compaction.

Inspired by Claude Code's plan/task re-injection: when conversation history
is compacted, active plans and incomplete tasks are re-injected into the
context so the LLM doesn't lose track of what it's working on.

Usage::

    from claw_compactor.fusion.plan_reinjection import PlanTaskTracker

    tracker = PlanTaskTracker()
    tracker.record_plan("Build authentication system", steps=[...])
    tracker.record_task("T-001", "Implement login endpoint", status="running")

    # After compaction:
    injection_msg = tracker.build_injection_message()
    messages.append(injection_msg)

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from claw_compactor.tokens import estimate_tokens

# Budget limits.
PLAN_INJECTION_MAX_TOKENS = 10_000
TASK_INJECTION_MAX_TOKENS = 5_000
TOTAL_INJECTION_MAX_TOKENS = 15_000

# Patterns to detect plans and tasks in messages.
_PLAN_PATTERN = re.compile(
    r'(?:plan|roadmap|strategy|approach|steps?)[\s:]+\n((?:\s*[-*\d]+\.?\s+.+\n?)+)',
    re.IGNORECASE | re.MULTILINE,
)
_TASK_PATTERN = re.compile(
    r'(?:T-\d+|task|todo|action item)[\s:]+(.+)',
    re.IGNORECASE,
)
_STATUS_PATTERN = re.compile(
    r'(?:status|state)[\s:]+(\w+)',
    re.IGNORECASE,
)


@dataclass
class PlanItem:
    """A tracked plan with steps."""
    title: str
    steps: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class TaskItem:
    """A tracked task."""
    task_id: str
    title: str
    status: str = "pending"  # pending, running, done, failed, blocked
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class PlanTaskTracker:
    """Tracks active plans and tasks for re-injection after compaction.

    Plans and tasks can be explicitly recorded via the API, or auto-detected
    from conversation messages using heuristic patterns.
    """

    def __init__(self) -> None:
        self._plans: list[PlanItem] = []
        self._tasks: dict[str, TaskItem] = {}
        self._auto_id_counter: int = 0

    # ------------------------------------------------------------------
    # Explicit API
    # ------------------------------------------------------------------

    def record_plan(
        self,
        title: str,
        steps: Optional[list[str]] = None,
    ) -> PlanItem:
        """Record an active plan."""
        plan = PlanItem(title=title, steps=steps or [])
        self._plans.append(plan)
        return plan

    def complete_plan(self, title: str) -> None:
        """Mark a plan as completed."""
        for plan in self._plans:
            if plan.title == title:
                plan.completed = True
                return

    def record_task(
        self,
        task_id: str,
        title: str,
        status: str = "pending",
    ) -> TaskItem:
        """Record or update a task."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.title = title
            task.status = status
            task.updated_at = time.time()
        else:
            task = TaskItem(task_id=task_id, title=title, status=status)
            self._tasks[task_id] = task
        return task

    def update_task_status(self, task_id: str, status: str) -> None:
        """Update a task's status."""
        if task_id in self._tasks:
            self._tasks[task_id].status = status
            self._tasks[task_id].updated_at = time.time()

    # ------------------------------------------------------------------
    # Auto-detection from messages
    # ------------------------------------------------------------------

    def scan_messages(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Scan messages for plans and tasks, auto-recording them.

        Returns stats about what was found.
        """
        plans_found = 0
        tasks_found = 0

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Detect plans.
            for match in _PLAN_PATTERN.finditer(content):
                steps_text = match.group(1)
                steps = [
                    line.strip().lstrip("-*0123456789. ")
                    for line in steps_text.strip().split("\n")
                    if line.strip()
                ]
                if steps:
                    self._auto_id_counter += 1
                    title = f"Auto-detected plan #{self._auto_id_counter}"
                    self.record_plan(title, steps)
                    plans_found += 1

            # Detect tasks.
            for match in _TASK_PATTERN.finditer(content):
                task_title = match.group(1).strip()[:200]
                if task_title:
                    self._auto_id_counter += 1
                    task_id = f"auto-{self._auto_id_counter}"
                    status = "pending"
                    status_match = _STATUS_PATTERN.search(content)
                    if status_match:
                        detected = status_match.group(1).lower()
                        if detected in ("running", "done", "failed", "blocked", "pending"):
                            status = detected
                    self.record_task(task_id, task_title, status)
                    tasks_found += 1

        return {"plans_found": plans_found, "tasks_found": tasks_found}

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    @property
    def active_plans(self) -> list[PlanItem]:
        """Return plans that are not completed."""
        return [p for p in self._plans if not p.completed]

    @property
    def active_tasks(self) -> list[TaskItem]:
        """Return tasks that are not done or cancelled."""
        return [
            t for t in self._tasks.values()
            if t.status not in ("done", "cancelled")
        ]

    def build_injection_message(self) -> Optional[dict[str, Any]]:
        """Build a system message with active plans and tasks for re-injection.

        Returns None if there's nothing to inject.
        """
        parts: list[str] = []
        total_tokens = 0

        # Plans section.
        active_plans = self.active_plans
        if active_plans:
            plan_lines = ["## Active Plans (re-injected after compaction)\n"]
            for plan in active_plans:
                plan_lines.append(f"### {plan.title}")
                for i, step in enumerate(plan.steps, 1):
                    plan_lines.append(f"  {i}. {step}")
                plan_lines.append("")

            plan_text = "\n".join(plan_lines)
            plan_tokens = estimate_tokens(plan_text)
            if plan_tokens <= PLAN_INJECTION_MAX_TOKENS:
                parts.append(plan_text)
                total_tokens += plan_tokens
            else:
                # Truncate steps.
                truncated_lines = plan_lines[:20]
                truncated_lines.append(
                    f"\n[...truncated, {len(plan_lines) - 20} more lines]"
                )
                parts.append("\n".join(truncated_lines))
                total_tokens += estimate_tokens("\n".join(truncated_lines))

        # Tasks section.
        active_tasks = self.active_tasks
        if active_tasks:
            task_lines = ["## Active Tasks (re-injected after compaction)\n"]
            task_lines.append("| ID | Title | Status |")
            task_lines.append("|-----|-------|--------|")
            for task in active_tasks:
                task_lines.append(f"| {task.task_id} | {task.title[:80]} | {task.status} |")
            task_lines.append("")

            task_text = "\n".join(task_lines)
            task_tokens = estimate_tokens(task_text)
            if total_tokens + task_tokens <= TOTAL_INJECTION_MAX_TOKENS:
                parts.append(task_text)
                total_tokens += task_tokens

        if not parts:
            return None

        return {
            "role": "system",
            "content": "\n".join(parts),
            "_metadata": {
                "type": "plan_task_reinjection",
                "plans_count": len(active_plans),
                "tasks_count": len(active_tasks),
                "tokens": total_tokens,
            },
        }

    def clear_completed(self) -> int:
        """Remove completed plans and done tasks. Returns count removed."""
        removed = 0
        self._plans = [p for p in self._plans if not p.completed]
        before = len(self._tasks)
        self._tasks = {
            k: v for k, v in self._tasks.items()
            if v.status not in ("done", "cancelled")
        }
        removed += before - len(self._tasks)
        return removed

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state."""
        return {
            "plans": [
                {"title": p.title, "steps": p.steps, "completed": p.completed}
                for p in self._plans
            ],
            "tasks": [
                {
                    "id": t.task_id,
                    "title": t.title,
                    "status": t.status,
                }
                for t in self._tasks.values()
            ],
        }
