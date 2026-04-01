"""SkillReinjection — re-inject recently used skill schemas after compaction.

Inspired by Claude Code's skill schema re-injection: when history is compacted,
recently used tool/skill schemas are re-injected so the LLM retains awareness
of available tools.

Usage::

    from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker

    tracker = SkillSchemaTracker()
    tracker.record_usage("read_file", schema={...})
    tracker.record_usage("edit_file", schema={...})

    # After compaction:
    injection_msg = tracker.build_injection_message()
    messages.append(injection_msg)

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from claw_compactor.tokens import estimate_tokens


# Budget for skill schema injection.
SKILL_INJECTION_MAX_TOKENS = 10_000
MAX_SKILLS_TO_INJECT = 15


@dataclass
class SkillRecord:
    """A tracked skill/tool usage."""
    name: str
    schema: dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    last_used_at: float = field(default_factory=time.time)
    description: str = ""


class SkillSchemaTracker:
    """Tracks recently used skills/tools for re-injection after compaction.

    Skills are ranked by recency and usage frequency. After compaction,
    the most relevant skill schemas are re-injected into the context.
    """

    def __init__(self, max_skills: int = MAX_SKILLS_TO_INJECT) -> None:
        self._skills: dict[str, SkillRecord] = {}
        self._max_skills = max_skills

    def record_usage(
        self,
        name: str,
        schema: Optional[dict[str, Any]] = None,
        description: str = "",
    ) -> SkillRecord:
        """Record a skill/tool usage."""
        if name in self._skills:
            record = self._skills[name]
            record.usage_count += 1
            record.last_used_at = time.time()
            if schema is not None:
                record.schema = schema
            if description:
                record.description = description
        else:
            record = SkillRecord(
                name=name,
                schema=schema or {},
                usage_count=1,
                description=description,
            )
            self._skills[name] = record
        return record

    def scan_messages_for_tools(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Scan messages to auto-detect tool usage patterns.

        Looks for tool-role messages and function_call patterns.
        Returns stats about tools found.
        """
        tools_found = 0
        for msg in messages:
            role = msg.get("role", "")

            # Tool result messages.
            if role == "tool":
                tool_name = msg.get("name", msg.get("tool_call_id", ""))
                if tool_name:
                    self.record_usage(tool_name)
                    tools_found += 1

            # Assistant messages with tool_calls.
            tool_calls = msg.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        if name:
                            self.record_usage(name)
                            tools_found += 1

            # Legacy function_call format.
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                name = fc.get("name", "")
                if name:
                    self.record_usage(name)
                    tools_found += 1

        return {"tools_found": tools_found}

    @property
    def recent_skills(self) -> list[SkillRecord]:
        """Return skills sorted by recency, limited to max_skills."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: (s.last_used_at, s.usage_count),
            reverse=True,
        )
        return sorted_skills[: self._max_skills]

    def build_injection_message(self) -> Optional[dict[str, Any]]:
        """Build a system message with recently used skill schemas.

        Returns None if there are no skills to inject.
        """
        skills = self.recent_skills
        if not skills:
            return None

        lines: list[str] = [
            "## Recently Used Tools (re-injected after compaction)\n"
        ]
        total_tokens = 0

        for skill in skills:
            skill_line = f"- **{skill.name}**"
            if skill.description:
                skill_line += f": {skill.description}"
            skill_line += f" (used {skill.usage_count}x)"

            line_tokens = estimate_tokens(skill_line)
            if total_tokens + line_tokens > SKILL_INJECTION_MAX_TOKENS:
                lines.append(
                    f"\n[...{len(skills) - len(lines) + 1} more tools truncated]"
                )
                break
            lines.append(skill_line)
            total_tokens += line_tokens

            # Include schema if available and within budget.
            if skill.schema:
                import json
                schema_text = json.dumps(skill.schema, indent=2)
                schema_tokens = estimate_tokens(schema_text)
                if total_tokens + schema_tokens <= SKILL_INJECTION_MAX_TOKENS:
                    lines.append(f"  ```json\n  {schema_text}\n  ```")
                    total_tokens += schema_tokens

        if len(lines) <= 1:
            return None

        content = "\n".join(lines)
        return {
            "role": "system",
            "content": content,
            "_metadata": {
                "type": "skill_schema_reinjection",
                "skills_count": len(skills),
                "tokens": estimate_tokens(content),
            },
        }

    def clear(self) -> None:
        """Clear all tracked skills."""
        self._skills.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state."""
        return {
            "skills": [
                {
                    "name": s.name,
                    "usage_count": s.usage_count,
                    "description": s.description,
                    "has_schema": bool(s.schema),
                }
                for s in self.recent_skills
            ],
        }
