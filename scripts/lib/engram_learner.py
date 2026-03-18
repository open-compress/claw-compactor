"""engram_learner.py — Engram v2: failure learning for claw-compactor.

Scans JSONL session logs, classifies error events into known failure patterns,
and generates compression rules (with evidence thresholds) that can be exported
for insertion into MEMORY.md.

Zero required dependencies beyond the Python 3.9+ standard library.

Part of claw-compactor / Engram layer. License: MIT.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureEvent:
    """A single classified failure extracted from a session log."""

    pattern_name: str       # key from ERROR_PATTERNS
    raw_message: str        # original error text
    source_file: str        # absolute path to the JSONL file
    line_number: int        # 1-based line in the JSONL file
    role: str = "unknown"   # message role if available
    timestamp: str = ""     # ISO timestamp if available


@dataclass(frozen=True)
class CompressionRule:
    """A learnt compression rule derived from repeated failure patterns.

    Only generated when ``evidence_count >= 2``.
    """

    pattern_name: str
    description: str
    evidence_count: int
    example_messages: tuple[str, ...]   # up to 3 representative raw messages
    suggested_annotation: str           # short text for MEMORY.md


# ---------------------------------------------------------------------------
# Error pattern registry
# ---------------------------------------------------------------------------

# Each entry maps a pattern name → list of regex fragments (any match = hit).
# Patterns are compiled once at import time.

_RAW_PATTERNS: dict[str, list[str]] = {
    "FILE_NOT_FOUND": [
        r"No such file or directory",
        r"FileNotFoundError",
        r"ENOENT",
        r"cannot find.*file",
        r"file not found",
    ],
    "MODULE_NOT_FOUND": [
        r"ModuleNotFoundError",
        r"Cannot find module",
        r"No module named",
        r"ImportError.*no module",
        r"module not found",
    ],
    "PERMISSION_DENIED": [
        r"Permission denied",
        r"EACCES",
        r"PermissionError",
        r"Access is denied",
        r"not permitted",
    ],
    "TIMEOUT": [
        r"TimeoutError",
        r"timed out",
        r"ETIMEDOUT",
        r"deadline exceeded",
        r"operation timed out",
    ],
    "BUILD_FAILED": [
        r"Build failed",
        r"compilation error",
        r"make.*Error",
        r"exit code [1-9]",
        r"FAILED.*build",
    ],
    "TEST_FAILED": [
        r"FAILED.*test",
        r"AssertionError",
        r"test.*failed",
        r"pytest.*FAILED",
        r"FAIL.*suite",
    ],
    "SYNTAX_ERROR": [
        r"SyntaxError",
        r"syntax error",
        r"unexpected token",
        r"unexpected.*EOF",
        r"invalid syntax",
    ],
    "TYPE_ERROR": [
        r"TypeError",
        r"type error",
        r"cannot read propert",  # JS: "Cannot read property X of undefined"
        r"is not a function",
        r"unsupported operand type",
    ],
    "IMPORT_ERROR": [
        r"ImportError",
        r"cannot import name",
        r"failed to import",
        r"import.*failed",
        r"unresolved import",
    ],
    "CONNECTION_ERROR": [
        r"ConnectionError",
        r"connection refused",
        r"ECONNREFUSED",
        r"network error",
        r"failed to connect",
    ],
    "AUTH_FAILED": [
        r"401 Unauthorized",
        r"authentication failed",
        r"invalid credentials",
        r"AuthenticationError",
        r"access token.*expired",
    ],
    "RATE_LIMITED": [
        r"429 Too Many Requests",
        r"rate limit",
        r"RateLimitError",
        r"quota exceeded",
        r"too many requests",
    ],
    "OUT_OF_MEMORY": [
        r"MemoryError",
        r"out of memory",
        r"OOM",
        r"Cannot allocate memory",
        r"JavaScript heap out of memory",
    ],
    "DISK_FULL": [
        r"No space left on device",
        r"ENOSPC",
        r"disk full",
        r"DiskFullError",
        r"not enough space",
    ],
}

# Compile all patterns once
ERROR_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    name: [re.compile(frag, re.IGNORECASE) for frag in frags]
    for name, frags in _RAW_PATTERNS.items()
}

# Minimum evidence count required to emit a CompressionRule
_MIN_EVIDENCE = 2

# Max example messages kept per rule
_MAX_EXAMPLES = 3

# Fields in a JSONL line that may carry error text
_TEXT_FIELDS = ("content", "text", "message", "error", "output", "stderr", "stdout")

# Role values that typically contain error information
_ERROR_ROLES = {"assistant", "tool", "system", "error"}


# ---------------------------------------------------------------------------
# EngramLearner
# ---------------------------------------------------------------------------


class EngramLearner:
    """Learn from session failures to generate compression rules.

    Usage::

        learner = EngramLearner()
        failures = learner.scan_session("/path/to/session/dir")
        rules = learner.generate_rules(failures)
        md_block = learner.export_rules(rules)
    """

    # Expose pattern map as a class attribute for introspection / testing
    ERROR_PATTERNS: dict[str, list[re.Pattern[str]]] = ERROR_PATTERNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_session(self, session_dir: str) -> list[FailureEvent]:
        """Scan all JSONL files in *session_dir* and return classified failures.

        Each line of every ``*.jsonl`` file is parsed as a JSON object.  Lines
        that contain error-like text (according to ``ERROR_PATTERNS``) are
        converted to :class:`FailureEvent` instances.

        Args:
            session_dir: Path to the directory containing session JSONL files.

        Returns:
            List of :class:`FailureEvent` objects, in file/line order.
        """
        root = Path(session_dir)
        if not root.exists():
            logger.warning("EngramLearner.scan_session: directory not found: %s", session_dir)
            return []

        events: list[FailureEvent] = []
        for jsonl_path in sorted(root.rglob("*.jsonl")):
            events.extend(self._scan_file(jsonl_path))

        logger.info(
            "EngramLearner: scanned %s, found %d failure events",
            session_dir,
            len(events),
        )
        return events

    def classify_failure(self, event: dict) -> str:
        """Classify a single raw event dict against ERROR_PATTERNS.

        Args:
            event: Arbitrary dict (e.g. a parsed JSONL line).

        Returns:
            The first matching pattern name, or ``"UNKNOWN"`` if none match.
        """
        text = _extract_text(event)
        return self._classify_text(text)

    def generate_rules(self, failures: list[FailureEvent]) -> list[CompressionRule]:
        """Derive compression rules from a list of failure events.

        Only patterns with ``evidence_count >= 2`` produce a rule.

        Args:
            failures: Output of :meth:`scan_session`.

        Returns:
            List of :class:`CompressionRule` objects sorted by evidence_count
            descending (highest evidence first).
        """
        # Bucket failures by pattern name
        buckets: dict[str, list[FailureEvent]] = {}
        for evt in failures:
            buckets.setdefault(evt.pattern_name, []).append(evt)

        rules: list[CompressionRule] = []
        for name, evts in buckets.items():
            count = len(evts)
            if count < _MIN_EVIDENCE:
                logger.debug(
                    "EngramLearner: skipping rule %s (evidence=%d < %d)",
                    name,
                    count,
                    _MIN_EVIDENCE,
                )
                continue

            examples = tuple(e.raw_message[:200] for e in evts[:_MAX_EXAMPLES])
            annotation = _build_annotation(name, count)

            rules.append(
                CompressionRule(
                    pattern_name=name,
                    description=_DESCRIPTIONS.get(name, name),
                    evidence_count=count,
                    example_messages=examples,
                    suggested_annotation=annotation,
                )
            )

        rules.sort(key=lambda r: r.evidence_count, reverse=True)
        return rules

    def export_rules(self, rules: list[CompressionRule]) -> str:
        """Format rules as a Markdown block suitable for insertion into MEMORY.md.

        Args:
            rules: Output of :meth:`generate_rules`.

        Returns:
            A formatted Markdown string.  Empty string if *rules* is empty.
        """
        if not rules:
            return ""

        lines: list[str] = [
            "## Learnt Failure Patterns (Engram v2)",
            "",
            "Auto-generated by EngramLearner. Review before committing.",
            "",
        ]

        for rule in rules:
            lines.append(f"### {rule.pattern_name} (seen {rule.evidence_count}x)")
            lines.append(f"- **Description**: {rule.description}")
            lines.append(f"- **Annotation**: {rule.suggested_annotation}")
            if rule.example_messages:
                lines.append("- **Examples**:")
                for ex in rule.example_messages:
                    # Truncate long examples for readability
                    safe = ex.replace("\n", " ")[:120]
                    lines.append(f"  - `{safe}`")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_file(self, path: Path) -> list[FailureEvent]:
        """Parse a single JSONL file and return failure events."""
        events: list[FailureEvent] = []
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for lineno, raw_line in enumerate(fh, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    event = self._parse_line(raw_line, str(path), lineno)
                    if event is not None:
                        events.append(event)
        except OSError as exc:
            logger.warning("EngramLearner: cannot read %s: %s", path, exc)
        return events

    def _parse_line(
        self, raw_line: str, source_file: str, lineno: int
    ) -> FailureEvent | None:
        """Parse a single JSONL line.  Return a FailureEvent or None."""
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            # Non-JSON line — try treating the raw text as the message
            pattern = self._classify_text(raw_line)
            if pattern == "UNKNOWN":
                return None
            return FailureEvent(
                pattern_name=pattern,
                raw_message=raw_line[:500],
                source_file=source_file,
                line_number=lineno,
            )

        if not isinstance(obj, dict):
            return None

        text = _extract_text(obj)
        if not text:
            return None

        pattern = self._classify_text(text)
        if pattern == "UNKNOWN":
            return None

        role = obj.get("role", "unknown")
        timestamp = obj.get("timestamp", obj.get("ts", ""))

        return FailureEvent(
            pattern_name=pattern,
            raw_message=text[:500],
            source_file=source_file,
            line_number=lineno,
            role=str(role),
            timestamp=str(timestamp),
        )

    def _classify_text(self, text: str) -> str:
        """Return the first matching ERROR_PATTERNS key, or ``"UNKNOWN"``."""
        if not text:
            return "UNKNOWN"
        for name, compiled_patterns in ERROR_PATTERNS.items():
            for pat in compiled_patterns:
                if pat.search(text):
                    return name
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(obj: dict) -> str:
    """Pull the most informative text from a JSONL event dict."""
    parts: list[str] = []
    for field_name in _TEXT_FIELDS:
        val = obj.get(field_name)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            # List of content blocks (Anthropic-style)
            for block in val:
                if isinstance(block, dict):
                    inner = block.get("text", block.get("content", ""))
                    if isinstance(inner, str) and inner.strip():
                        parts.append(inner.strip())
    return " | ".join(parts)


def _build_annotation(pattern_name: str, count: int) -> str:
    """Build a short MEMORY.md annotation for a pattern."""
    desc = _DESCRIPTIONS.get(pattern_name, pattern_name.replace("_", " ").title())
    return (
        f"[{pattern_name}] {desc} occurred {count} time(s) in recent sessions. "
        "Investigate root cause and add mitigation."
    )


_DESCRIPTIONS: dict[str, str] = {
    "FILE_NOT_FOUND": "A required file was missing from the expected path.",
    "MODULE_NOT_FOUND": "A Python/Node module was not installed or importable.",
    "PERMISSION_DENIED": "A file or network operation was blocked by OS permissions.",
    "TIMEOUT": "An operation exceeded its time limit.",
    "BUILD_FAILED": "The build or compilation step exited with an error.",
    "TEST_FAILED": "One or more automated tests failed.",
    "SYNTAX_ERROR": "Source code contained a syntax error.",
    "TYPE_ERROR": "A value had an unexpected or incompatible type.",
    "IMPORT_ERROR": "A module import failed at runtime.",
    "CONNECTION_ERROR": "A network connection could not be established.",
    "AUTH_FAILED": "Authentication or authorisation was rejected.",
    "RATE_LIMITED": "An API or service enforced a rate limit.",
    "OUT_OF_MEMORY": "The process ran out of available memory.",
    "DISK_FULL": "The disk or volume ran out of free space.",
}
