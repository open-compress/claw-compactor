"""LogCrunch — Build/test log compression FusionStage.

Preserves ERROR/WARN/FATAL lines, stack traces, and failure-related lines.
Compresses repeated INFO/DEBUG lines to occurrence summaries.
Normalises timestamps to relative deltas.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from lib.fusion.base import FusionStage, FusionContext, FusionResult
from lib.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Standard log level prefix: optional timestamp, optional logger name, level.
_LEVEL_RE = re.compile(
    r'(?i)\b(ERROR|ERR|FATAL|CRITICAL|WARN(?:ING)?|INFO|DEBUG|TRACE|VERBOSE)\b'
)

_ERROR_LEVEL_RE = re.compile(r'(?i)\b(ERROR|ERR|FATAL|CRITICAL)\b')
_WARN_LEVEL_RE = re.compile(r'(?i)\bWARN(?:ING)?\b')
_INFO_DEBUG_RE = re.compile(r'(?i)\b(INFO|DEBUG|TRACE|VERBOSE)\b')

# Lines that always matter regardless of log level.
_IMPORTANT_CONTENT_RE = re.compile(
    r'(?i)(failed|failure|exception|error|assert|panic|abort|traceback|caused by)',
)

# Stack-trace indicators: indented lines or common stack frame patterns.
_STACK_INDENT_RE = re.compile(r'^(\s{2,}|\t)')
_STACK_FRAME_RE = re.compile(
    r'(?:'
    r'^\s+at\s+'                  # Java/JS: "  at ..."
    r'|^\s+File\s+"'              # Python: '  File "...'
    r'|^\s+in\s+\w'               # Go: '  in funcName'
    r'|\bTraceback\b'             # Python: 'Traceback (most recent call last):'
    r'|\bgoroutine\s+\d+\b'       # Go goroutine dump
    r')',
    re.IGNORECASE,
)

# Common timestamp formats — we capture the group so we can normalise.
_TIMESTAMP_RE = re.compile(
    r'(?:'
    r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'  # ISO 8601
    r'|\d{2}:\d{2}:\d{2}(?:\.\d+)?'      # HH:MM:SS
    r'|\d{10,13}'                          # Unix epoch (seconds or ms)
    r')'
)

# How many trailing non-important INFO/DEBUG lines to keep as "context".
_TAIL_CONTEXT = 2
# Minimum repetition count before we collapse.
_MIN_REPEAT = 3


class _LineInfo(NamedTuple):
    raw: str
    important: bool  # must always be preserved
    in_trace: bool   # part of a stack-trace block
    level: str       # "error" | "warn" | "info_debug" | "other"
    norm: str        # normalised version of the line (timestamps replaced)


def _classify_line(line: str) -> _LineInfo:
    """Classify a single log line."""
    level_match = _LEVEL_RE.search(line)
    level_str = level_match.group(1).upper() if level_match else ""

    is_error = bool(_ERROR_LEVEL_RE.search(line))
    is_warn = bool(_WARN_LEVEL_RE.search(line))
    is_info_debug = bool(_INFO_DEBUG_RE.search(line)) and not is_error and not is_warn
    is_important_content = bool(_IMPORTANT_CONTENT_RE.search(line))
    is_stack = bool(_STACK_FRAME_RE.search(line))

    important = is_error or is_warn or is_important_content or is_stack

    if is_error:
        level = "error"
    elif is_warn:
        level = "warn"
    elif is_info_debug:
        level = "info_debug"
    else:
        level = "other"

    norm = _TIMESTAMP_RE.sub("<TS>", line)

    return _LineInfo(
        raw=line,
        important=important,
        in_trace=is_stack,
        level=level,
        norm=norm,
    )


def _is_stack_continuation(line: str) -> bool:
    """Return True if this line looks like it belongs inside a stack trace."""
    return bool(_STACK_FRAME_RE.search(line) or _STACK_INDENT_RE.match(line))


def _normalise_timestamps(lines: list[str]) -> list[str]:
    """Replace absolute timestamps with relative deltas (+Xs) where possible."""
    # We do a best-effort pass: find the first ISO timestamp and use it as t0.
    first_ts: float | None = None
    result: list[str] = []

    for line in lines:
        m = re.search(
            r'(\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)',
            line,
        )
        if m:
            try:
                h, mn, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
                frac = float("0." + m.group(5)) if m.group(5) else 0.0
                ts = h * 3600 + mn * 60 + s + frac
                if first_ts is None:
                    first_ts = ts
                delta = ts - first_ts
                new_line = line[: m.start()] + f"[+{delta:.3f}s]" + line[m.end():]
                result.append(new_line)
                continue
            except (ValueError, IndexError):
                pass
        result.append(line)

    return result


def _compress_log(lines: list[str]) -> list[str]:
    """
    Core compression logic:
    - Always keep important lines (error/warn/important content/stack traces).
    - Collapse runs of repeated info/debug lines.
    - Keep first + last occurrence of repeated patterns.
    """
    classified = [_classify_line(ln) for ln in lines]
    output: list[str] = []

    # Track whether we are inside a stack-trace block.
    in_trace = False
    trace_buffer: list[str] = []

    # Track runs of info/debug lines with the same normalised form.
    run_norm: str | None = None
    run_lines: list[str] = []

    def flush_run() -> None:
        nonlocal run_norm, run_lines
        if not run_lines:
            return
        if len(run_lines) >= _MIN_REPEAT:
            output.append(run_lines[0])
            output.append(f"[... repeated {len(run_lines) - 2} more times ...]")
            output.append(run_lines[-1])
        else:
            output.extend(run_lines)
        run_norm = None
        run_lines = []

    def flush_trace() -> None:
        nonlocal in_trace, trace_buffer
        output.extend(trace_buffer)
        in_trace = False
        trace_buffer = []

    i = 0
    while i < len(classified):
        info = classified[i]
        line = info.raw

        # Detect start of stack trace block.
        if not in_trace and _STACK_FRAME_RE.search(line):
            flush_run()
            in_trace = True
            trace_buffer = [line]
            i += 1
            # Collect continuation lines.
            while i < len(classified):
                next_info = classified[i]
                if _is_stack_continuation(next_info.raw) or next_info.in_trace:
                    trace_buffer.append(next_info.raw)
                    i += 1
                else:
                    break
            flush_trace()
            continue

        # Important line — always keep.
        if info.important:
            flush_run()
            output.append(line)
            i += 1
            continue

        # Info/debug line — try to collapse repetitions.
        if info.level == "info_debug":
            if info.norm == run_norm:
                run_lines.append(line)
            else:
                flush_run()
                run_norm = info.norm
                run_lines = [line]
            i += 1
            continue

        # Other lines (no level detected): keep them but break any run.
        flush_run()
        output.append(line)
        i += 1

    flush_run()
    return output


class LogCrunch(FusionStage):
    """Build/test log compression. Preserves errors, warnings and stack traces."""

    name = "log_crunch"
    order = 16

    def __init__(self, normalise_timestamps: bool = True) -> None:
        self._normalise_timestamps = normalise_timestamps

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "log"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        lines = ctx.content.splitlines()

        if self._normalise_timestamps:
            lines = _normalise_timestamps(lines)

        compressed_lines = _compress_log(lines)
        compressed = "\n".join(compressed_lines)
        compressed_tokens = estimate_tokens(compressed)

        original_count = len(lines)
        compressed_count = len(compressed_lines)
        markers = [f"log_crunch:{original_count}->{compressed_count} lines"]

        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )
