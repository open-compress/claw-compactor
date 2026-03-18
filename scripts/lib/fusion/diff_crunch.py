"""DiffCrunch — git diff compression FusionStage.

Preserves file headers, hunk headers, and all changed lines (+/-).
Compresses context blocks (unchanged lines) to at most 1 line at each end.
Stores large diffs in RewindStore for full retrieval.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re

from lib.fusion.base import FusionStage, FusionContext, FusionResult
from lib.rewind.marker import embed_marker
from lib.rewind.store import RewindStore
from lib.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Context lines to keep at the start/end of each context block.
_CONTEXT_KEEP = 1

# Line count above which we store the original in RewindStore.
_LARGE_DIFF_THRESHOLD = 200

# ---------------------------------------------------------------------------
# Line-type classification
# ---------------------------------------------------------------------------

# File header patterns (unified diff format).
_FILE_HEADER_RE = re.compile(r'^(--- |--- a/|\+\+\+ |\+\+\+ b/|diff --git |index [0-9a-f]+\.\.|new file mode|deleted file mode|rename from |rename to |old mode |new mode )')
_HUNK_HEADER_RE = re.compile(r'^@@')
_ADDED_RE = re.compile(r'^\+(?!\+\+)')       # + lines that are not +++
_REMOVED_RE = re.compile(r'^-(?!--)')         # - lines that are not ---
_NO_NEWLINE_RE = re.compile(r'^\\ No newline')


def _line_type(line: str) -> str:
    """Classify a diff line.

    Returns one of: "file_header" | "hunk_header" | "added" | "removed"
                  | "no_newline" | "context"
    """
    if _FILE_HEADER_RE.match(line):
        return "file_header"
    if _HUNK_HEADER_RE.match(line):
        return "hunk_header"
    if _ADDED_RE.match(line):
        return "added"
    if _REMOVED_RE.match(line):
        return "removed"
    if _NO_NEWLINE_RE.match(line):
        return "no_newline"
    return "context"


# ---------------------------------------------------------------------------
# Context block compression
# ---------------------------------------------------------------------------

def _compress_context_block(block: list[str]) -> list[str]:
    """
    Compress a run of context lines.

    If the block has <= 2*_CONTEXT_KEEP lines: keep all.
    Otherwise: keep first _CONTEXT_KEEP, emit ellipsis, keep last _CONTEXT_KEEP.
    """
    keep = _CONTEXT_KEEP
    if len(block) <= keep * 2:
        return list(block)

    head = block[:keep]
    tail = block[-keep:]
    omitted = len(block) - keep * 2
    ellipsis = f" [... {omitted} unchanged line{'s' if omitted != 1 else ''} ...]"
    return head + [ellipsis] + tail


# ---------------------------------------------------------------------------
# Main compression logic
# ---------------------------------------------------------------------------

def _compress_diff(lines: list[str]) -> list[str]:
    """
    Walk diff lines, preserving structural lines and compressing context blocks.
    """
    output: list[str] = []
    context_buffer: list[str] = []

    def flush_context() -> None:
        if context_buffer:
            output.extend(_compress_context_block(context_buffer))
            context_buffer.clear()

    for line in lines:
        ltype = _line_type(line)

        if ltype == "context":
            context_buffer.append(line)
        else:
            flush_context()
            output.append(line)

    # Flush any trailing context.
    flush_context()
    return output


# ---------------------------------------------------------------------------
# Summary generation (for very large diffs)
# ---------------------------------------------------------------------------

def _summarise_diff(lines: list[str]) -> str:
    """Generate a high-level summary of a large diff."""
    files_changed: list[str] = []
    added_lines = 0
    removed_lines = 0
    hunks = 0

    current_file: str | None = None
    for line in lines:
        ltype = _line_type(line)
        if ltype == "file_header":
            if line.startswith("+++ "):
                path = line[4:].strip()
                # Strip "b/" prefix from git diff output.
                if path.startswith("b/"):
                    path = path[2:]
                if path != "/dev/null":
                    current_file = path
                    if current_file not in files_changed:
                        files_changed.append(current_file)
        elif ltype == "hunk_header":
            hunks += 1
        elif ltype == "added":
            added_lines += 1
        elif ltype == "removed":
            removed_lines += 1

    summary_lines = [
        f"[Large diff summary: {len(files_changed)} file(s) changed, "
        f"+{added_lines} insertions, -{removed_lines} deletions, {hunks} hunk(s)]",
        "Files:",
    ]
    for f in files_changed:
        summary_lines.append(f"  {f}")

    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# FusionStage implementation
# ---------------------------------------------------------------------------

class DiffCrunch(FusionStage):
    """git diff compression — preserves headers and changes, compresses context."""

    name = "diff_crunch"
    order = 18

    def __init__(
        self,
        rewind_store: RewindStore | None = None,
        large_diff_threshold: int = _LARGE_DIFF_THRESHOLD,
        context_keep: int = _CONTEXT_KEEP,
    ) -> None:
        self._rewind_store = rewind_store
        self._large_diff_threshold = large_diff_threshold
        self._context_keep = context_keep

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "diff"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        lines = ctx.content.splitlines()
        original_line_count = len(lines)
        markers: list[str] = []
        warnings: list[str] = []

        is_large = original_line_count > self._large_diff_threshold

        if is_large and self._rewind_store is not None:
            # Store the full original for later retrieval.
            hash_id = self._rewind_store.store(
                original=ctx.content,
                compressed="",  # will be filled in after compression
                original_tokens=original_tokens,
                compressed_tokens=0,
            )
            markers.append(f"diff_crunch:large:hash={hash_id}")

        # Compress the diff.
        compressed_lines = _compress_diff(lines)
        compressed = "\n".join(compressed_lines)

        if is_large:
            summary = _summarise_diff(lines)
            compressed = summary + "\n\n" + compressed
            if self._rewind_store is not None:
                compressed = embed_marker(
                    compressed,
                    original_count=original_line_count,
                    compressed_count=len(compressed_lines),
                    hash_id=hash_id,
                )
            warnings.append(
                f"diff_crunch: large diff ({original_line_count} lines) — summary prepended"
            )

        compressed_tokens = estimate_tokens(compressed)
        markers.insert(0, f"diff_crunch:{original_line_count}->{len(compressed_lines)} lines")

        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
            warnings=warnings,
        )
