"""SearchCrunch — grep/ripgrep output compression FusionStage.

Parses "file:line:content" search output, groups by file, deduplicates
identical matches, merges consecutive line numbers into ranges, and
truncates to top-N files by match count when the result set is large.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from lib.fusion.base import FusionStage, FusionContext, FusionResult
from lib.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum number of files to retain; the rest are summarised.
_MAX_FILES = 20

# Maximum matches to show per file before truncating.
_MAX_MATCHES_PER_FILE = 50

# Matches the canonical "file:line:content" format produced by grep/rg.
# We also tolerate Windows paths like "C:\path\to\file:10:content".
_GREP_LINE_RE = re.compile(
    r'^(?P<file>(?:[A-Za-z]:[\\/]|/|\.[\\/])?[^\x00:]+?)'
    r':(?P<line>\d+)'
    r':(?P<content>.*)$'
)

# Lines that look like binary-match notifications or separator lines.
_SEPARATOR_RE = re.compile(r'^--$')
_BINARY_RE = re.compile(r'Binary file .+ matches')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Match:
    line_no: int
    content: str


@dataclass
class _FileMatches:
    path: str
    matches: list[_Match] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_grep_output(text: str) -> tuple[dict[str, _FileMatches], list[str]]:
    """
    Parse grep/rg output into per-file match collections.

    Returns:
        (file_map, unparsed_lines) where file_map maps path -> _FileMatches
        and unparsed_lines holds lines that did not match the grep format.
    """
    file_map: dict[str, _FileMatches] = {}
    unparsed: list[str] = []

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if _SEPARATOR_RE.match(raw_line):
            continue
        if _BINARY_RE.match(raw_line):
            unparsed.append(raw_line)
            continue

        m = _GREP_LINE_RE.match(raw_line)
        if m:
            path = m.group("file")
            line_no = int(m.group("line"))
            content = m.group("content")
            if path not in file_map:
                file_map[path] = _FileMatches(path=path)
            file_map[path].matches.append(_Match(line_no=line_no, content=content))
        else:
            unparsed.append(raw_line)

    return file_map, unparsed


def _dedup_matches(matches: list[_Match]) -> list[_Match]:
    """Remove matches with identical content on the same line number."""
    seen: set[tuple[int, str]] = set()
    result: list[_Match] = []
    for m in matches:
        key = (m.line_no, m.content.strip())
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _merge_consecutive(matches: list[_Match]) -> list[str]:
    """
    Merge consecutive or adjacent line numbers into range strings.

    Returns a list of formatted strings like:
      "  L10: content"
      "  L12-15: [4 lines]"
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: m.line_no)
    output: list[str] = []

    i = 0
    while i < len(sorted_matches):
        start = sorted_matches[i]
        j = i + 1
        # Extend run while line numbers are consecutive.
        while j < len(sorted_matches) and sorted_matches[j].line_no == sorted_matches[j - 1].line_no + 1:
            j += 1

        run = sorted_matches[i:j]
        if len(run) == 1:
            output.append(f"  L{start.line_no}: {start.content}")
        elif len(run) == 2:
            # Two lines — show both individually; the range marker adds no value.
            for r in run:
                output.append(f"  L{r.line_no}: {r.content}")
        else:
            first_content = run[0].content
            last_content = run[-1].content
            output.append(f"  L{run[0].line_no}: {first_content}")
            output.append(f"  L{run[0].line_no + 1}-{run[-1].line_no - 1}: [{len(run) - 2} lines omitted]")
            output.append(f"  L{run[-1].line_no}: {last_content}")

        i = j

    return output


def _format_file_section(fm: _FileMatches, max_matches: int) -> list[str]:
    """Format a single file's matches into output lines."""
    deduped = _dedup_matches(fm.matches)
    total = len(deduped)

    truncated = deduped[:max_matches]
    lines = _merge_consecutive(truncated)

    section: list[str] = [f"{fm.path} ({total} match{'es' if total != 1 else ''}):"]
    section.extend(lines)
    if total > max_matches:
        section.append(f"  ... [{total - max_matches} more matches not shown]")
    return section


# ---------------------------------------------------------------------------
# FusionStage implementation
# ---------------------------------------------------------------------------

class SearchCrunch(FusionStage):
    """grep/ripgrep search result compression."""

    name = "search_crunch"
    order = 17

    def __init__(
        self,
        max_files: int = _MAX_FILES,
        max_matches_per_file: int = _MAX_MATCHES_PER_FILE,
    ) -> None:
        self._max_files = max_files
        self._max_matches_per_file = max_matches_per_file

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "search"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        file_map, unparsed = _parse_grep_output(ctx.content)

        if not file_map:
            # Nothing parseable — return as-is.
            return FusionResult(
                content=ctx.content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                skipped=True,
                warnings=["search_crunch: no grep-format lines found"],
            )

        # Sort files by descending match count, then alphabetically.
        sorted_files = sorted(
            file_map.values(),
            key=lambda fm: (-len(fm.matches), fm.path),
        )

        total_files = len(sorted_files)
        omitted_files = max(0, total_files - self._max_files)
        top_files = sorted_files[: self._max_files]

        output_sections: list[str] = []

        # Summary header.
        total_matches = sum(len(fm.matches) for fm in sorted_files)
        output_sections.append(
            f"Search results: {total_matches} matches across {total_files} file{'s' if total_files != 1 else ''}"
        )
        if omitted_files:
            output_sections.append(
                f"[Showing top {self._max_files} of {total_files} files by match count]"
            )

        for fm in top_files:
            output_sections.append("")
            output_sections.extend(_format_file_section(fm, self._max_matches_per_file))

        if omitted_files:
            omitted_names = [fm.path for fm in sorted_files[self._max_files:]]
            output_sections.append("")
            output_sections.append(
                f"[{omitted_files} additional file{'s' if omitted_files != 1 else ''} omitted: "
                + ", ".join(omitted_names[:5])
                + (" ..." if len(omitted_names) > 5 else "")
                + "]"
            )

        if unparsed:
            output_sections.append("")
            output_sections.append(f"[{len(unparsed)} non-grep line(s):]")
            output_sections.extend(f"  {ln}" for ln in unparsed[:10])
            if len(unparsed) > 10:
                output_sections.append(f"  ... [{len(unparsed) - 10} more]")

        compressed = "\n".join(output_sections)
        compressed_tokens = estimate_tokens(compressed)

        markers = [f"search_crunch:{total_files} files, {total_matches} matches"]
        if omitted_files:
            markers.append(f"search_crunch:omitted {omitted_files} files")

        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )
