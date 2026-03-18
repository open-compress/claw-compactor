"""StructuralCollapse — repeated structural pattern compressor FusionStage.

Detects and compresses repeated structural patterns in text, providing large
token savings on:
  - Import blocks (Python, JS/TS, Java)
  - Repeated assertions or similar test lines
  - Repeated log entries with only parameter differences
  - Config/env variable listings

Algorithm summary:
  1. Import Collapse: consecutive import lines → [imports: a,b,c,...]
  2. Repeated Line Collapse: 3+ lines sharing a template → first + summary + last
  3. Short-circuit: passes through content unchanged when no patterns found

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult
from claw_compactor.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum line count before the stage bothers scanning.
_MIN_LINES = 10

# Minimum consecutive repeated lines before collapsing.
_MIN_REPEAT = 3

# Maximum number of import names to list inline before truncating.
_MAX_IMPORT_NAMES_INLINE = 20

# ---------------------------------------------------------------------------
# Import detection regexes
# ---------------------------------------------------------------------------

# Python: "import foo", "import foo as bar", "from foo import bar, baz"
_PY_IMPORT_RE = re.compile(
    r'^\s*(?:from\s+\S+\s+import\s+.+|import\s+.+)$'
)

# JavaScript / TypeScript:
#   import x from 'y'
#   import { x } from 'y'
#   import * as x from 'y'
#   const x = require('y')
_JS_IMPORT_RE = re.compile(
    r"""^\s*(?:import\s+.+\s+from\s+['"].+['"]|"""
    r"""(?:const|let|var)\s+\S+\s*=\s*require\s*\(\s*['"].+['"]\s*\))"""
)

# Java / Kotlin: "import com.example.Foo;" or "import com.example.*;"
# Must contain a dot in the qualified name to distinguish from Python's
# bare "import os".  The trailing semicolon is optional (Kotlin omits it).
_JAVA_IMPORT_RE = re.compile(r'^\s*import\s+\w[\w]*(?:\.\w[\w]*)+(?:\.\*)?;?\s*$')


# ---------------------------------------------------------------------------
# Template extraction for repeated-line detection
# ---------------------------------------------------------------------------

# Patterns whose matched content is considered "variable" across similar lines.
# Order matters: more specific patterns first.
_VARIABLE_PARTS = [
    # Quoted strings (single or double)
    re.compile(r'"[^"]*"'),
    re.compile(r"'[^']*'"),
    # Numbers (int, float)
    re.compile(r'\b\d+(?:\.\d+)?\b'),
    # Identifiers inside brackets: result["key"] → result[<VAR>]
    re.compile(r'\[([^\[\]]+)\]'),
    # Snake_case identifiers ending with a short variable suffix:
    #   word_x, word_y, word_1, word_abc  (suffix up to 4 chars)
    # This handles patterns like expected_x / expected_y / item_01 / item_02.
    re.compile(r'\b([A-Za-z_]\w*)_([A-Za-z0-9]{1,4})\b'),
]

# Compiled substitution pattern that replaces variable parts with a fixed
# placeholder so identical-structure lines share the same template.
_PLACEHOLDER = '<VAR>'


def _extract_template(line: str) -> str:
    """Return a structural template for *line* by replacing varying parts.

    Variable parts (strings, numbers, bracket-contents) become '<VAR>'.
    The resulting template captures the stable structural skeleton.
    """
    t = line
    for pattern in _VARIABLE_PARTS:
        t = pattern.sub(_PLACEHOLDER, t)
    # Collapse multiple adjacent placeholders into one to avoid template
    # proliferation from e.g. result["a"]["b"].
    t = re.sub(r'(<VAR>\s*,?\s*)+', _PLACEHOLDER, t)
    return t


# ---------------------------------------------------------------------------
# Import name extraction helpers
# ---------------------------------------------------------------------------

def _py_import_names(line: str) -> list[str]:
    """Extract imported symbol names from a Python import line."""
    stripped = line.strip()
    # "from X import a, b, c"
    m = re.match(r'from\s+\S+\s+import\s+(.+)', stripped)
    if m:
        raw = m.group(1)
        # Handle "from X import (a, b, c)" with optional parens
        raw = raw.strip('()')
        return [n.strip().split(' as ')[0].strip() for n in raw.split(',') if n.strip()]
    # "import a, b as B, c"
    m2 = re.match(r'import\s+(.+)', stripped)
    if m2:
        return [n.strip().split(' as ')[0].strip() for n in m2.group(1).split(',') if n.strip()]
    return []


def _js_import_names(line: str) -> list[str]:
    """Extract imported symbol names from a JS/TS import/require line."""
    stripped = line.strip()
    # import { a, b, c } from '...'
    m = re.search(r'\{\s*([^}]+)\}', stripped)
    if m:
        return [n.strip().split(' as ')[0].strip() for n in m.group(1).split(',') if n.strip()]
    # import x from '...' — default import
    m2 = re.match(r'import\s+(\w+)\s+from', stripped)
    if m2:
        return [m2.group(1)]
    # import * as x from '...'
    m3 = re.match(r'import\s+\*\s+as\s+(\w+)\s+from', stripped)
    if m3:
        return [f'*{m3.group(1)}']
    # const x = require('...')
    m4 = re.match(r'(?:const|let|var)\s+(\w+)\s*=\s*require', stripped)
    if m4:
        return [m4.group(1)]
    return []


def _java_import_names(line: str) -> list[str]:
    """Extract the simple class name from a Java import line."""
    stripped = line.strip().rstrip(';')
    m = re.match(r'import\s+([\w.]+(?:\.\*)?)', stripped)
    if m:
        parts = m.group(1).split('.')
        return [parts[-1]]
    return []


# ---------------------------------------------------------------------------
# Import language detector
# ---------------------------------------------------------------------------

def _detect_import_language(line: str) -> str | None:
    """Return 'python', 'js', or 'java' if line is an import, else None.

    Java is checked before Python because Java import lines like
    'import java.util.List;' also match the Python import regex.
    """
    if _JAVA_IMPORT_RE.match(line):
        return 'java'
    if _JS_IMPORT_RE.match(line):
        return 'js'
    if _PY_IMPORT_RE.match(line):
        return 'python'
    return None


def _extract_names(line: str, lang: str) -> list[str]:
    """Extract imported names from *line* given the detected *lang*."""
    if lang == 'python':
        return _py_import_names(line)
    if lang == 'js':
        return _js_import_names(line)
    if lang == 'java':
        return _java_import_names(line)
    return []


# ---------------------------------------------------------------------------
# Import block collapsing
# ---------------------------------------------------------------------------

class _ImportBlock(NamedTuple):
    start: int   # index of first import line (in the working list)
    end: int     # index of last import line (inclusive)
    lang: str
    names: list[str]


def _find_import_blocks(lines: list[str]) -> list[_ImportBlock]:
    """Scan *lines* and return a list of consecutive import blocks.

    A block is 3+ consecutive import lines of the same language family.
    """
    blocks: list[_ImportBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        lang = _detect_import_language(lines[i])
        if lang is None:
            i += 1
            continue
        # Start of a potential block
        j = i
        names: list[str] = []
        block_lang = lang
        while j < n:
            l = _detect_import_language(lines[j])
            if l != block_lang:
                break
            names.extend(_extract_names(lines[j], block_lang))
            j += 1
        count = j - i
        if count >= _MIN_REPEAT:
            blocks.append(_ImportBlock(start=i, end=j - 1, lang=block_lang, names=names))
        i = j  # advance past the entire run regardless
    return blocks


def _format_import_summary(block: _ImportBlock) -> str:
    """Format an import block as a compact inline summary."""
    names = block.names
    if len(names) > _MAX_IMPORT_NAMES_INLINE:
        shown = names[:_MAX_IMPORT_NAMES_INLINE]
        rest = len(names) - _MAX_IMPORT_NAMES_INLINE
        name_str = ','.join(shown) + f',+{rest}more'
    else:
        name_str = ','.join(names) if names else '...'
    return f'[imports: {name_str}]'


# ---------------------------------------------------------------------------
# Repeated line collapsing
# ---------------------------------------------------------------------------

class _RepeatedRun(NamedTuple):
    start: int   # index in lines
    end: int     # inclusive
    template: str
    count: int


def _find_repeated_runs(lines: list[str]) -> list[_RepeatedRun]:
    """Find runs of 3+ consecutive lines sharing the same structural template."""
    runs: list[_RepeatedRun] = []
    n = len(lines)
    i = 0
    while i < n:
        template = _extract_template(lines[i])
        # Skip very short templates (< 8 chars) — too generic to be meaningful
        if len(template.strip()) < 8:
            i += 1
            continue
        j = i + 1
        while j < n and _extract_template(lines[j]) == template:
            j += 1
        count = j - i
        if count >= _MIN_REPEAT:
            runs.append(_RepeatedRun(start=i, end=j - 1, template=template, count=count))
        i = j
    return runs


def _format_repeated_summary(run: _RepeatedRun) -> str:
    """Format a repeated-line run as a compact summary."""
    middle = run.count - 2  # first and last are kept verbatim
    return f'[... {middle} similar line{"s" if middle != 1 else ""} ...]'


# ---------------------------------------------------------------------------
# Core collapse engine
# ---------------------------------------------------------------------------

class _CollapseStats(NamedTuple):
    import_blocks_collapsed: int
    repeated_runs_collapsed: int
    lines_before: int
    lines_after: int


def _apply_collapse(lines: list[str]) -> tuple[list[str], _CollapseStats]:
    """Apply import and repeated-line collapse to *lines*.

    Returns the new line list and statistics about what was collapsed.
    Strategy:
      1. Find import blocks.
      2. Find repeated runs that do NOT overlap with any import block.
      3. Build output by iterating lines and applying substitutions in order.
    """
    import_blocks = _find_import_blocks(lines)

    # Build a set of line indices covered by import blocks so we skip them
    # when searching for generic repeated runs.
    import_covered: set[int] = set()
    for blk in import_blocks:
        for idx in range(blk.start, blk.end + 1):
            import_covered.add(idx)

    # Find repeated runs only on non-import lines — create a filtered view.
    non_import_lines = [
        (idx, line) for idx, line in enumerate(lines)
        if idx not in import_covered
    ]
    # We need the repeated run detector to work on contiguous sequences, so we
    # only apply it to gaps between import blocks.  The simplest correct
    # approach: collect contiguous ranges of non-import indices.
    repeated_runs_by_orig_idx: dict[int, _RepeatedRun] = {}

    # Build contiguous segments of non-import lines and scan each.
    segments: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for item in non_import_lines:
        if current and item[0] != current[-1][0] + 1:
            segments.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        segments.append(current)

    for seg in segments:
        seg_lines = [line for _, line in seg]
        seg_runs = _find_repeated_runs(seg_lines)
        for run in seg_runs:
            # Translate segment-local indices back to original line indices
            orig_start = seg[run.start][0]
            orig_end = seg[run.end][0]
            orig_run = _RepeatedRun(
                start=orig_start,
                end=orig_end,
                template=run.template,
                count=run.count,
            )
            # Mark all indices in this run
            for idx in range(orig_start, orig_end + 1):
                repeated_runs_by_orig_idx[idx] = orig_run

    # ----------------------------------------------------------------
    # Build output
    # ----------------------------------------------------------------
    output: list[str] = []
    i = 0
    n = len(lines)
    import_blocks_collapsed = 0
    repeated_runs_collapsed = 0

    # Index import blocks by start for O(1) lookup
    import_block_by_start: dict[int, _ImportBlock] = {blk.start: blk for blk in import_blocks}
    # Track which repeated runs we've already emitted (by start index)
    emitted_runs: set[int] = set()

    while i < n:
        # Check for import block starting here
        if i in import_block_by_start:
            blk = import_block_by_start[i]
            output.append(_format_import_summary(blk))
            import_blocks_collapsed += 1
            i = blk.end + 1
            continue

        # Check for repeated run starting or continuing here
        if i in repeated_runs_by_orig_idx:
            run = repeated_runs_by_orig_idx[i]
            if run.start == i and i not in emitted_runs:
                # Emit first line, summary, last line
                output.append(lines[run.start])
                output.append(_format_repeated_summary(run))
                output.append(lines[run.end])
                emitted_runs.add(i)
                repeated_runs_collapsed += 1
                i = run.end + 1
                continue
            elif i in emitted_runs or run.start != i:
                # Interior line of an already-emitted run — skip
                i += 1
                continue

        output.append(lines[i])
        i += 1

    stats = _CollapseStats(
        import_blocks_collapsed=import_blocks_collapsed,
        repeated_runs_collapsed=repeated_runs_collapsed,
        lines_before=len(lines),
        lines_after=len(output),
    )
    return output, stats


# ---------------------------------------------------------------------------
# FusionStage implementation
# ---------------------------------------------------------------------------

class StructuralCollapse(FusionStage):
    """Detect and compress repeated structural patterns in text or code.

    Handles:
      - Import blocks (Python, JS/TS, Java): collapsed to [imports: a,b,c]
      - Repeated lines (3+) sharing a structural template: first + summary + last

    order = 20: runs after Cortex (5) and SemanticDedup (12), before Neurosyntax (25).
    """

    name = "structural_collapse"
    order = 20

    def should_apply(self, ctx: FusionContext) -> bool:
        """Apply to code and text content_types with more than 10 lines."""
        if ctx.content_type not in ("code", "text"):
            return False
        return ctx.content.count('\n') >= _MIN_LINES

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        lines = ctx.content.splitlines()

        output_lines, stats = _apply_collapse(lines)

        # Preserve trailing newline if original had one
        compressed = '\n'.join(output_lines)
        if ctx.content.endswith('\n') and not compressed.endswith('\n'):
            compressed += '\n'

        compressed_tokens = estimate_tokens(compressed)

        markers: list[str] = []
        if stats.import_blocks_collapsed:
            markers.append(
                f"structural_collapse:imports:{stats.import_blocks_collapsed} block(s) collapsed"
            )
        if stats.repeated_runs_collapsed:
            markers.append(
                f"structural_collapse:repeated:{stats.repeated_runs_collapsed} run(s) collapsed"
            )
        if stats.lines_before != stats.lines_after:
            markers.append(
                f"structural_collapse:lines:{stats.lines_before}->{stats.lines_after}"
            )

        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )
