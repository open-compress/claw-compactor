"""Neurosyntax — AST-aware code compression FusionStage.

Uses tree-sitter for multi-language AST parsing when available; falls back to
safe regex-based compression that strips comments and normalizes whitespace
without touching code semantics.

Critical safety rule: identifier names are NEVER shortened.  Class names,
function names, and variable names are semantic anchors that LLMs use to
understand code context.  Shortening them destroys comprehension and causes
downstream task failures (validated on SWE-bench).

Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby,
PHP, Swift, Kotlin, Scala, Bash, R, Perl.

Part of claw-compactor v7. License: MIT.
"""
from __future__ import annotations

import re
from typing import Any

from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult
from claw_compactor.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Optional tree-sitter import
# ---------------------------------------------------------------------------
_TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter_language_pack as tslp  # type: ignore[import]
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Comment patterns per language family
# ---------------------------------------------------------------------------
_HASH_COMMENT_LANGS = {"python", "ruby", "bash", "sh", "perl", "r"}
_SLASH_COMMENT_LANGS = {"javascript", "typescript", "java", "go", "rust", "c", "cpp", "csharp", "kotlin", "swift"}

# Matches a full-line Python/Ruby/shell comment (optional leading whitespace + #)
_HASH_COMMENT_RE = re.compile(r"^\s*#")
# Matches a full-line C-family comment (optional leading whitespace + //)
_SLASH_COMMENT_RE = re.compile(r"^\s*//")
# Matches a full-line block-comment opener or closer  /* ... */
_BLOCK_OPEN_RE = re.compile(r"^\s*/\*")
_BLOCK_CLOSE_RE = re.compile(r"\*/\s*$")

# Annotations that must be preserved even inside comment lines
_IMPORTANT_COMMENT_RE = re.compile(
    r"type:\s*ignore|noqa|pragma|TODO|FIXME|HACK|NOTE"
    r"|eslint-disable|@ts-ignore|@ts-expect-error",
    re.IGNORECASE,
)

# Python triple-quote docstring openers
_TRIPLE_DOUBLE_RE = re.compile(r'^\s*(""")')
_TRIPLE_SINGLE_RE = re.compile(r"^\s*(''')")

# Python import lines
_IMPORT_RE = re.compile(r"^\s*(import |from \S+ import )")


class Neurosyntax(FusionStage):
    """AST-aware code compression. Uses tree-sitter when available, regex fallback otherwise."""

    name = "neurosyntax"
    order = 25  # after Cortex(5), before dictionary/dedup stages

    SUPPORTED_LANGS = {"python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"}

    def __init__(self) -> None:
        self._tree_sitter_available = _TREE_SITTER_AVAILABLE

    # ------------------------------------------------------------------
    # FusionStage interface
    # ------------------------------------------------------------------

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "code"

    def apply(self, ctx: FusionContext) -> FusionResult:
        language = ctx.language
        original_tokens = estimate_tokens(ctx.content)

        if self._tree_sitter_available and language in self.SUPPORTED_LANGS:
            compressed = self._ast_compress(ctx.content, language)
        else:
            compressed = self._fallback_compress(ctx.content, language)

        compressed_tokens = estimate_tokens(compressed)
        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
        )

    # ------------------------------------------------------------------
    # Regex fallback (primary path — tree-sitter is optional)
    # ------------------------------------------------------------------

    def _fallback_compress(self, text: str, language: str | None) -> str:
        """Safe regex-based code compression. No identifier shortening."""
        lines = text.split("\n")
        result: list[str] = []
        in_block_comment = False
        in_docstring = False
        docstring_quote: str | None = None
        docstring_first_content: str | None = None
        is_python = (language == "python")

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # ---- Block comment tracking (C-family) ----
            if not is_python and not in_block_comment and _BLOCK_OPEN_RE.match(line):
                in_block_comment = True
                if self._is_important_comment(line):
                    result.append(line.rstrip())
                # else: skip the opening line entirely
                if "*/" in line:
                    in_block_comment = False  # single-line /* ... */
                i += 1
                continue

            if in_block_comment:
                if self._is_important_comment(line):
                    result.append(line.rstrip())
                if _BLOCK_CLOSE_RE.search(line):
                    in_block_comment = False
                i += 1
                continue

            # ---- Python docstring collapsing ----
            if is_python and not in_docstring:
                quote = self._docstring_opener(stripped)
                if quote is not None:
                    # Check if it closes on the same line (after the opener)
                    rest = stripped[len(quote):]
                    if rest.endswith(quote) and len(rest) >= len(quote):
                        # Single-line docstring — keep as-is
                        result.append(line.rstrip())
                        i += 1
                        continue
                    # Multi-line docstring: record first content line
                    first_content = rest.strip()
                    in_docstring = True
                    docstring_quote = quote
                    docstring_first_content = first_content
                    indent = len(line) - len(line.lstrip())
                    # Emit collapsed single-line version once we know the content
                    # We'll finalize when we hit the closing quote
                    i += 1
                    # Collect until closing quote
                    closing_found = False
                    while i < len(lines):
                        dl = lines[i]
                        ds = dl.strip()
                        if docstring_quote in ds:
                            closing_found = True
                            in_docstring = False
                            # emit collapsed form
                            preview = docstring_first_content or ds.replace(docstring_quote, "").strip()
                            if preview:
                                result.append(" " * indent + quote + preview + " " + quote)
                            i += 1
                            break
                        if not docstring_first_content:
                            docstring_first_content = ds
                        i += 1
                    if not closing_found:
                        in_docstring = False
                    continue

            # ---- Pure comment lines ----
            if self._is_pure_comment(line, language):
                if self._is_important_comment(line):
                    result.append(line.rstrip())
                # else: drop
                i += 1
                continue

            # ---- Blank line deduplication ----
            if not stripped:
                if result and not result[-1].strip():
                    i += 1
                    continue  # skip consecutive blanks
                result.append("")
                i += 1
                continue

            # ---- Trailing whitespace strip ----
            result.append(line.rstrip())
            i += 1

        return "\n".join(result)

    # ------------------------------------------------------------------
    # Tree-sitter AST path (optional)
    # ------------------------------------------------------------------

    def _ast_compress(self, text: str, language: str) -> str:
        """AST-aware compression using tree-sitter."""
        try:
            parser = tslp.get_parser(language)
            tree = parser.parse(text.encode())
            root = tree.root_node
            lines = text.split("\n")
            keep_ranges = self._collect_keep_ranges(root, language)
            return self._reconstruct(lines, keep_ranges)
        except Exception:  # noqa: BLE001 — graceful fallback
            return self._fallback_compress(text, language)

    def _collect_keep_ranges(self, root: Any, language: str) -> list[tuple[int, int]]:
        """Walk the AST and return (start_line, end_line) ranges to keep (0-indexed, inclusive)."""
        keep: list[tuple[int, int]] = []
        self._walk(root, keep, language)
        return sorted(set_merge(keep))

    def _walk(self, node: Any, keep: list[tuple[int, int]], language: str) -> None:
        """Recursively walk tree-sitter nodes and collect keep ranges."""
        node_type = node.type

        # Always keep: import statements, top-level declarations, type annotations
        if node_type in {
            "import_statement", "import_from_statement",  # Python
            "import_declaration", "import_specifier",     # JS/TS
            "use_declaration",                            # Rust
            "package_declaration", "import_declaration",  # Java/Go
        }:
            keep.append((node.start_point[0], node.end_point[0]))
            return

        # Always keep: function / method / class signatures (first line only for bodies)
        if node_type in {
            "function_definition", "function_declaration", "method_definition",
            "class_definition", "class_declaration",
            "decorated_definition",  # Python decorators + def/class
        }:
            sig_end = node.start_point[0]
            # Keep decorator lines too
            keep.append((node.start_point[0], sig_end))
            # Walk children to keep signature parts and returns; compress body
            for child in node.children:
                if child.type == "block" or child.type == "statement_block":
                    self._compress_body(child, keep)
                else:
                    keep.append((child.start_point[0], child.end_point[0]))
            return

        # Always keep: error handling
        if node_type in {
            "try_statement", "except_clause", "finally_clause",
            "catch_clause", "try_expression",
        }:
            keep.append((node.start_point[0], node.end_point[0]))
            return

        # Recurse into everything else
        for child in node.children:
            self._walk(child, keep, language)

    def _compress_body(self, block_node: Any, keep: list[tuple[int, int]]) -> None:
        """Keep only first line + return/raise statements from a function body."""
        if not block_node.children:
            return
        first = block_node.children[0]
        keep.append((first.start_point[0], first.end_point[0]))
        for child in block_node.children:
            if child.type in {"return_statement", "raise_statement", "throw_statement"}:
                keep.append((child.start_point[0], child.end_point[0]))

    def _reconstruct(self, lines: list[str], keep_ranges: list[tuple[int, int]]) -> str:
        """Rebuild source from original lines, keeping only the kept ranges."""
        if not keep_ranges:
            return "\n".join(lines)
        kept: list[str] = []
        for start, end in keep_ranges:
            for ln in range(start, min(end + 1, len(lines))):
                kept.append(lines[ln].rstrip())
        return "\n".join(kept)

    # ------------------------------------------------------------------
    # Comment helpers
    # ------------------------------------------------------------------

    def _is_pure_comment(self, line: str, language: str | None) -> bool:
        """Return True if the line is entirely a comment (no code)."""
        if not line.strip():
            return False
        lang = (language or "").lower()
        if lang in _HASH_COMMENT_LANGS or lang == "":
            if _HASH_COMMENT_RE.match(line):
                return True
        if lang in _SLASH_COMMENT_LANGS:
            if _SLASH_COMMENT_RE.match(line):
                return True
        # Python fallback
        if lang == "python" and _HASH_COMMENT_RE.match(line):
            return True
        return False

    def _is_important_comment(self, line: str) -> bool:
        """Return True if the comment contains a marker that must be preserved."""
        return bool(_IMPORTANT_COMMENT_RE.search(line))

    # ------------------------------------------------------------------
    # Docstring helpers
    # ------------------------------------------------------------------

    def _docstring_opener(self, stripped: str) -> str | None:
        """Return the triple-quote token if this line opens a Python docstring, else None."""
        if stripped.startswith('"""'):
            return '"""'
        if stripped.startswith("'''"):
            return "'''"
        return None


# ---------------------------------------------------------------------------
# Utility: merge overlapping line ranges
# ---------------------------------------------------------------------------

def set_merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent (start, end) ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
