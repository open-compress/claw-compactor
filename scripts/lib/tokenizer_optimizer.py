"""Token-level format optimization.

Applies encoding-aware transformations that reduce token count while
preserving all semantic information. Each transformation targets
specific tokenizer inefficiencies in cl100k_base / o200k_base.

Key insight: the same information can be encoded in fewer tokens
by choosing formats the tokenizer handles more efficiently.

Part of claw-compactor. License: MIT.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Chinese full-width punctuation → half-width (canonical map from unicode_maps)
from claw_compactor.unicode_maps import ZH_PUNCT_MAP as _ZH_PUNCT_MAP, normalize_zh_punctuation as _normalize_zh  # noqa: E402

# Bold/italic markdown decorators
_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_ITALIC_RE = re.compile(r'(?<!\*)\*([^*]+?)\*(?!\*)')

# Inline code that's just a plain word (not actual code)
_TRIVIAL_CODE_RE = re.compile(r'`([a-zA-Z0-9_.-]+)`')

# Markdown table detection
_TABLE_SEP_RE = re.compile(r'^[\s|:\-]+$')

# Bullet patterns
_BULLET_RE = re.compile(r'^(\s*)([-*+])\s+', re.MULTILINE)

# Multiple spaces / excessive indentation
_MULTI_SPACE_RE = re.compile(r'  +')
_LEADING_SPACES_RE = re.compile(r'^( {4,})', re.MULTILINE)


def strip_bold_italic(text: str) -> str:
    """Remove **bold** and *italic* markdown decorators."""
    if not text:
        return ""
    text = _BOLD_RE.sub(r'\1', text)
    text = _ITALIC_RE.sub(r'\1', text)
    return text


def normalize_punctuation(text: str) -> str:
    """Replace Chinese fullwidth punctuation with ASCII equivalents.

    Delegates to :func:`lib.unicode_maps.normalize_zh_punctuation` which
    is the canonical single source of truth for the character map.
    """
    return _normalize_zh(text)


def strip_trivial_backticks(text: str) -> str:
    """Remove backticks around simple words (not real code).

    Keeps backticks when content contains spaces or special chars.
    """
    if not text:
        return ""
    return _TRIVIAL_CODE_RE.sub(r'\1', text)


def minimize_whitespace(text: str) -> str:
    """Reduce multiple spaces and excessive indentation."""
    if not text:
        return ""
    # Reduce multiple spaces to single
    text = _MULTI_SPACE_RE.sub(' ', text)
    # Cap leading indentation at 4 spaces
    text = _LEADING_SPACES_RE.sub('    ', text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def compact_bullets(text: str) -> str:
    """Remove bullet prefixes from long consecutive bullet lists (3+).

    Short lists (1-2 items) keep their bullets.
    """
    if not text:
        return ""
    lines = text.split('\n')
    result: List[str] = []
    bullet_run: List[str] = []
    bullet_re = re.compile(r'^(\s*[-*+])\s+(.*)')

    def flush():
        if len(bullet_run) >= 3:
            # Strip bullet prefix
            for content in bullet_run:
                result.append(content)
        else:
            # Keep original bullets
            for content in bullet_run:
                result.append('- ' + content)
        bullet_run.clear()

    for line in lines:
        m = bullet_re.match(line)
        if m:
            bullet_run.append(m.group(2))
        else:
            flush()
            result.append(line)
    flush()
    return '\n'.join(result)


def compress_table_to_kv(text: str) -> str:
    """Convert markdown tables to compact key:value or compact format."""
    if not text:
        return ""

    lines = text.split('\n')
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if '|' in line and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            headers = [c.strip() for c in line.strip().strip('|').split('|')]
            i += 2
            rows: List[List[str]] = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(cells)
                i += 1

            if len(headers) == 2:
                for row in rows:
                    k = row[0] if len(row) > 0 else ''
                    v = row[1] if len(row) > 1 else ''
                    if k or v:
                        result.append(f"{k}: {v}")
            else:
                for row in rows:
                    result.append(' | '.join(row))
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def optimize_tokens(text: str, aggressive: bool = False) -> str:
    """Apply all token-saving optimizations.

    Args:
        text: Input text.
        aggressive: If True, apply more aggressive transformations
                    (strip bold/italic, compact bullets, strip backticks).
    """
    if not text:
        return ""
    result = normalize_punctuation(text)
    result = compress_table_to_kv(result)
    result = minimize_whitespace(result)
    if aggressive:
        result = strip_bold_italic(result)
        result = strip_trivial_backticks(result)
        result = compact_bullets(result)
    return result


def estimate_savings(original: str, optimized: str) -> dict:
    """Calculate token savings between original and optimized text."""
    from claw_compactor.tokens import estimate_tokens
    orig_tokens = estimate_tokens(original)
    opt_tokens = estimate_tokens(optimized)
    reduction = ((orig_tokens - opt_tokens) / orig_tokens * 100) if orig_tokens else 0.0
    return {
        "original_tokens": orig_tokens,
        "optimized_tokens": opt_tokens,
        "original_chars": len(original),
        "optimized_chars": len(optimized),
        "token_reduction_pct": round(reduction, 2),
    }
