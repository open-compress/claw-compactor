"""Markdown parsing and manipulation utilities.

Part of claw-compactor. License: MIT.
"""

import re
import logging
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional

from claw_compactor.unicode_maps import ZH_PUNCT_MAP as _ZH_PUNCT_MAP, normalize_zh_punctuation as _normalize_zh

logger = logging.getLogger(__name__)

# Emoji pattern (broad: emoticons, symbols, pictographs, etc.)
_EMOJI_RE = re.compile(
    '[\U0001F600-\U0001F64F'   # emoticons
    '\U0001F300-\U0001F5FF'    # symbols & pictographs
    '\U0001F680-\U0001F6FF'    # transport & map
    '\U0001F1E0-\U0001F1FF'    # flags
    '\U00002702-\U000027B0'    # dingbats
    '\U0001F900-\U0001F9FF'    # supplemental symbols
    '\U0001FA00-\U0001FA6F'    # chess symbols
    '\U0001FA70-\U0001FAFF'    # symbols extended-A
    '\U00002600-\U000026FF'    # misc symbols
    ']+', re.UNICODE
)

# Header regex
_HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)

# Table separator line
_TABLE_SEP_RE = re.compile(r'^[\s|:\-]+$')


def parse_sections(text: str) -> List[Tuple[str, str, int]]:
    """Parse *text* into sections delimited by markdown headers.

    Returns a list of (header, body, level) tuples.
    A preamble (text before the first header) is returned with header=''.
    """
    if not text:
        return []

    sections: List[Tuple[str, str, int]] = []
    lines = text.split('\n')
    current_header = ''
    current_level = 0
    current_body_lines: List[str] = []

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            # Save previous section
            body = '\n'.join(current_body_lines).strip()
            if current_header or body:
                sections.append((current_header, body, current_level))
            current_header = m.group(2).strip()
            current_level = len(m.group(1))
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Last section
    body = '\n'.join(current_body_lines).strip()
    if current_header or body:
        sections.append((current_header, body, current_level))

    return sections


def strip_markdown_redundancy(text: str) -> str:
    """Remove excessive blank lines and trailing whitespace."""
    if not text:
        return ""
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def remove_duplicate_lines(text: str) -> str:
    """Remove exact duplicate non-blank lines, preserving order."""
    if not text:
        return ""
    seen = set()
    result = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            # Preserve blank lines
            result.append(line)
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        result.append(line)
    return '\n'.join(result)


def normalize_chinese_punctuation(text: str) -> str:
    """Replace Chinese fullwidth punctuation with ASCII equivalents.

    Delegates to :func:`lib.unicode_maps.normalize_zh_punctuation` which
    is the canonical single source of truth for the character map.
    """
    return _normalize_zh(text)


def strip_emoji(text: str) -> str:
    """Remove emoji characters from *text*."""
    if not text:
        return ""
    result = _EMOJI_RE.sub('', text)
    # Collapse multiple spaces left by emoji removal
    result = re.sub(r'  +', ' ', result)
    return result


def remove_empty_sections(text: str) -> str:
    """Remove markdown sections that have no meaningful body content."""
    if not text:
        return ""
    sections = parse_sections(text)
    if not sections:
        return text

    # Determine which sections have children (a deeper section follows)
    has_child = [False] * len(sections)
    for idx, (header, body, level) in enumerate(sections):
        if level > 0:
            # Look backwards for a parent
            for pidx in range(idx - 1, -1, -1):
                _, _, plevel = sections[pidx]
                if plevel > 0 and plevel < level:
                    has_child[pidx] = True
                    break

    result_lines: List[str] = []
    for idx, (header, body, level) in enumerate(sections):
        if not header and not body:
            continue
        if header and not body.strip() and not has_child[idx]:
            continue  # Empty section with no children
        if header:
            result_lines.append('#' * level + ' ' + header)
        if body.strip():
            result_lines.append(body)
        result_lines.append('')  # Blank line between sections

    return '\n'.join(result_lines).strip()


def compress_markdown_table(text: str) -> str:
    """Convert markdown tables to compact key:value notation.

    A 2-column table becomes ``Key: Value`` lines.
    Multi-column tables become ``Col1 | Col2 | ...`` lines (no header row / separator).
    """
    if not text:
        return ""

    lines = text.split('\n')
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Detect a table: line with | ... | followed by separator |---|
        if '|' in line and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            # Parse header row
            headers = [c.strip() for c in line.strip().strip('|').split('|')]
            i += 2  # skip header + separator
            rows: List[List[str]] = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(cells)
                i += 1

            if len(headers) >= 5:
                # Wide tables: preserve as-is but without header/separator
                for row in rows:
                    result.append('| ' + ' | '.join(row) + ' |')
            elif len(headers) == 2:
                # 2-column: key: value format
                for row in rows:
                    k = row[0] if len(row) > 0 else ''
                    v = row[1] if len(row) > 1 else ''
                    if k or v:
                        result.append(f"- {k}: {v}")
            else:
                # Multi-column: compact format using headers as labels
                for row in rows:
                    parts = []
                    for ci, cell in enumerate(row):
                        if ci == 0:
                            parts.append(cell)
                        elif ci < len(headers):
                            parts.append(f"{headers[ci]}={cell}")
                        else:
                            parts.append(cell)
                    result.append(', '.join(parts))
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def merge_similar_bullets(text: str, threshold: float = 0.80) -> str:
    """Merge bullet lines with high similarity.

    Uses SequenceMatcher ratio. When two bullets exceed *threshold*,
    keep the longer one.
    """
    if not text:
        return ""

    lines = text.split('\n')
    bullet_re = re.compile(r'^(\s*[-*+]\s+)(.*)')
    result: List[str] = []
    bullets: List[Tuple[str, str, str]] = []  # (prefix, content, full_line)

    def flush_bullets():
        if not bullets:
            return
        kept = list(bullets)
        merged_out: List[bool] = [False] * len(kept)
        for i in range(len(kept)):
            if merged_out[i]:
                continue
            for j in range(i + 1, len(kept)):
                if merged_out[j]:
                    continue
                ratio = SequenceMatcher(None, kept[i][1], kept[j][1]).ratio()
                if ratio >= threshold:
                    # Keep the longer one
                    if len(kept[j][1]) > len(kept[i][1]):
                        merged_out[i] = True
                        break
                    else:
                        merged_out[j] = True
        for idx, (prefix, content, full_line) in enumerate(kept):
            if not merged_out[idx]:
                result.append(full_line)
        bullets.clear()

    for line in lines:
        m = bullet_re.match(line)
        if m:
            bullets.append((m.group(1), m.group(2), line))
        else:
            flush_bullets()
            result.append(line)

    flush_bullets()
    return '\n'.join(result)


def merge_short_bullets(text: str, max_words: int = 3, max_merge: int = 10) -> str:
    """Combine consecutive short bullet points into comma-separated form.

    Bullets with <= *max_words* words are candidates. Up to *max_merge*
    consecutive short bullets are joined into one line.
    """
    if not text:
        return ""

    bullet_re = re.compile(r'^(\s*[-*+]\s+)(.*)')
    lines = text.split('\n')
    result: List[str] = []
    short_bullets: List[str] = []
    prefix = '- '

    def flush_short():
        nonlocal prefix
        if not short_bullets:
            return
        if len(short_bullets) <= 2:
            for sb in short_bullets:
                result.append(prefix + sb)
        else:
            # Merge into one line
            result.append(prefix + ', '.join(short_bullets))
        short_bullets.clear()

    for line in lines:
        m = bullet_re.match(line)
        if m:
            content = m.group(2).strip()
            prefix = m.group(1)
            if len(content.split()) <= max_words:
                short_bullets.append(content)
                if len(short_bullets) >= max_merge:
                    flush_short()
            else:
                flush_short()
                result.append(line)
        else:
            flush_short()
            result.append(line)

    flush_short()
    return '\n'.join(result)
