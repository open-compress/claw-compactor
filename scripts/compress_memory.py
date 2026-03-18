#!/usr/bin/env python3
"""Compress memory files using rule-based preprocessing + LLM semantic compression.

Two-phase approach:
1. Rule engine: dedup lines, strip markdown redundancy, merge similar entries
2. LLM prompt: generate a prompt for semantic compression of remaining content

Usage:
    python3 compress_memory.py <path> [--dry-run] [--output FILE] [--older-than DAYS] [--no-llm]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens, using_tiktoken
from claw_compactor.markdown import (
    strip_markdown_redundancy, remove_duplicate_lines, parse_sections,
    normalize_chinese_punctuation, strip_emoji, remove_empty_sections,
    compress_markdown_table, merge_similar_bullets, merge_short_bullets,
)
from claw_compactor.dedup import find_duplicates, merge_duplicates
from claw_compactor.exceptions import FileNotFoundError_, MemCompressError

logger = logging.getLogger(__name__)

# LLM prompt template for semantic compression
COMPRESS_PROMPT = """You are a memory compression assistant. Compress the following text to approximately {target_pct}% of its current size while preserving ALL factual information, decisions, configurations, and actionable items.

Rules:
- Keep all names, IPs, paths, tokens, dates, and technical details EXACTLY
- Remove filler words, redundant explanations, and verbose phrasing
- Merge related items
- Use concise notation (key:value, abbreviations)
- Preserve markdown structure (headers, bullets)
- Output ONLY the compressed text, no commentary

Text to compress:
---
{content}
---

Compressed version:"""


def _file_age_days(path: Path) -> float:
    """Return file age in days based on mtime."""
    return (time.time() - path.stat().st_mtime) / 86400


def rule_compress(
    text: str,
    enable_emoji_strip: bool = True,
) -> str:
    """Apply all rule-based compression passes to *text*.

    Returns the compressed text. Never increases token count.
    """
    if not text:
        return ""

    result = text

    # 1. Normalize Chinese punctuation
    result = normalize_chinese_punctuation(result)

    # 2. Strip markdown redundancy (excess blanks, trailing whitespace)
    result = strip_markdown_redundancy(result)

    # 3. Remove duplicate lines
    result = remove_duplicate_lines(result)

    # 4. Remove empty sections
    result = remove_empty_sections(result)

    # 5. Compress markdown tables to key:value
    result = compress_markdown_table(result)

    # 6. Strip emoji if enabled
    if enable_emoji_strip:
        result = strip_emoji(result)

    # 7. Merge similar bullets
    result = merge_similar_bullets(result)

    # 8. Merge short bullets
    result = merge_short_bullets(result)

    # 9. Final cleanup
    result = strip_markdown_redundancy(result)

    return result


def generate_llm_prompt(content: str, target_pct: int = 50) -> str:
    """Generate an LLM prompt for semantic compression of *content*."""
    return COMPRESS_PROMPT.format(content=content, target_pct=target_pct)


def _collect_files(
    target: str,
    older_than: Optional[int] = None,
) -> List[Path]:
    """Collect markdown files from *target* (file or directory).

    If *older_than* is set, only include files older than N days.
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError_(f"Path not found: {target}")

    if path.is_file():
        if older_than is not None and _file_age_days(path) < older_than:
            return []
        return [path]

    # Directory: collect all .md files recursively, excluding junk dirs
    _EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv",
                     "target", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache"}
    files = []
    for f in sorted(path.rglob("*.md")):
        if _EXCLUDE_DIRS.intersection(f.relative_to(path).parts):
            continue
        files.append(f)
    if older_than is not None:
        files = [f for f in files if _file_age_days(f) >= older_than]
    return files


def compress_file(
    path: Path,
    dry_run: bool = False,
    output: Optional[str] = None,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """Compress a single file using rule-based compression.

    Args:
        path: File to compress.
        dry_run: If True, don't write changes.
        output: Optional output file path.
        no_llm: If True, skip LLM prompt generation.

    Returns a dict with compression statistics.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    original_tokens = estimate_tokens(original)

    compressed = rule_compress(original)
    compressed_tokens = estimate_tokens(compressed)

    reduction_pct = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens else 0.0
    result: Dict[str, Any] = {
        "file": str(path),
        "original_tokens": original_tokens,
        "rule_compressed_tokens": compressed_tokens,
        "rule_reduction_pct": round(reduction_pct, 2),
        "dry_run": dry_run,
    }

    if not no_llm and compressed.strip():
        result["llm_prompt"] = generate_llm_prompt(compressed)

    if not dry_run:
        target = Path(output) if output else path
        target.write_text(compressed, encoding="utf-8")
        result["written_to"] = str(target)

    return result


def llm_compress_file(
    path: Path,
    target_pct: int = 40,
) -> Dict[str, Any]:
    """Generate an LLM compression prompt for a file and write it to a .prompt file.

    Returns stats dict with original_tokens, rule_compressed_tokens, prompt_file, etc.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    original_tokens = estimate_tokens(text)
    compressed = rule_compress(text)
    rule_tokens = estimate_tokens(compressed)
    prompt = generate_llm_prompt(compressed, target_pct)
    prompt_tokens = estimate_tokens(prompt)

    prompt_path = path.with_suffix(".prompt.md")
    prompt_path.write_text(prompt, encoding="utf-8")

    return {
        "file": str(path),
        "original_tokens": original_tokens,
        "rule_compressed_tokens": rule_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_file": str(prompt_path),
        "target_pct": target_pct,
        "instruction": f"Feed this prompt to an LLM for further {target_pct}% compression.",
    }


def main():
    parser = argparse.ArgumentParser(description="Compress memory files")
    parser.add_argument("path", help="File or directory to compress")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--older-than", type=int, help="Only files older than N days")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM prompt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = _collect_files(args.path, older_than=args.older_than)
    results = []
    for f in files:
        r = compress_file(f, dry_run=args.dry_run, output=args.output, no_llm=args.no_llm)
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            saved = r["original_tokens"] - r["rule_compressed_tokens"]
            print(f"{r['file']}: {r['original_tokens']} → {r['rule_compressed_tokens']} tokens (saved {saved})")


if __name__ == "__main__":
    main()
