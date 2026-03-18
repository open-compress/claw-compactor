#!/usr/bin/env python3
"""Audit workspace memory files for token usage, staleness, and compression opportunities.

Scans all markdown files in a workspace and reports:
- Total token budget usage
- File age distribution
- Stale entries (not updated in N days)
- Compression suggestions

Usage:
    python3 audit_memory.py <workspace_path> [--stale-days 14] [--json]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens
from claw_compactor.markdown import parse_sections, compress_markdown_table, strip_emoji
from claw_compactor.exceptions import FileNotFoundError_

logger = logging.getLogger(__name__)

# Default memory token budgets
DEFAULT_BUDGETS = {
    "MEMORY.md": 2000,
    "TOOLS.md": 1500,
    "AGENTS.md": 2000,
    "daily_total": 5000,
    "workspace_total": 15000,
}


def _has_tables(text: str) -> bool:
    """Check if text contains markdown tables."""
    return '|' in text and '---' in text


def _has_emoji(text: str) -> bool:
    """Check if text contains emoji characters."""
    from claw_compactor.markdown import _EMOJI_RE
    return bool(_EMOJI_RE.search(text))


def _count_empty_sections(text: str) -> int:
    """Count sections with no meaningful body content."""
    from claw_compactor.markdown import parse_sections
    sections = parse_sections(text)
    return sum(1 for h, b, _ in sections if h and not b.strip())


def _file_age_days(path: Path) -> float:
    """Return the age of *path* in days since last modification."""
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / 86400


def audit_file(
    path: Path,
    stale_days: int = 14,
) -> Dict[str, Any]:
    """Audit a single markdown file.

    Returns dict with name, tokens, is_stale, suggestions, etc.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = estimate_tokens(text)
    age = _file_age_days(path)
    is_stale = age > stale_days

    suggestions: List[str] = []

    # Check for tables that could be compressed
    if '|' in text and '---' in text:
        compressed = compress_markdown_table(text)
        if len(compressed) < len(text) * 0.9:
            suggestions.append("Table detected — compress_markdown_table could save tokens")

    # Check for emoji
    stripped = strip_emoji(text)
    if len(stripped) < len(text):
        suggestions.append("Contains emoji — strip_emoji could save tokens")

    # Check for empty sections
    sections = parse_sections(text)
    empty_count = sum(1 for h, b, _ in sections if h and not b.strip())
    if empty_count > 0:
        suggestions.append(f"{empty_count} empty section(s) — remove_empty_sections")

    # Check token budget
    budget = DEFAULT_BUDGETS.get(path.name, DEFAULT_BUDGETS["workspace_total"])
    if tokens > budget:
        suggestions.append(f"Over budget: {tokens:,} tokens (budget: {budget:,})")

    if is_stale:
        suggestions.append(f"Stale: not modified in {age:.0f} days")

    return {
        "path": str(path),
        "file": str(path),
        "name": path.name,
        "tokens": tokens,
        "age_days": round(age, 1),
        "is_stale": is_stale,
        "suggestions": suggestions,
        "sections": len(sections),
    }


def audit_workspace(
    workspace: str,
    stale_days: int = 14,
) -> Dict[str, Any]:
    """Audit all memory files in *workspace*.

    Raises FileNotFoundError_ if workspace doesn't exist.
    """
    p = Path(workspace)
    if not p.exists():
        raise FileNotFoundError_(f"Workspace not found: {workspace}")

    files: List[Path] = []
    for f in sorted(p.glob("*.md")):
        files.append(f)
    mem_dir = p / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            files.append(f)

    if not files:
        return {
            "total_files": 0,
            "total_tokens": 0,
            "files": [],
            "age_distribution": {},
            "suggestions": [],
        }

    file_results = [audit_file(f, stale_days=stale_days) for f in files]
    total_tokens = sum(r["tokens"] for r in file_results)

    # Age distribution
    age_bins = {"<7d": 0, "7-30d": 0, "30-90d": 0, ">90d": 0}
    for r in file_results:
        age = r["age_days"]
        if age < 7:
            age_bins["<7d"] += 1
        elif age < 30:
            age_bins["7-30d"] += 1
        elif age < 90:
            age_bins["30-90d"] += 1
        else:
            age_bins[">90d"] += 1

    # Aggregate suggestions
    all_suggestions = []
    for r in file_results:
        for s in r["suggestions"]:
            all_suggestions.append(f"[{r['name']}] {s}")

    return {
        "total_files": len(file_results),
        "total_tokens": total_tokens,
        "files": file_results,
        "age_distribution": age_bins,
        "suggestions": all_suggestions,
    }


def format_report(result: Dict[str, Any]) -> str:
    """Format audit result as a human-readable report."""
    lines = [
        "=== Memory Audit Report ===",
        f"Files: {result['total_files']}",
        f"Total tokens: {result['total_tokens']:,}",
        "",
        "Age distribution:",
    ]
    for bucket, count in result.get("age_distribution", {}).items():
        lines.append(f"  {bucket}: {count} files")

    if result.get("suggestions"):
        lines.append("\nSuggestions:")
        for s in result["suggestions"]:
            lines.append(f"  - {s}")
    else:
        lines.append("\nNo suggestions — workspace looks healthy.")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit workspace memory files")
    parser.add_argument("workspace", help="Workspace directory")
    parser.add_argument("--stale-days", type=int, default=14, help="Days before stale")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    result = audit_workspace(args.workspace, stale_days=args.stale_days)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
