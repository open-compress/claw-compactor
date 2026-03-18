#!/usr/bin/env python3
"""Estimate token counts for memory files in a workspace.

Scans markdown files, estimates token usage, and reports compression potential.

Usage:
    python3 estimate_tokens.py <path> [--json] [--threshold N]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens, using_tiktoken
from claw_compactor.markdown import strip_markdown_redundancy
from claw_compactor.exceptions import FileNotFoundError_

logger = logging.getLogger(__name__)

# Compression potential scoring
POTENTIAL_THRESHOLDS = {
    "high": 2000,
    "medium": 500,
    "low": 0,
}


def _score_potential(tokens: int, stripped_tokens: int) -> str:
    """Score compression potential based on token count and reducibility."""
    ratio = (tokens - stripped_tokens) / tokens if tokens > 0 else 0
    if tokens >= POTENTIAL_THRESHOLDS["high"] or ratio >= 0.15:
        return "high"
    if ratio > 0.05 or tokens >= POTENTIAL_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _collect_md_files(path: Path) -> List[Path]:
    """Collect markdown files from path (file or directory)."""
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError_(f"Path not found: {path}")
    files = []
    # Root-level .md files
    for f in sorted(path.glob("*.md")):
        files.append(f)
    # memory/ subdirectory
    mem_dir = path / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            files.append(f)
    return files


def scan_path(path: str, threshold: int = 0) -> List[Dict[str, Any]]:
    """Scan *path* for markdown files and estimate token usage.

    Returns a list of dicts sorted by token count descending.
    Raises FileNotFoundError_ if path doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError_(f"Path not found: {path}")

    files = _collect_md_files(p) if p.is_dir() else [p]
    results: List[Dict[str, Any]] = []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        tokens = estimate_tokens(text)
        stripped = strip_markdown_redundancy(text)
        stripped_tokens = estimate_tokens(stripped)
        potential = _score_potential(tokens, stripped_tokens)

        if tokens >= threshold:
            results.append({
                "file": str(f),
                "name": f.name,
                "tokens": tokens,
                "stripped_tokens": stripped_tokens,
                "potential": potential,
                "size_bytes": len(text.encode("utf-8")),
            })

    results.sort(key=lambda r: r["tokens"], reverse=True)
    return results


def format_human(results: List[Dict[str, Any]]) -> str:
    """Format scan results as a human-readable report."""
    if not results:
        return "No files found or all below threshold."

    total = sum(r["tokens"] for r in results)
    lines = [
        "=== Token Estimation Report ===",
        f"Engine: {'tiktoken' if using_tiktoken() else 'heuristic'}",
        f"Files: {len(results)}",
        f"Total tokens: {total:,}",
        "",
    ]
    for r in results:
        lines.append(f"  {r['name']:30s} {r['tokens']:>8,} tokens  [{r['potential']}]")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Estimate token usage in memory files")
    parser.add_argument("path", help="File or directory to scan")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--threshold", type=int, default=0, help="Min tokens to show")
    args = parser.parse_args()

    results = scan_path(args.path, threshold=args.threshold)
    if args.json:
        print(json.dumps({"files": results, "total_tokens": sum(r["tokens"] for r in results)}, indent=2))
    else:
        print(format_human(results))


if __name__ == "__main__":
    main()
