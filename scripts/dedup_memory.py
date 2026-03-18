#!/usr/bin/env python3
"""Find and merge near-duplicate entries across memory files.

Uses shingle hashing for efficient similarity detection without O(n^2) comparison.

Usage:
    python3 dedup_memory.py <path> [--json] [--auto-merge] [--threshold 0.6]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens
from claw_compactor.markdown import parse_sections, strip_markdown_redundancy
from claw_compactor.dedup import find_duplicates, merge_duplicates, SIMILARITY_THRESHOLD
from claw_compactor.exceptions import FileNotFoundError_

logger = logging.getLogger(__name__)


def _collect_entries(target: str) -> List[Dict[str, Any]]:
    """Collect bullet/paragraph entries from markdown files at *target*.

    Returns a list of dicts with 'text', 'source', and 'section' keys.
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError_(f"Path not found: {target}")

    files = [path] if path.is_file() else sorted(path.rglob("*.md"))
    entries: List[Dict[str, Any]] = []

    for f in files:
        text = f.read_text(encoding="utf-8")
        if not text.strip():
            continue
        sections = parse_sections(text)
        for header, body, level in sections:
            if not body.strip():
                continue
            # Split body into bullet lines or paragraphs
            for line in body.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Skip very short lines
                    entries.append({
                        "text": line,
                        "source": str(f),
                        "section": header,
                    })

    return entries


def run_dedup(
    target: str,
    threshold: float = SIMILARITY_THRESHOLD,
    auto_merge: bool = False,
) -> Dict[str, Any]:
    """Run deduplication on *target* (file or directory).

    Returns a dict with statistics and duplicate groups.
    """
    entries = _collect_entries(target)
    texts = [e["text"] for e in entries]

    tokens_before = estimate_tokens('\n'.join(texts))
    groups = find_duplicates(texts, threshold=threshold)

    result: Dict[str, Any] = {
        "total_entries": len(entries),
        "duplicate_groups": groups,
        "duplicate_group_count": len(groups),
        "entries_removed": 0,
        "tokens_before": tokens_before,
    }

    if auto_merge and groups:
        merged = merge_duplicates(texts, groups)
        tokens_after = estimate_tokens('\n'.join(merged))
        result["entries_removed"] = len(texts) - len(merged)
        result["tokens_after"] = tokens_after
        result["tokens_saved"] = tokens_before - tokens_after

    if groups:
        result["groups"] = []
        for g in groups:
            group_entries = [entries[i] for i in g["indices"]]
            result["groups"].append({
                "similarity": g["similarity"],
                "entries": [e["text"][:100] for e in group_entries],
                "sources": list(set(e["source"] for e in group_entries)),
            })

    return result


def format_human(result: Dict[str, Any]) -> str:
    """Format dedup results as a human-readable report."""
    lines = ["# Deduplication Report", ""]
    lines.append(f"Total entries scanned: {result['total_entries']}")
    groups = result['duplicate_groups']
    num_groups = len(groups) if isinstance(groups, list) else groups
    lines.append(f"Duplicate groups found: {num_groups}")

    if not num_groups:
        lines.append("\nNo duplicates found.")
        return '\n'.join(lines)

    lines.append(f"Entries removed: {result.get('entries_removed', 0)}")
    if "tokens_saved" in result:
        lines.append(f"Tokens saved: {result['tokens_saved']}")

    if "groups" in result:
        lines.append("\n## Groups")
        for i, g in enumerate(result["groups"]):
            lines.append(f"\n### Group {i + 1} (similarity: {g['similarity']:.2f})")
            for entry in g["entries"]:
                lines.append(f"  - {entry}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Find near-duplicate memory entries")
    parser.add_argument("path", help="File or directory to scan")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--auto-merge", action="store_true")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    result = run_dedup(args.path, threshold=args.threshold, auto_merge=args.auto_merge)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_human(result))


if __name__ == "__main__":
    main()
