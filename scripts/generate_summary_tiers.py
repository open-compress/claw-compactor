#!/usr/bin/env python3
"""Generate tiered summaries from MEMORY.md files.

Creates Level 0/1/2 summary templates with token budgets:
- Level 0 (Ultra-compact): ~200 tokens - key facts only
- Level 1 (Working memory): ~1000 tokens - active context
- Level 2 (Full context): ~3000 tokens - complete reference

Usage:
    python3 generate_summary_tiers.py <path> [--json] [--output-dir DIR]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens
from claw_compactor.markdown import parse_sections
from claw_compactor.exceptions import FileNotFoundError_

logger = logging.getLogger(__name__)

# Tier definitions
TIERS = {
    0: {"name": "Ultra-compact", "budget": 200, "description": "Key facts and critical decisions only"},
    1: {"name": "Working memory", "budget": 1000, "description": "Active context for daily work"},
    2: {"name": "Full context", "budget": 3000, "description": "Complete reference with details"},
}

# Section priority for compression (higher = keep more)
SECTION_PRIORITIES = {
    "decision": 10,
    "critical": 10,
    "important": 9,
    "action": 8,
    "todo": 8,
    "config": 7,
    "setup": 7,
    "architecture": 7,
    "preference": 6,
    "convention": 6,
    "lesson": 5,
    "note": 4,
    "log": 3,
    "history": 2,
    "archive": 1,
}

DEFAULT_PRIORITY = 5


def _classify_section(header: str) -> int:
    """Classify a section header by priority.

    Returns a priority score (1-10). Higher = more important.
    """
    h = header.lower()
    for keyword, priority in SECTION_PRIORITIES.items():
        if keyword in h:
            return priority
    return DEFAULT_PRIORITY


def _find_memory_files(target: str) -> List[Path]:
    """Find memory files to process.

    Raises FileNotFoundError_ if target doesn't exist.
    """
    p = Path(target)
    if not p.exists():
        raise FileNotFoundError_(f"Path not found: {target}")

    if p.is_file():
        return [p]

    files = []
    # Prioritize MEMORY.md
    mem = p / "MEMORY.md"
    if mem.exists():
        files.append(mem)
    # Add other root .md files
    for f in sorted(p.glob("*.md")):
        if f.name != "MEMORY.md" and f not in files:
            files.append(f)
    # memory/ subdirectory
    mem_dir = p / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            files.append(f)

    return files


def generate_tiers(files: List[Path]) -> Dict[str, Any]:
    """Generate tier analysis from memory files.

    Returns a dict with total_tokens, total_sections, and per-tier info.
    """
    # Collect all sections with priorities
    all_sections: List[Dict[str, Any]] = []
    total_tokens = 0

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        tokens = estimate_tokens(text)
        total_tokens += tokens
        sections = parse_sections(text)
        for header, body, level in sections:
            sec_tokens = estimate_tokens(header + '\n' + body) if (header or body) else 0
            priority = _classify_section(header) if header else DEFAULT_PRIORITY
            all_sections.append({
                "header": header,
                "body": body,
                "level": level,
                "tokens": sec_tokens,
                "priority": priority,
                "file": str(f),
            })

    # Sort by priority descending, then by token count ascending
    all_sections.sort(key=lambda s: (-s["priority"], s["tokens"]))

    # Build tiers
    tiers: Dict[int, Dict[str, Any]] = {}
    for tier_level, tier_def in TIERS.items():
        budget = tier_def["budget"]
        selected: List[Dict[str, Any]] = []
        used = 0
        for sec in all_sections:
            if used + sec["tokens"] <= budget:
                selected.append(sec)
                used += sec["tokens"]
        tiers[tier_level] = {
            "name": tier_def["name"],
            "budget": budget,
            "description": tier_def["description"],
            "sections_included": len(selected),
            "tokens_used": used,
            "sections": selected,
        }

    return {
        "total_tokens": total_tokens,
        "total_sections": len(all_sections),
        "tiers": tiers,
    }


def format_tier_template(result: Dict[str, Any], level: int) -> str:
    """Format a tier as a markdown template."""
    tier = result["tiers"][level]
    lines = [
        f"# Level {level} — {tier['name']}",
        f"Budget: {tier['budget']} tokens | Used: {tier['tokens_used']}",
        f"Sections: {tier['sections_included']}",
        "",
    ]
    for sec in tier["sections"]:
        if sec["header"]:
            lines.append(f"## {sec['header']}")
        if sec["body"]:
            lines.append(sec["body"])
        lines.append("")

    return '\n'.join(lines)


def format_human(result: Dict[str, Any]) -> str:
    """Format tier analysis as a human-readable report."""
    lines = [
        "=== Summary Tier Analysis ===",
        f"Total tokens: {result['total_tokens']:,}",
        f"Total sections: {result['total_sections']}",
        "",
    ]
    for level in range(3):
        tier = result["tiers"][level]
        lines.append(f"Level {level} ({tier['name']}):")
        lines.append(f"  Budget: {tier['budget']} tokens")
        lines.append(f"  Used: {tier['tokens_used']} tokens")
        lines.append(f"  Sections: {tier['sections_included']}")
        lines.append("")

    return '\n'.join(lines)


def extract_key_facts(text: str) -> List[str]:
    """Extract key facts from markdown text.

    Identifies lines with key:value patterns, important markers, and
    critical information. Returns deduplicated list of fact strings.
    """
    if not text:
        return []

    facts: List[str] = []
    seen: set = set()

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip headers
        if line.startswith('#'):
            continue
        # Strip bullet prefix
        clean = line.lstrip('- *+').strip()
        if not clean:
            continue

        # Key:value patterns, important markers
        is_fact = (
            ':' in clean
            or any(m in line for m in ['⚠️', 'Critical', 'Important', 'IMPORTANT', 'WARNING'])
            or any(c.isdigit() for c in clean)  # Contains numbers
        )

        if is_fact and clean not in seen:
            seen.add(clean)
            facts.append(clean)

    return facts


def generate_auto_summary(
    files: List[Path],
    budget: int = 200,
) -> str:
    """Generate an automatic summary from memory files within token budget.

    Extracts key facts and fills up to budget tokens.
    """
    all_facts: List[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        all_facts.extend(extract_key_facts(text))

    lines = ["# Auto Summary", ""]
    used_tokens = estimate_tokens('\n'.join(lines))

    for fact in all_facts:
        fact_line = f"- {fact}"
        fact_tokens = estimate_tokens(fact_line)
        if used_tokens + fact_tokens > budget:
            break
        lines.append(fact_line)
        used_tokens += fact_tokens

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate tiered summaries")
    parser.add_argument("path", help="File or directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output-dir", help="Write tier files to this directory")
    args = parser.parse_args()

    files = _find_memory_files(args.path)
    result = generate_tiers(files)

    if args.json:
        # Make JSON-serializable (remove section bodies for brevity)
        output = {
            "total_tokens": result["total_tokens"],
            "total_sections": result["total_sections"],
            "tiers": {
                k: {kk: vv for kk, vv in v.items() if kk != "sections"}
                for k, v in result["tiers"].items()
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_human(result))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for level in range(3):
            (out / f"MEMORY-L{level}.md").write_text(
                format_tier_template(result, level), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
