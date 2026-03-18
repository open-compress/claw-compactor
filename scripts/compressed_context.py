#!/usr/bin/env python3
"""Compressed Context Protocol -- compress text for expensive model consumption.

Generates ultra-compressed context + decompression instructions for system prompts.
Three compression levels: ultra, medium, light.

Usage:
    python3 compressed_context.py <file> [--level ultra|medium|light] [--output FILE]

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from claw_compactor.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Decompression instructions to prepend to system prompt
DECOMPRESS_INSTRUCTIONS = {
    "ultra": (
        "Compressed notation: key:val=attribute, loc:X+Y=locations, "
        "Ny+=N+ years, slash-separated=alternatives. "
        "Expand naturally when responding."
    ),
    "medium": (
        "Text uses abbreviated notation: key:value pairs, "
        "condensed lists, minimal punctuation. Read as natural language."
    ),
    "light": (
        "Text is lightly condensed. Read normally."
    ),
}

# Common words to abbreviate in ultra mode
ULTRA_ABBREVS = {
    "experience": "exp",
    "management": "mgmt",
    "development": "dev",
    "approximately": "~",
    "application": "app",
    "applications": "apps",
    "configuration": "config",
    "information": "info",
    "environment": "env",
    "infrastructure": "infra",
    "architecture": "arch",
    "implementation": "impl",
    "performance": "perf",
    "operations": "ops",
    "production": "prod",
    "repository": "repo",
    "repositories": "repos",
    "documentation": "docs",
    "communication": "comms",
    "organization": "org",
    "technology": "tech",
    "technologies": "tech",
    "cryptocurrency": "crypto",
    "quantitative": "quant",
    "distributed": "dist",
    "international": "intl",
    "professional": "pro",
    "certificate": "cert",
    "authentication": "auth",
    "authorization": "authz",
    "database": "db",
    "kubernetes": "k8s",
    "continuous": "cont",
    "integration": "integ",
    "deployment": "deploy",
    "monitoring": "mon",
    "notification": "notif",
    "requirements": "reqs",
    "specification": "spec",
    "administrator": "admin",
    "description": "desc",
    "transaction": "tx",
    "transactions": "txs",
    "currently": "curr",
    "previously": "prev",
    "following": "foll",
    "including": "incl",
    "especially": "esp",
    "engineering": "eng",
    "university": "univ",
    "founded": "founder",
    "established": "est",
    "headquarters": "HQ",
    "years of": "y+",
    "based in": "loc:",
    "located in": "loc:",
    "offices in": "offices:",
    "founder of": "founder:",
    "CEO of": "CEO:",
    "CTO of": "CTO:",
}

# Filler phrases to remove in ultra mode
ULTRA_FILLERS = [
    "In addition,", "Furthermore,", "Moreover,", "Additionally,",
    "It is worth noting that", "It should be noted that",
    "As a matter of fact,", "In fact,", "Actually,",
    "Basically,", "Essentially,", "In other words,",
    "That being said,", "Having said that,",
    "At the end of the day,", "When it comes to",
    "In terms of", "With regard to", "With respect to",
    "As mentioned earlier,", "As previously stated,",
    "It is important to note that", "Please note that",
    "In conclusion,", "To summarize,", "To sum up,",
    "extensive experience", "extensive experience in",
    "He has", "She has", "They have",
    "which is", "that is", "who is",
    "a wide range of", "a variety of",
]

# Medium-mode abbreviations (less aggressive)
MEDIUM_ABBREVS = {
    "configuration": "config",
    "application": "app",
    "environment": "env",
    "infrastructure": "infra",
    "implementation": "impl",
    "documentation": "docs",
    "database": "db",
    "kubernetes": "k8s",
}


def compress_ultra(text: str) -> str:
    """Apply ultra compression -- aggressive abbreviation and filler removal."""
    if not text:
        return ""

    result = text

    # Remove fillers
    for filler in ULTRA_FILLERS:
        result = result.replace(filler, "")

    # Apply abbreviations (case-insensitive for the word, preserve surrounding)
    for word, abbrev in ULTRA_ABBREVS.items():
        # Replace whole words
        result = re.sub(r'\b' + re.escape(word) + r'\b', abbrev, result, flags=re.IGNORECASE)

    # Remove articles and common short fillers
    result = re.sub(r'\b(?:the|a|an|is|are|was|were|has|have|had|been|being)\b\s*', '', result, flags=re.IGNORECASE)
    # Remove "of" in common phrases but keep meaningful ones
    result = re.sub(r'\bof\b\s+', ' ', result)
    # Remove "and" → "+"
    result = re.sub(r'\band\b', '+', result)
    # Remove "with" → "w/"
    result = re.sub(r'\bwith\b', 'w/', result)
    # Remove "for" → "4"
    result = re.sub(r'\bfor\b', '4', result)
    # "in" → "in" (keep, too short to abbreviate)

    # Clean up spacing
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'^\s+', '', result, flags=re.MULTILINE)

    return result.strip()


def compress_medium(text: str) -> str:
    """Apply medium compression -- moderate abbreviation."""
    if not text:
        return ""

    result = text

    # Apply medium abbreviations only
    for word, abbrev in MEDIUM_ABBREVS.items():
        result = re.sub(r'\b' + re.escape(word) + r'\b', abbrev, result, flags=re.IGNORECASE)

    # Remove some fillers
    for filler in ULTRA_FILLERS[:5]:  # Only the most common
        result = result.replace(filler, "")

    # Clean up
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def compress_light(text: str) -> str:
    """Apply light compression -- just cleanup."""
    if not text:
        return ""

    result = text
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def compress(text: str, level: str) -> Dict[str, str]:
    """Compress text at the specified level.

    Returns dict with compressed text, instructions, and level.
    Raises ValueError for invalid level.
    """
    if level not in DECOMPRESS_INSTRUCTIONS:
        raise ValueError(f"Invalid compression level: {level}. Use: ultra, medium, light")

    compressors = {
        "ultra": compress_ultra,
        "medium": compress_medium,
        "light": compress_light,
    }

    compressed = compressors[level](text)
    return {
        "compressed": compressed,
        "instructions": DECOMPRESS_INSTRUCTIONS[level],
        "level": level,
    }


def compress_with_stats(text: str, level: str) -> Dict[str, Any]:
    """Compress text and return statistics.

    Returns dict with compressed text, token counts, and reduction percentage.
    """
    result = compress(text, level)
    orig_tokens = estimate_tokens(text)
    comp_tokens = estimate_tokens(result["compressed"])
    inst_tokens = estimate_tokens(result["instructions"])

    # Net includes instruction overhead
    net_tokens = comp_tokens + inst_tokens
    reduction = ((orig_tokens - comp_tokens) / orig_tokens * 100) if orig_tokens > 0 else 0.0

    return {
        "compressed": result["compressed"],
        "instructions": result["instructions"],
        "level": level,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "instruction_tokens": inst_tokens,
        "net_tokens": net_tokens,
        "reduction_pct": round(reduction, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Compressed Context Protocol")
    parser.add_argument("file", help="File to compress")
    parser.add_argument("--level", default="ultra", choices=["ultra", "medium", "light"])
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    stats = compress_with_stats(text, args.level)

    if args.output:
        Path(args.output).write_text(stats["compressed"], encoding="utf-8")

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        pct = stats["reduction_pct"]
        print(f"Level: {args.level}")
        print(f"Original: {stats['original_tokens']} tokens")
        print(f"Compressed: {stats['compressed_tokens']} tokens ({pct:.1f}% reduction)")
        print(f"Instructions: {stats['instruction_tokens']} tokens")


if __name__ == "__main__":
    main()
