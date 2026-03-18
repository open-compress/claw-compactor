#!/usr/bin/env python3
"""Dictionary-based compression for workspace memory files.

Learns high-frequency phrases from the workspace, builds a codebook,
and applies lossless substitution compression.

Usage:
    python3 dictionary_compress.py <workspace> --build       # Scan + generate codebook
    python3 dictionary_compress.py <workspace> --compress     # Apply codebook
    python3 dictionary_compress.py <workspace> --decompress   # Expand codes back
    python3 dictionary_compress.py <workspace> --stats        # Show compression effect

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
from claw_compactor.dictionary import (
    build_codebook, compress_text, decompress_text,
    save_codebook, load_codebook, compression_stats,
)
from claw_compactor.tokens import estimate_tokens
from claw_compactor.exceptions import FileNotFoundError_, MemCompressError

logger = logging.getLogger(__name__)

DEFAULT_CODEBOOK_PATH = "memory/.codebook.json"


def _collect_md_files(workspace: Path) -> List[Path]:
    """Collect all markdown files in workspace."""
    files: List[Path] = []
    for name in ["MEMORY.md", "TOOLS.md", "AGENTS.md", "SOUL.md", "USER.md"]:
        p = workspace / name
        if p.exists():
            files.append(p)
    mem_dir = workspace / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            if not f.name.startswith('.'):
                files.append(f)
    return files


def _read_texts(files: List[Path]) -> List[str]:
    """Read all files into a list of strings."""
    return [f.read_text(encoding="utf-8", errors="replace") for f in files]


def cmd_build(
    workspace: Path,
    codebook_path: Path,
    min_freq: int = 3,
    max_entries: int = 200,
) -> Dict[str, Any]:
    """Scan workspace and build codebook."""
    files = _collect_md_files(workspace)
    texts = _read_texts(files)
    cb = build_codebook(texts, min_freq=min_freq, max_entries=max_entries)
    save_codebook(cb, codebook_path)
    return {
        "codebook_entries": len(cb),
        "codebook_path": str(codebook_path),
        "files_scanned": len(files),
    }


def cmd_compress(
    workspace: Path,
    codebook_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Apply codebook compression to all workspace files."""
    cb = load_codebook(codebook_path)
    files = _collect_md_files(workspace)
    total_before = 0
    total_after = 0

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        before = estimate_tokens(text)
        compressed = compress_text(text, cb)
        after = estimate_tokens(compressed)
        total_before += before
        total_after += after
        if not dry_run:
            f.write_text(compressed, encoding="utf-8")

    return {
        "files": len(files),
        "tokens_before": total_before,
        "tokens_after": total_after,
        "tokens_saved": total_before - total_after,
        "dry_run": dry_run,
    }


def cmd_decompress(
    workspace: Path,
    codebook_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Expand codebook codes back to original phrases."""
    cb = load_codebook(codebook_path)
    files = _collect_md_files(workspace)

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        decompressed = decompress_text(text, cb)
        if not dry_run:
            f.write_text(decompressed, encoding="utf-8")

    return {"files": len(files), "dry_run": dry_run}


def cmd_stats(
    workspace: Path,
    codebook_path: Path,
) -> Dict[str, Any]:
    """Show compression statistics."""
    cb = load_codebook(codebook_path)
    files = _collect_md_files(workspace)
    texts = _read_texts(files)
    combined = '\n'.join(texts)
    compressed = compress_text(combined, cb)
    stats = compression_stats(combined, compressed, cb)
    stats["files"] = len(files)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Dictionary-based compression")
    parser.add_argument("workspace", help="Workspace directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Build codebook")
    group.add_argument("--compress", action="store_true", help="Apply compression")
    group.add_argument("--decompress", action="store_true", help="Expand codes")
    group.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--codebook", default=None, help="Codebook path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ws = Path(args.workspace)
    cb_path = Path(args.codebook) if args.codebook else ws / DEFAULT_CODEBOOK_PATH

    if args.build:
        result = cmd_build(ws, cb_path)
    elif args.compress:
        result = cmd_compress(ws, cb_path, dry_run=args.dry_run)
    elif args.decompress:
        result = cmd_decompress(ws, cb_path, dry_run=args.dry_run)
    else:
        result = cmd_stats(ws, cb_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
