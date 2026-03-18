#!/usr/bin/env python3
"""CLI entry point for claw-compactor."""

import sys


def main():
    """Dispatch to mem_compress.py with rewritten arguments."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    args = sys.argv[1:]
    if len(args) >= 2 and not args[0].startswith("-"):
        command, workspace = args[0], args[1]
        args = [workspace, command] + args[2:]
    elif len(args) == 1 and args[0] in ("-h", "--help"):
        pass
    sys.argv = ["claw-compactor"] + args

    from scripts.mem_compress import main as _run

    _run()


if __name__ == "__main__":
    main()
