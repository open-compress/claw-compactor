#!/usr/bin/env python3
"""
engram_cli.py — Command-line interface for the Engram (Observational Memory) engine.

Commands:
    observe   --thread <id>               Force Observer run for a thread
    reflect   --thread <id>               Force Reflector run for a thread
    status    [--thread <id>]             Show status for one or all threads
    ingest    --thread <id> --input <f>   Import messages from JSON/JSONL file
    context   --thread <id>               Print injectable context string
    daemon    --thread <id>               Stdin daemon: read JSONL messages, auto-compress
    auto      [--daemon] [--dry-run]      Multi-channel auto-runner (uses engram.yaml)

Usage:
    python3 scripts/engram_cli.py <workspace> <command> [options]

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure scripts/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.engram import EngramEngine
from lib.config import load_engram_config, engram_engine_kwargs
from lib.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_observe(engine: EngramEngine, args: argparse.Namespace) -> int:
    """Force Observer run for a thread."""
    thread_id = args.thread
    result = engine.observe(thread_id)
    if result is None:
        print(f"No pending messages for thread '{thread_id}'.", file=sys.stderr)
        return 1
    tokens = estimate_tokens(result)
    if getattr(args, "json", False):
        print(json.dumps({"thread_id": thread_id, "observation_tokens": tokens, "text": result}))
    else:
        print(f"✅ Observer completed for thread '{thread_id}' ({tokens} tokens produced).")
        print()
        print(result)
    return 0


def cmd_reflect(engine: EngramEngine, args: argparse.Namespace) -> int:
    """Force Reflector run for a thread."""
    thread_id = args.thread
    result = engine.reflect(thread_id)
    if result is None:
        print(f"No observations for thread '{thread_id}'.", file=sys.stderr)
        return 1
    tokens = estimate_tokens(result)
    if getattr(args, "json", False):
        print(json.dumps({"thread_id": thread_id, "reflection_tokens": tokens, "text": result}))
    else:
        print(f"✅ Reflector completed for thread '{thread_id}' ({tokens} tokens produced).")
        print()
        print(result)
    return 0


def cmd_status(engine: EngramEngine, args: argparse.Namespace) -> int:
    """Show status for one or all threads."""
    thread_id = getattr(args, "thread", None)

    if thread_id:
        threads = [thread_id]
    else:
        threads = engine.storage.list_threads()
        if not threads:
            print("No Engram threads found.")
            return 0

    rows = []
    for tid in threads:
        ctx = engine.get_context(tid)
        rows.append({
            "thread_id": tid,
            "pending_count": ctx["stats"]["pending_count"],
            "pending_tokens": ctx["stats"]["pending_tokens"],
            "observation_tokens": ctx["stats"]["observation_tokens"],
            "reflection_tokens": ctx["stats"]["reflection_tokens"],
            "total_tokens": ctx["stats"]["total_tokens"],
            "last_observed_at": ctx["meta"].get("last_observed_at", "—"),
            "last_reflected_at": ctx["meta"].get("last_reflected_at", "—"),
        })

    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    # Human-readable table
    print(f"{'Thread':<24} {'Pending':>7} {'Obs tok':>8} {'Ref tok':>8} {'Total':>7} Last Observed")
    print("─" * 80)
    for r in rows:
        print(
            f"{r['thread_id']:<24} "
            f"{r['pending_count']:>7} "
            f"{r['observation_tokens']:>8,} "
            f"{r['reflection_tokens']:>8,} "
            f"{r['total_tokens']:>7,} "
            f"{r['last_observed_at']}"
        )
    return 0


def cmd_ingest(engine: EngramEngine, args: argparse.Namespace) -> int:
    """
    Import messages from a JSON or JSONL file into a thread.

    Supported formats:
      • JSONL  — one message dict per line  (``{"role":…,"content":…}``)
      • JSON   — an array of message dicts  (``[{"role":…,"content":…}, …]``)
    """
    thread_id = args.thread
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    text = input_path.read_text(encoding="utf-8")
    messages = []

    # Try JSON array first, then JSONL
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            messages = parsed
        elif isinstance(parsed, dict):
            messages = [parsed]
        else:
            print("Error: JSON file must contain an array or a single message object.", file=sys.stderr)
            return 1
    except json.JSONDecodeError:
        # Try JSONL
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping line {lineno}: {exc}", file=sys.stderr)

    if not messages:
        print("No messages found in input file.", file=sys.stderr)
        return 1

    # Filter out messages with no content before batch ingestion
    valid_messages = [
        {"role": msg.get("role", "user"), "content": msg.get("content", ""),
         "timestamp": msg.get("timestamp")}
        for msg in messages
        if msg.get("content")
    ]
    count = len(valid_messages)

    if count > 0:
        engine.batch_ingest(thread_id, valid_messages)

    if getattr(args, "json", False):
        print(json.dumps({"thread_id": thread_id, "ingested": count}))
    else:
        print(f"✅ Ingested {count} messages into thread '{thread_id}'.")
    return 0


def cmd_context(engine: EngramEngine, args: argparse.Namespace) -> int:
    """Print the injectable context string for a thread."""
    thread_id = args.thread
    ctx_str = engine.build_system_context(thread_id)
    if not ctx_str:
        print(f"No context available for thread '{thread_id}'.", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        ctx = engine.get_context(thread_id)
        print(json.dumps({
            "thread_id": thread_id,
            "context": ctx_str,
            "stats": ctx["stats"],
        }, ensure_ascii=False, indent=2))
    else:
        print(ctx_str)
    return 0


def cmd_auto(engine: EngramEngine, args: argparse.Namespace) -> int:
    """
    Multi-channel auto-runner: scan sessions, detect channels, ingest concurrently.

    Delegates to engram_auto.EngramAutoRunner using the config from engram.yaml.
    The *engine* argument is unused here (auto-runner builds its own per-thread engines).

    Phase 1 additions:
      --max-sessions N       Cap sessions processed per run (default 20).
      --max-run-seconds S    Soft deadline in seconds (default 120).
    """
    from engram_auto import EngramAutoRunner, DEFAULT_MAX_SESSIONS_PER_RUN, DEFAULT_MAX_RUN_SECONDS
    from lib.config import load_engram_config

    cfg_path = getattr(args, "config", None)
    if cfg_path:
        cfg_path = Path(cfg_path).expanduser()
    engram_cfg = load_engram_config(cfg_path)

    workspace = Path(args.workspace)

    max_sessions = getattr(args, "max_sessions", DEFAULT_MAX_SESSIONS_PER_RUN)
    max_run_seconds = getattr(args, "max_run_seconds", DEFAULT_MAX_RUN_SECONDS)

    runner = EngramAutoRunner(
        workspace=workspace,
        engram_cfg=engram_cfg,
        dry_run=getattr(args, "dry_run", False),
        max_sessions_per_run=max_sessions,
        max_run_seconds=max_run_seconds,
    )

    if getattr(args, "daemon", False):
        interval = getattr(args, "interval", 900)
        runner.run_daemon(interval_seconds=interval)
        return 0  # unreachable but satisfies type checker

    totals = runner.run_once()
    if totals:
        print("Ingestion summary:")
        for tid, count in sorted(totals.items()):
            print(f"  {tid}: {count} messages")
    else:
        print("Nothing to ingest (all sessions up to date).")
    return 0


def cmd_daemon(engine: EngramEngine, args: argparse.Namespace) -> int:
    """
    Daemon mode — read JSONL messages from stdin and auto-compress in real-time.

    Each line on stdin must be a JSON object::

        {"role": "user", "content": "Hello!", "timestamp": "12:00"}

    Special control lines:
        {"__cmd": "observe"}   — force observe
        {"__cmd": "reflect"}   — force reflect
        {"__cmd": "status"}    — print status
        {"__cmd": "quit"}      — exit daemon

    Output: status JSON object after each message.
    """
    thread_id = args.thread
    quiet = getattr(args, "quiet", False)

    if not quiet:
        print(
            f"Engram daemon started for thread '{thread_id}'. "
            f"Reading JSONL from stdin (Ctrl-D to exit).",
            file=sys.stderr,
        )
        print(
            f"Thresholds: observe={engine.observer_threshold} tokens, "
            f"reflect={engine.reflector_threshold} tokens",
            file=sys.stderr,
        )

    try:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                result = {"error": f"JSON parse error: {exc}", "line": raw_line}
                print(json.dumps(result, ensure_ascii=False))
                sys.stdout.flush()
                continue

            # Handle control commands
            cmd = obj.get("__cmd")
            if cmd == "observe":
                text = engine.observe(thread_id)
                print(json.dumps({
                    "action": "observe",
                    "done": text is not None,
                    "tokens": estimate_tokens(text) if text else 0,
                }, ensure_ascii=False))
                sys.stdout.flush()
                continue
            elif cmd == "reflect":
                text = engine.reflect(thread_id)
                print(json.dumps({
                    "action": "reflect",
                    "done": text is not None,
                    "tokens": estimate_tokens(text) if text else 0,
                }, ensure_ascii=False))
                sys.stdout.flush()
                continue
            elif cmd == "status":
                ctx = engine.get_context(thread_id)
                print(json.dumps({
                    "action": "status",
                    "stats": ctx["stats"],
                    "meta": ctx["meta"],
                }, ensure_ascii=False))
                sys.stdout.flush()
                continue
            elif cmd == "quit":
                break

            # Regular message
            role = obj.get("role", "user")
            content = obj.get("content", "")
            timestamp = obj.get("timestamp")
            if not content:
                continue

            status = engine.add_message(thread_id, role=role, content=content, timestamp=timestamp)
            print(json.dumps({"action": "add_message", **status}, ensure_ascii=False))
            sys.stdout.flush()

    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

    if not quiet:
        print("Engram daemon exiting.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _make_engine(workspace: Path, args: argparse.Namespace) -> EngramEngine:
    """Construct an EngramEngine from CLI args, engram.yaml, and environment.

    Priority: CLI args > engram.yaml > env vars > .env file > built-in defaults.
    """
    # Load unified config (handles .env, yaml, env-var overrides internally)
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        cfg_path = Path(cfg_path).expanduser()
    engram_cfg = load_engram_config(cfg_path)
    kwargs = engram_engine_kwargs(engram_cfg)

    # CLI threshold args override config (only if explicitly set by user)
    if getattr(args, "observer_threshold", None) is not None:
        arg_ot = args.observer_threshold
        if arg_ot != DEFAULT_OBSERVER_THRESHOLD:
            kwargs["observer_threshold"] = arg_ot
    if getattr(args, "reflector_threshold", None) is not None:
        arg_rt = args.reflector_threshold
        if arg_rt != DEFAULT_REFLECTOR_THRESHOLD:
            kwargs["reflector_threshold"] = arg_rt

    return EngramEngine(workspace_path=workspace, **kwargs)


# Import defaults so _make_engine can reference them
from lib.engram import DEFAULT_OBSERVER_THRESHOLD, DEFAULT_REFLECTOR_THRESHOLD


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram_cli.py",
        description="Engram — LLM-driven Observational Memory for claw-compactor",
    )
    parser.add_argument("workspace", help="Workspace root directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to engram.yaml / engram.json config file",
    )

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    _common = argparse.ArgumentParser(add_help=False)
    _common.add_argument("--thread", required=True, help="Thread identifier")
    _common.add_argument("--json", action="store_true", help="JSON output")

    _thresholds = argparse.ArgumentParser(add_help=False)
    _thresholds.add_argument(
        "--observer-threshold", type=int, default=DEFAULT_OBSERVER_THRESHOLD,
        dest="observer_threshold",
    )
    _thresholds.add_argument(
        "--reflector-threshold", type=int, default=DEFAULT_REFLECTOR_THRESHOLD,
        dest="reflector_threshold",
    )

    sub.add_parser("observe", parents=[_common, _thresholds],
                   help="Force Observer run for a thread")
    sub.add_parser("reflect", parents=[_common, _thresholds],
                   help="Force Reflector run for a thread")

    p_status = sub.add_parser("status", help="Show thread status")
    p_status.add_argument("--thread", default=None, help="Thread (omit for all)")
    p_status.add_argument("--json", action="store_true")

    sub.add_parser("ingest", parents=[_common, _thresholds],
                   help="Import messages from JSON/JSONL file").add_argument(
        "--input", required=True, help="Input file path"
    )

    sub.add_parser("context", parents=[_common],
                   help="Print injectable system context")

    p_daemon = sub.add_parser("daemon", parents=[_thresholds],
                               help="Stdin daemon mode")
    p_daemon.add_argument("--thread", required=True)
    p_daemon.add_argument("--quiet", action="store_true")

    # --- auto subcommand ---
    from engram_auto import DEFAULT_MAX_SESSIONS_PER_RUN, DEFAULT_MAX_RUN_SECONDS
    p_auto = sub.add_parser(
        "auto",
        help="Multi-channel auto-runner (reads engram.yaml, processes sessions concurrently)",
    )
    p_auto.add_argument(
        "--daemon", action="store_true",
        help="Run continuously (default interval: 15 min)",
    )
    p_auto.add_argument(
        "--interval", type=int, default=900,
        help="Daemon interval in seconds (default: 900)",
    )
    p_auto.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Detect channels and convert but do not ingest",
    )
    p_auto.add_argument(
        "--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS_PER_RUN,
        dest="max_sessions",
        help=f"Max sessions to process per run (default: {DEFAULT_MAX_SESSIONS_PER_RUN})",
    )
    p_auto.add_argument(
        "--max-run-seconds", type=int, default=DEFAULT_MAX_RUN_SECONDS,
        dest="max_run_seconds",
        help=f"Soft deadline in seconds for a single run (default: {DEFAULT_MAX_RUN_SECONDS})",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)

    engine = _make_engine(workspace, args)

    handlers = {
        "observe": cmd_observe,
        "reflect": cmd_reflect,
        "status": cmd_status,
        "ingest": cmd_ingest,
        "context": cmd_context,
        "daemon": cmd_daemon,
        "auto": cmd_auto,
    }

    handler = handlers[args.command]
    sys.exit(handler(engine, args))


if __name__ == "__main__":
    main()
