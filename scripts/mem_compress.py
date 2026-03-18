#!/usr/bin/env python3
"""Unified entry point for claw-compactor skill.

Usage:
    python3 mem_compress.py <workspace> <command> [options]

Commands:
    compress    Rule-based compression of memory files
    estimate    Token count estimation
    dedup       Cross-file duplicate detection
    tiers       Generate tiered summaries
    audit       Workspace memory health check
    observe     Compress session transcripts into observations
    dict        Dictionary-based compression
    optimize    Tokenizer-level format optimization
    full        Run complete pipeline (all steps in order)
    benchmark   Performance report with before/after stats"""

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root (for claw_compactor symlink/package) and scripts/ are on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from claw_compactor.tokens import estimate_tokens, using_tiktoken
from claw_compactor.exceptions import FileNotFoundError_, MemCompressError


def _workspace_path(workspace: str) -> Path:
    """Validate and return workspace Path. Exits on error."""
    p = Path(workspace)
    if not p.exists():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)
    if not p.is_dir():
        print(f"Error: workspace is not a directory: {workspace}", file=sys.stderr)
        sys.exit(1)
    return p


def _count_tokens_in_workspace(workspace: Path) -> int:
    """Count total tokens in all .md files in workspace."""
    total = 0
    for f in sorted(workspace.glob("*.md")):
        total += estimate_tokens(f.read_text(encoding="utf-8", errors="replace"))
    mem_dir = workspace / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            total += estimate_tokens(f.read_text(encoding="utf-8", errors="replace"))
    return total


def _collect_md_files(workspace: Path) -> List[Path]:
    """Collect all .md files in workspace (root + memory/)."""
    files: List[Path] = []
    for f in sorted(workspace.glob("*.md")):
        files.append(f)
    mem_dir = workspace / "memory"
    if mem_dir.is_dir():
        for f in sorted(mem_dir.glob("*.md")):
            if not f.name.startswith('.'):
                files.append(f)
    return files


# ── Command handlers ─────────────────────────────────────────────


def cmd_estimate(workspace: Path, args) -> int:
    """Estimate token counts for workspace files."""
    from estimate_tokens import scan_path, format_human
    files = _collect_md_files(workspace)
    if not files:
        print("No markdown files found.", file=sys.stderr)
        return 1
    results = scan_path(str(workspace), threshold=getattr(args, 'threshold', 0))
    if args.json:
        print(json.dumps({"files": results, "total_tokens": sum(r["tokens"] for r in results)}, indent=2))
    else:
        print(format_human(results))
    return 0


def cmd_compress(workspace: Path, args) -> int:
    """Run rule-based compression on workspace files."""
    from compress_memory import compress_file, _collect_files
    dry_run = getattr(args, 'dry_run', False)
    older_than = getattr(args, 'older_than', None)

    files = _collect_files(str(workspace), older_than=older_than)
    if not files:
        print("No files to compress.", file=sys.stderr)
        return 1

    results = []
    for f in files:
        r = compress_file(f, dry_run=dry_run, no_llm=True)
        r["rule_reduction_pct"] = round(
            (r["original_tokens"] - r["rule_compressed_tokens"]) / r["original_tokens"] * 100, 1
        ) if r["original_tokens"] > 0 else 0.0
        results.append(r)

    total_before = sum(r["original_tokens"] for r in results)
    total_after = sum(r["rule_compressed_tokens"] for r in results)
    total_saved = total_before - total_after

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            saved = r["original_tokens"] - r["rule_compressed_tokens"]
            print(f"{r['file']}: {r['original_tokens']} → {r['rule_compressed_tokens']} tokens (saved {saved})")
        print(f"\nTotal: {total_before} → {total_after} tokens (saved {total_saved})")
    return 0


def cmd_dedup(workspace: Path, args) -> int:
    """Find and report duplicate entries."""
    from dedup_memory import run_dedup, format_human
    threshold = getattr(args, 'threshold_val', 0.6)
    auto_merge = getattr(args, 'auto_merge', False)
    result = run_dedup(str(workspace), threshold=threshold, auto_merge=auto_merge)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_human(result))
    return 0


def cmd_tiers(workspace: Path, args) -> int:
    """Generate tiered summaries."""
    from generate_summary_tiers import generate_tiers, format_human, _find_memory_files
    files = _find_memory_files(str(workspace))
    if not files:
        print("No memory files found.", file=sys.stderr)
        return 1
    result = generate_tiers(files)
    if args.json:
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
    return 0


def cmd_audit(workspace: Path, args) -> int:
    """Audit workspace memory health."""
    from audit_memory import audit_workspace, format_report
    stale_days = getattr(args, 'stale_days', 14)
    result = audit_workspace(str(workspace), stale_days=stale_days)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))
    return 0


def cmd_observe(workspace: Path, args) -> int:
    """Scan session transcripts and generate observations."""
    from observation_compressor import parse_session_jsonl, extract_tool_interactions, rule_extract_observations, format_observations_md

    # OpenClaw stores transcripts per-agent at ~/.openclaw/agents/<agentId>/sessions/
    sessions_base = os.path.expanduser("~/.openclaw/agents")
    sessions_dirs = []
    if os.path.isdir(sessions_base):
        for agent_id in os.listdir(sessions_base):
            agent_sessions = os.path.join(sessions_base, agent_id, "sessions")
            if os.path.isdir(agent_sessions):
                sessions_dirs.append(agent_sessions)
    if not sessions_dirs:
        print(f"No session directories found under {sessions_base}/*/sessions/", file=sys.stderr)
        return 1
    # Process all agents, not just the first (fixes multi-agent setups where
    # sorted order may put a low-traffic agent like 'anvil' before 'main')

    # Load tracker
    mem_dir = workspace / "memory"
    mem_dir.mkdir(exist_ok=True)
    tracker_path = mem_dir / ".observed-sessions.json"
    tracker: Dict[str, str] = {}
    if tracker_path.exists():
        try:
            tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            tracker = {}

    # Find session files across all agent dirs
    session_files = sorted(
        sf for d in sessions_dirs for sf in Path(d).glob("*.jsonl")
    )
    since = getattr(args, 'since', None)

    new_count = 0
    obs_dir = mem_dir / "observations"
    obs_dir.mkdir(exist_ok=True)

    for sf in session_files:
        if sf.name in tracker:
            continue

        # Apply --since filter
        if since:
            try:
                # Try to extract date from filename
                fname = sf.stem
                if fname < since:
                    continue
            except Exception:
                pass

        try:
            messages = parse_session_jsonl(sf)
            interactions = extract_tool_interactions(messages)
            if not interactions:
                tracker[sf.name] = datetime.now().isoformat()
                continue

            observations = rule_extract_observations(interactions)
            if observations:
                md = format_observations_md(observations)
                obs_file = obs_dir / f"{sf.stem}.md"
                obs_file.write_text(md, encoding="utf-8")
                new_count += 1

            tracker[sf.name] = datetime.now().isoformat()
        except Exception as e:
            print(f"Warning: failed to process {sf.name}: {e}", file=sys.stderr)

    # Save tracker
    tracker_path.write_text(json.dumps(tracker, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"processed": new_count, "total_tracked": len(tracker)}))
    else:
        print(f"Processed {new_count} new session(s), {len(tracker)} total tracked.")
    return 0


def cmd_dict(workspace: Path, args) -> int:
    """Dictionary-based compression."""
    from dictionary_compress import cmd_build, cmd_stats
    from claw_compactor.dictionary import save_codebook

    mem_dir = workspace / "memory"
    mem_dir.mkdir(exist_ok=True)
    cb_path = mem_dir / ".codebook.json"

    result = cmd_build(workspace, cb_path, min_freq=2)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Codebook: {result['codebook_entries']} entries from {result['files_scanned']} files")
        print(f"Saved to: {result['codebook_path']}")
    return 0


def cmd_optimize(workspace: Path, args) -> int:
    """Apply tokenizer-level format optimization."""
    from claw_compactor.tokenizer_optimizer import optimize_tokens, estimate_savings

    dry_run = getattr(args, 'dry_run', False)
    files = _collect_md_files(workspace)
    if not files:
        print("No files found.", file=sys.stderr)
        return 1

    total_before = 0
    total_after = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        optimized = optimize_tokens(text, aggressive=True)
        before = estimate_tokens(text)
        after = estimate_tokens(optimized)
        total_before += before
        total_after += after
        if not dry_run:
            f.write_text(optimized, encoding="utf-8")

    saved = total_before - total_after
    if args.json:
        print(json.dumps({
            "before": total_before,
            "after": total_after,
            "saved": saved,
            "files": len(files),
        }))
    else:
        print(f"Tokenizer optimization: {total_before} → {total_after} tokens (saved {saved})")
    return 0


def cmd_engram(workspace: Path, args) -> int:
    """Engram — LLM-driven Observational Memory (Layer 6)."""
    import subprocess
    engram_cli = Path(__file__).resolve().parent / "engram_cli.py"
    cmd_args = [sys.executable, str(engram_cli), str(workspace)]

    sub_command = getattr(args, 'engram_command', None)
    if sub_command:
        cmd_args.append(sub_command)
    else:
        cmd_args.append("status")

    thread = getattr(args, 'thread', None)
    if thread:
        cmd_args.extend(["--thread", thread])

    if getattr(args, 'json', False):
        cmd_args.append("--json")

    result = subprocess.run(cmd_args)
    return result.returncode


def cmd_full(workspace: Path, args) -> int:
    """Run complete compression pipeline."""
    from compress_memory import compress_file, _collect_files, rule_compress
    from dictionary_compress import cmd_build
    from dedup_memory import run_dedup
    from generate_summary_tiers import generate_tiers, _find_memory_files

    # 1. Count initial tokens
    before_tokens = _count_tokens_in_workspace(workspace)
    print(f"Before: {before_tokens:,} tokens")

    # 2. Observe (scan session transcripts)
    try:
        observe_args = argparse.Namespace(json=False, since=getattr(args, 'since', None))
        cmd_observe(workspace, observe_args)
    except Exception as e:
        print(f"  observe: skipped ({e})")

    # 2b. Engram status check (non-blocking — Layer 6)
    try:
        from claw_compactor.engram import EngramEngine
        engram_engine = EngramEngine(workspace_path=workspace)
        threads = engram_engine.storage.list_threads()
        if threads:
            total_engram_tokens = sum(
                engram_engine.get_context(t)["stats"]["total_tokens"]
                for t in threads
            )
            print(f"  engram: {len(threads)} thread(s), {total_engram_tokens:,} memory tokens")
        else:
            print("  engram: no threads yet (use 'engram' command to add)")
    except Exception as e:
        print(f"  engram: skipped ({e})")

    # 3. Compress (rule engine)
    files = _collect_files(str(workspace))
    for f in files:
        compress_file(f, dry_run=False, no_llm=True)
    print(f"  compress: processed {len(files)} files")

    # 4. Dict (dictionary compression)
    mem_dir = workspace / "memory"
    mem_dir.mkdir(exist_ok=True)
    cb_path = mem_dir / ".codebook.json"
    try:
        result = cmd_build(workspace, cb_path, min_freq=2)
        print(f"  dict: {result['codebook_entries']} entries")
    except Exception as e:
        print(f"  dict: skipped ({e})")

    # 5. Dedup (report only)
    try:
        dedup_result = run_dedup(str(workspace))
        print(f"  dedup: {dedup_result['duplicate_groups']} groups found")
    except Exception as e:
        print(f"  dedup: skipped ({e})")

    # 6. Tiers
    try:
        tier_files = _find_memory_files(str(workspace))
        if tier_files:
            tier_result = generate_tiers(tier_files)
            print(f"  tiers: {tier_result['total_sections']} sections analyzed")
    except Exception as e:
        print(f"  tiers: skipped ({e})")

    # 7. Final count
    after_tokens = _count_tokens_in_workspace(workspace)
    saved = before_tokens - after_tokens
    pct = (saved / before_tokens * 100) if before_tokens > 0 else 0

    if getattr(args, 'report', False):
        # Chat-friendly report for pasting into conversation
        print()
        print(f"🦞 **Claw Compactor — Compression Report**")
        print(f"   📁 Files processed: {len(files)}")
        print(f"   📊 Before: {before_tokens:,} tokens → After: {after_tokens:,} tokens")
        print(f"   💰 Saved: **{saved:,} tokens ({pct:.1f}%)**")
        cost = saved * 0.000015
        if cost > 0.01:
            print(f"   💵 Est. cost saved per call: ${cost:.4f}")
    else:
        print(f"After: {after_tokens:,} tokens")
        print(f"Tokens saved: {saved:,} ({pct:.0f}%)")
    return 0


def cmd_benchmark(workspace: Path, args) -> int:
    """Non-destructive performance benchmark."""
    from compress_memory import rule_compress
    from claw_compactor.dictionary import build_codebook, compress_text
    from claw_compactor.rle import compress as rle_compress
    from claw_compactor.tokenizer_optimizer import optimize_tokens

    files = _collect_md_files(workspace)
    if not files:
        if not args.json:
            print("No files found.", file=sys.stderr)
        return 1

    # Read all files
    texts = {}
    for f in files:
        texts[str(f)] = f.read_text(encoding="utf-8", errors="replace")
    combined = '\n'.join(texts.values())

    # Baseline
    baseline_tokens = estimate_tokens(combined)

    # Step 1: Rule engine
    rule_compressed = rule_compress(combined)
    rule_tokens = estimate_tokens(rule_compressed)

    # Step 2: Dictionary compress
    cb = build_codebook(list(texts.values()), min_freq=2)
    dict_compressed = compress_text(rule_compressed, cb)
    dict_tokens = estimate_tokens(dict_compressed)

    # Step 3: RLE
    ws_paths = [str(workspace)]
    rle_compressed = rle_compress(dict_compressed, ws_paths)
    rle_tokens = estimate_tokens(rle_compressed)

    # Step 4: Tokenizer optimize
    tok_optimized = optimize_tokens(rle_compressed, aggressive=True)
    tok_tokens = estimate_tokens(tok_optimized)

    steps = [
        {"name": "Rule Engine", "before": baseline_tokens, "after": rule_tokens},
        {"name": "Dictionary Compress", "before": rule_tokens, "after": dict_tokens},
        {"name": "RLE Patterns", "before": dict_tokens, "after": rle_tokens},
        {"name": "Tokenizer Optimize", "before": rle_tokens, "after": tok_tokens},
    ]
    for s in steps:
        s["saved"] = s["before"] - s["after"]
        s["pct"] = round((s["saved"] / s["before"] * 100), 1) if s["before"] > 0 else 0.0

    total_saved = baseline_tokens - tok_tokens
    total_pct = round((total_saved / baseline_tokens * 100), 1) if baseline_tokens > 0 else 0.0

    if args.json:
        print(json.dumps({
            "steps": steps,
            "total_before": baseline_tokens,
            "total_after": tok_tokens,
            "total_saved": total_saved,
            "total_pct": total_pct,
        }))
        return 0

    # Human report
    today = date.today().isoformat()
    print(f"=== claw-compactor Performance Report ===")
    print(f"Date: {today}")
    print(f"Engine: {'tiktoken' if using_tiktoken() else 'heuristic'}")
    print(f"Files: {len(files)}")
    print()
    print(f"{'Step':<22} | {'Before':>8} | {'After':>8} | {'Saved':>6} | {'%':>6}")
    print("-" * 58)
    for s in steps:
        print(f"{s['name']:<22} | {s['before']:>8,} | {s['after']:>8,} | {s['saved']:>6,} | {s['pct']:>5.1f}%")
    print("-" * 58)
    print(f"{'TOTAL (memory)':<22} | {baseline_tokens:>8,} | {tok_tokens:>8,} | {total_saved:>6,} | {total_pct:>5.1f}%")
    print()
    print(f"💰 Total savings: {total_saved:,} tokens ({total_pct:.1f}%)")
    print()

    # Session transcript info — scan all agent session dirs
    sessions_base = os.path.expanduser("~/.openclaw/agents")
    session_count = 0
    if os.path.isdir(sessions_base):
        for agent_id in os.listdir(sessions_base):
            agent_sessions = Path(sessions_base) / agent_id / "sessions"
            if agent_sessions.is_dir():
                session_count += len(list(agent_sessions.glob("*.jsonl")))
    print(f"Session Transcripts: {session_count} files found")
    print()

    # Recommendations
    print("Recommendations:")
    if total_pct < 5:
        print("  - Workspace is already well-optimized")
    else:
        if steps[0]["pct"] > 3:
            print("  - Run 'compress' to apply rule engine savings")
        if steps[1]["pct"] > 2:
            print("  - Run 'dict' to apply dictionary compression")
        if steps[3]["pct"] > 1:
            print("  - Run 'optimize' for tokenizer-level savings")
    if session_count > 0:
        print(f"  - Run 'observe' to compress {session_count} session transcript(s)")

    return 0


# ── Command map & parser ─────────────────────────────────────────

def cmd_install(workspace: Path, args) -> int:
    """Install auto-compression into HEARTBEAT.md and optionally set up cron."""
    script_path = Path(__file__).resolve()
    heartbeat_path = workspace / "HEARTBEAT.md"

    # Build the heartbeat entry — uses 'auto' for every-interaction compression + reporting
    entry = f"""
## Memory Compression (auto — openclaw-token-compressor)
- If savings >5%: auto-run `python3 {script_path} {workspace} full`
- If unprocessed session transcripts exist: run observe
- Record results to `memory/heartbeat-state.json`
"""

    # Check if already installed
    if heartbeat_path.exists():
        existing = heartbeat_path.read_text(encoding="utf-8")
        if "claw-compactor" in existing or "openclaw-token-compressor" in existing:
            print("✅ Already installed in HEARTBEAT.md")
        else:
            with open(heartbeat_path, "a", encoding="utf-8") as f:
                f.write(entry)
            print(f"✅ Installed into HEARTBEAT.md")
    else:
        with open(heartbeat_path, "w", encoding="utf-8") as f:
            f.write("# HEARTBEAT.md\n" + entry)
        print(f"✅ Created HEARTBEAT.md with auto-compression")

    # Update SKILL.md guidance for the agent
    skill_path = script_path.parent.parent / "SKILL.md"
    print(f"\n📋 Agent instructions (in SKILL.md):")
    print(f"   Every session start: run `python3 {script_path} {workspace} auto`")
    print(f"   Paste the report into chat so user sees savings each interaction.")
    print(f"\n⏰ Optional cron (hourly auto-compress):")
    print(f"   0 * * * * cd {script_path.parent} && python3 {script_path} {workspace} auto --json >> {workspace}/memory/.compactor-log.jsonl 2>&1")
    print(f"\n   Script: {script_path}")
    print(f"   Workspace: {workspace}")
    return 0


def cmd_auto(workspace: Path, args) -> int:
    """Auto-compress and report savings since last run. Designed for every-interaction use."""
    from compress_memory import compress_file, _collect_files

    tracker_path = workspace / "memory" / ".compactor-state.json"
    tracker_path.parent.mkdir(exist_ok=True)

    # Load previous state
    prev_state: Dict[str, Any] = {}
    if tracker_path.exists():
        try:
            prev_state = json.loads(tracker_path.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}

    prev_tokens = prev_state.get("file_tokens", {})
    prev_total = prev_state.get("total_tokens", 0)
    prev_time = prev_state.get("last_run", "never")

    # Collect current files and token counts BEFORE compression
    files = _collect_md_files(workspace)
    before_tokens: Dict[str, int] = {}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        before_tokens[str(f.relative_to(workspace))] = estimate_tokens(text)
    total_before = sum(before_tokens.values())

    # Run compression pipeline (rule engine + dict + rle + optimize)
    for f in files:
        try:
            compress_file(str(f), dry_run=False, no_llm=True)
        except Exception:
            pass

    # Re-count after compression
    after_tokens: Dict[str, int] = {}
    for f in files:
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="replace")
            after_tokens[str(f.relative_to(workspace))] = estimate_tokens(text)
    total_after = sum(after_tokens.values())

    # Calculate deltas
    this_run_saved = total_before - total_after
    since_last_saved = prev_total - total_after if prev_total > 0 else this_run_saved

    # Build per-file change report (top changers)
    changes = []
    for fname, after in after_tokens.items():
        before = before_tokens.get(fname, after)
        if before != after:
            changes.append({
                "file": fname,
                "before": before,
                "after": after,
                "saved": before - after,
                "pct": round((before - after) / before * 100, 1) if before > 0 else 0
            })
    changes.sort(key=lambda x: x["saved"], reverse=True)

    # Save state
    new_state = {
        "total_tokens": total_after,
        "file_tokens": after_tokens,
        "last_run": datetime.now().isoformat(),
        "runs": prev_state.get("runs", 0) + 1,
        "cumulative_saved": prev_state.get("cumulative_saved", 0) + this_run_saved,
    }
    tracker_path.write_text(json.dumps(new_state, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps({
            "total_before": total_before,
            "total_after": total_after,
            "this_run_saved": this_run_saved,
            "since_last_saved": since_last_saved,
            "cumulative_saved": new_state["cumulative_saved"],
            "runs": new_state["runs"],
            "last_run": prev_time,
            "changes": changes[:10],
        }, indent=2))
        return 0

    # Chat-friendly report
    print(f"🦞 Claw Compactor — Auto Report")
    print(f"   Last run: {prev_time}")
    print(f"   Files scanned: {len(files)}")
    print()
    if this_run_saved > 0:
        pct = round(this_run_saved / total_before * 100, 1) if total_before > 0 else 0
        print(f"   This run: {total_before:,} → {total_after:,} tokens ({this_run_saved:,} saved, {pct}% ↓)")
        if changes:
            print(f"   Top changes:")
            for c in changes[:5]:
                print(f"     {c['file']}: {c['before']:,} → {c['after']:,} ({c['saved']:,} saved, {c['pct']}%)")
    else:
        print(f"   No new savings this run — workspace already optimized ✅")
        print(f"   Current total: {total_after:,} tokens")

    if prev_total > 0 and since_last_saved != this_run_saved:
        print(f"   Since last run: {since_last_saved:+,} tokens")

    print(f"   Cumulative saved (all time): {new_state['cumulative_saved']:,} tokens across {new_state['runs']} runs")
    cost_saved = new_state['cumulative_saved'] * 0.000015  # ~$15/1M tokens Opus input
    if cost_saved > 0.01:
        print(f"   Estimated cost saved: ${cost_saved:.2f}")
    print()
    return 0


COMMAND_MAP = {
    "compress": cmd_compress,
    "estimate": cmd_estimate,
    "dedup": cmd_dedup,
    "tiers": cmd_tiers,
    "audit": cmd_audit,
    "observe": cmd_observe,
    "dict": cmd_dict,
    "optimize": cmd_optimize,
    "full": cmd_full,
    "benchmark": cmd_benchmark,
    "install": cmd_install,
    "auto": cmd_auto,
    "engram": cmd_engram,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="claw-compactor: workspace memory compression toolkit"
    )
    parser.add_argument("workspace", help="Workspace directory path")

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # Add -v to all subparsers via parent
    _common = argparse.ArgumentParser(add_help=False)
    _common.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # compress
    p = sub.add_parser("compress", help="Rule-based compression", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--older-than", type=int, default=None)

    # estimate
    p = sub.add_parser("estimate", help="Token estimation", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--threshold", type=int, default=0)

    # dedup
    p = sub.add_parser("dedup", help="Duplicate detection", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--auto-merge", action="store_true")
    p.add_argument("--threshold-val", type=float, default=0.6)

    # tiers
    p = sub.add_parser("tiers", help="Generate tiered summaries", parents=[_common])
    p.add_argument("--json", action="store_true")

    # audit
    p = sub.add_parser("audit", help="Workspace audit", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--stale-days", type=int, default=14)

    # observe
    p = sub.add_parser("observe", help="Compress session transcripts", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--since", type=str, default=None)

    # dict
    p = sub.add_parser("dict", help="Dictionary compression", parents=[_common])
    p.add_argument("--json", action="store_true")

    # optimize
    p = sub.add_parser("optimize", help="Tokenizer optimization", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # full
    p = sub.add_parser("full", help="Run complete pipeline", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--report", action="store_true", help="Output chat-friendly report")

    # benchmark
    p = sub.add_parser("benchmark", help="Performance benchmark", parents=[_common])
    p.add_argument("--json", action="store_true")

    # install
    sub.add_parser("install", help="Install heartbeat auto-compression", parents=[_common])

    # auto
    p = sub.add_parser("auto", help="Auto-compress and report savings (every interaction)", parents=[_common])
    p.add_argument("--json", action="store_true")

    # engram (Layer 6 — LLM-driven Observational Memory)
    p = sub.add_parser("engram", help="Engram Observational Memory (Layer 6)", parents=[_common])
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "engram_command",
        nargs="?",
        default="status",
        choices=["observe", "reflect", "status", "ingest", "context", "daemon"],
        help="Engram sub-command (default: status)",
    )
    p.add_argument("--thread", type=str, default=None, help="Thread identifier")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    workspace = _workspace_path(args.workspace)
    handler = COMMAND_MAP[args.command]
    sys.exit(handler(workspace, args))


if __name__ == "__main__":
    main()
