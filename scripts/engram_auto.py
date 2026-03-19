#!/usr/bin/env python3
"""
engram_auto.py — Automated multi-channel Engram session ingestion.

Scans OpenClaw session JSONL files, detects which Discord channel / cron job /
subagent they belong to, converts them to Engram format, and ingests them
concurrently using a ThreadPoolExecutor.

Usage:
    python3 scripts/engram_auto.py [--config engram.yaml] [--workspace PATH]
                                   [--once | --daemon] [--dry-run]
                                   [--max-sessions N] [--max-run-seconds S]

Phase 1 refactor:
  - Per-run rate limiting (--max-sessions, default 20)
  - Soft-deadline support (--max-run-seconds, default 120s)
  - Structured summary: processed/skipped/failed/remaining_estimate
  - Stable thread IDs: channel-id → known-name mapping cached, avoiding
    thread-id drift from positional (name-first vs id-first) detection bugs
  - Channel name lookup takes priority over raw channel-ID fallback

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure scripts/ is on path when run directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from claw_compactor.config import load_engram_config, engram_engine_kwargs
from claw_compactor.engram import EngramEngine
from claw_compactor.engram_storage import EngramStorage

logger = logging.getLogger("engram_auto")


# ---------------------------------------------------------------------------
# Thread-ID detection
# ---------------------------------------------------------------------------

# Channel-name → canonical thread-id mapping
# Used to stabilise thread IDs even when channel ID appears in the text first.
_CHANNEL_NAME_MAP: Dict[str, str] = {
    "general": "discord-general",
    "open-compress": "discord-open-compress",
    "opencompress": "discord-open-compress",
    "aimm": "discord-aimm",
}

# Channel-ID → canonical thread-id mapping (populated from known channels).
# Populated from _CHANNEL_ID_MAP_STATIC and extended at runtime via
# detect_thread_id cache.  The keys are numeric Discord channel ID strings.
_CHANNEL_ID_NAME_MAP_STATIC: Dict[str, str] = {
    # Add known channel IDs here to get stable names even without a name match.
    # Format:  "<channel_id>": "<thread_id>"
    "1470169146539901001": "discord-general",
    "1476885945163714641": "discord-open-compress",
}

# Mutable runtime copy (allows tests / runtime to extend it)
_CHANNEL_ID_NAME_MAP: Dict[str, str] = dict(_CHANNEL_ID_NAME_MAP_STATIC)

_RE_CHANNEL_NAME = re.compile(r"#([\w][\w-]*)")
_RE_CHANNEL_ID = re.compile(r"channel[^\S\r\n]+id:(\d+)", re.IGNORECASE)
_RE_CRON = re.compile(r'cron job\s+"([^"]+)"', re.IGNORECASE)
_RE_SUBAGENT = re.compile(r"subagent", re.IGNORECASE)

# Thread-map cache file (relative to storage_base, populated at runtime)
_THREAD_MAP_FILENAME = ".thread-map.json"

# Default rate-limit / soft-deadline values (also used as CLI defaults)
DEFAULT_MAX_SESSIONS_PER_RUN: int = 20
DEFAULT_MAX_RUN_SECONDS: int = 120


def detect_thread_id(
    session_file: Path,
    thread_map_path: Optional[Path] = None,
) -> str:
    """
    Detect the thread/channel this session belongs to by inspecting its content.

    Detection priority (most to least specific):
      1. subagent keyword           → "subagent"
      2. cron job name              → "cron-{job_name}"
      3. Discord channel name known → _CHANNEL_NAME_MAP[name]  (e.g. "discord-general")
      4. Discord channel id + name  → resolve id via _CHANNEL_ID_NAME_MAP, then by name
      5. Discord channel id only    → "discord-channel-{id}"
      6. Discord channel name (generic) → "discord-{name}"
      7. fallback                   → "openclaw-main"

    Results are cached to *thread_map_path* (a JSON file) keyed by session stem.

    Args:
        session_file:     Path to the session JSONL file.
        thread_map_path:  Optional path to .thread-map.json cache file.

    Returns:
        Thread-ID string suitable for use as Engram thread identifier.
    """
    session_id = session_file.stem

    # --- Check cache first ---
    if thread_map_path is not None and thread_map_path.exists():
        try:
            cache: Dict[str, str] = json.loads(
                thread_map_path.read_text(encoding="utf-8")
            )
            if session_id in cache:
                return cache[session_id]
        except (json.JSONDecodeError, OSError):
            pass

    thread_id = _detect_thread_id_from_file(session_file)

    # --- Persist to cache ---
    if thread_map_path is not None:
        try:
            cache = {}
            if thread_map_path.exists():
                try:
                    cache = json.loads(thread_map_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            cache[session_id] = thread_id
            thread_map_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("detect_thread_id: cannot write cache %s: %s", thread_map_path, exc)

    return thread_id


def thread_priority(thread_id: str) -> int:
    """Lower number = higher scheduling priority in run_once()."""
    if thread_id == "openclaw-main":
        return 0
    if thread_id.startswith("discord-"):
        return 1
    if thread_id.startswith("cron-"):
        return 2
    if thread_id == "subagent":
        return 9
    return 3


def _make_run_id(prefix: str = "engram-auto") -> str:
    """Create a run id so logs can distinguish old residue vs current run."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{ts}"


def _detect_thread_id_from_file(session_file: Path) -> str:
    """
    Inner detection logic (no caching).

    Algorithm:
      Scan up to 20 user/system messages from the file and apply heuristics
      in priority order.  A single pass collects all candidate signals from
      each line and then applies the precedence rules at the end:

        Priority (highest → lowest):
          1. subagent keyword
          2. cron job name
          3. Known channel name (via _CHANNEL_NAME_MAP)
          4. Known channel ID (via _CHANNEL_ID_NAME_MAP)
          5. Unknown channel ID (→ "discord-channel-{id}")
          6. Unknown channel name (→ "discord-{name}")
          7. Fallback: "openclaw-main"

    This ordering prevents the "channel id fires before name" bug where
    '#general channel id:111' resolved to 'discord-channel-111' instead
    of 'discord-general'.
    """
    try:
        lines_checked = 0
        messages_checked = 0

        # Accumulated signals across all messages in this file
        found_subagent = False
        found_cron: Optional[str] = None
        found_known_name: Optional[str] = None     # mapped known channel name
        found_channel_id: Optional[str] = None     # raw channel id string
        found_generic_name: Optional[str] = None   # unknown channel name

        with session_file.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if messages_checked >= 20:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                lines_checked += 1
                if lines_checked > 400:
                    break

                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                role = ""
                text = ""

                if obj.get("type") == "message":
                    msg = obj.get("message", {})
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    text = _extract_text(content)
                elif "role" in obj:
                    role = obj.get("role", "")
                    text = _extract_text(obj.get("content", ""))
                else:
                    text = raw

                if role not in ("user", "system", ""):
                    continue

                messages_checked += 1
                if not text:
                    continue

                # --- subagent check (highest priority) ---
                if _RE_SUBAGENT.search(text):
                    found_subagent = True
                    break  # no need to scan further

                # --- cron job check ---
                if found_cron is None:
                    m = _RE_CRON.search(text)
                    if m:
                        found_cron = m.group(1).strip('"\'').strip()

                # --- Scan for #channel-name and channel id:N in same text block ---
                # Collect all channel names found in this message
                msg_channel_names: List[str] = []
                for ch_match in _RE_CHANNEL_NAME.finditer(text):
                    msg_channel_names.append(ch_match.group(1).lower())

                # Collect channel id if present
                ch_id_m = _RE_CHANNEL_ID.search(text)
                msg_channel_id: Optional[str] = ch_id_m.group(1) if ch_id_m else None

                # Resolve known channel names first
                for ch_name in msg_channel_names:
                    if ch_name in _CHANNEL_NAME_MAP and found_known_name is None:
                        found_known_name = _CHANNEL_NAME_MAP[ch_name]
                        break

                # Then try ID → name lookup
                if msg_channel_id is not None:
                    if msg_channel_id in _CHANNEL_ID_NAME_MAP and found_known_name is None:
                        found_known_name = _CHANNEL_ID_NAME_MAP[msg_channel_id]
                    elif found_channel_id is None:
                        found_channel_id = msg_channel_id

                # Generic name fallback (only if no known name found yet)
                if found_generic_name is None and found_known_name is None:
                    for ch_name in msg_channel_names:
                        if ch_name not in _CHANNEL_NAME_MAP:
                            found_generic_name = ch_name
                            break

    except OSError as exc:
        logger.warning("detect_thread_id: cannot read %s: %s", session_file, exc)
        return "openclaw-main"

    # --- Apply priority rules ---
    if found_subagent:
        return "subagent"
    if found_cron is not None:
        return f"cron-{found_cron}"
    if found_known_name is not None:
        return found_known_name
    if found_channel_id is not None:
        return f"discord-channel-{found_channel_id}"
    if found_generic_name is not None:
        return f"discord-{found_generic_name}"
    return "openclaw-main"


def _extract_text(content: object) -> str:
    """Flatten content (str, list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Session → Engram format conversion
# ---------------------------------------------------------------------------

def convert_session(session_file: Path, output_file: Path) -> int:
    """
    Convert an OpenClaw session JSONL to Engram-format JSONL.

    Returns the number of messages written.
    """
    count = 0
    with session_file.open("r", encoding="utf-8", errors="replace") as fin, \
            output_file.open("w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "message":
                continue

            msg = obj.get("message", {})
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            text = _extract_text(msg.get("content", ""))
            if not text.strip():
                continue

            out: Dict[str, object] = {"role": role, "content": text}
            ts = obj.get("timestamp", "")
            if ts:
                out["timestamp"] = ts

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    return count


# ---------------------------------------------------------------------------
# Per-thread lock registry
# ---------------------------------------------------------------------------

class _LockRegistry:
    """Thread-safe registry of per-thread-id locks."""

    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def get(self, thread_id: str) -> threading.Lock:
        with self._meta:
            if thread_id not in self._locks:
                self._locks[thread_id] = threading.Lock()
            return self._locks[thread_id]


# ---------------------------------------------------------------------------
# Structured run summary
# ---------------------------------------------------------------------------

class RunSummary:
    """Accumulated counters for a single run_once() call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.processed: int = 0
        self.skipped: int = 0
        self.failed: int = 0
        self.remaining_estimate: int = 0
        self.total_sessions: int = 0

    def inc_processed(self) -> None:
        with self._lock:
            self.processed += 1

    def inc_skipped(self) -> None:
        with self._lock:
            self.skipped += 1

    def inc_failed(self) -> None:
        with self._lock:
            self.failed += 1

    def set_remaining(self, n: int) -> None:
        with self._lock:
            self.remaining_estimate = n

    def to_dict(self, run_id: str = "") -> Dict:
        with self._lock:
            return {
                "run_id": run_id,
                "total_sessions": self.total_sessions,
                "processed": self.processed,
                "skipped": self.skipped,
                "failed": self.failed,
                "remaining_estimate": self.remaining_estimate,
            }


# ---------------------------------------------------------------------------
# Auto-runner
# ---------------------------------------------------------------------------

class EngramAutoRunner:
    """
    Scan sessions, detect channels, convert, and ingest concurrently.

    Args:
        workspace:           Workspace root (Engram stores data under memory/engram/).
        engram_cfg:          Output of load_engram_config().
        dry_run:             If True, detect and convert but do not call LLM or write.
        max_sessions_per_run:Max number of sessions to process per run_once() call.
                             Remaining sessions are deferred to the next run.
                             Default: DEFAULT_MAX_SESSIONS_PER_RUN (20).
        max_run_seconds:     Soft deadline in seconds for run_once().  When the
                             elapsed time exceeds this value, no new sessions are
                             started; in-flight ones are allowed to finish.
                             Default: DEFAULT_MAX_RUN_SECONDS (120).
    """

    def __init__(
        self,
        workspace: Path,
        engram_cfg: Dict,
        dry_run: bool = False,
        max_sessions_per_run: int = DEFAULT_MAX_SESSIONS_PER_RUN,
        max_run_seconds: int = DEFAULT_MAX_RUN_SECONDS,
    ) -> None:
        self.workspace = workspace
        self.cfg = engram_cfg
        self.dry_run = dry_run
        self.max_sessions_per_run = max_sessions_per_run
        self.max_run_seconds = max_run_seconds

        self.scan_dir = Path(engram_cfg["sessions"]["scan_dir"])
        self.max_age_hours: int = int(engram_cfg["sessions"].get("max_age_hours", 48))
        self.max_workers: int = int(engram_cfg["concurrency"].get("max_workers", 4))
        self.storage_base = Path(engram_cfg["storage"]["base_dir"])

        self._lock_reg = _LockRegistry()

        # Processed-sessions marker lives next to the storage root
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self._processed_marker = self.storage_base / ".processed_sessions"
        self._processed_cache: set = self._load_processed()
        self._processed_lock = threading.Lock()

        # Thread-map cache for detect_thread_id()
        self._thread_map_path = self.storage_base / _THREAD_MAP_FILENAME

        # Error-type dedup: only log each error class once per run
        self._reported_errors: set = set()
        self._error_counts: Dict[str, int] = {}
        self._reported_errors_lock = threading.Lock()

        # Engine kwargs (shared config; each thread constructs its own engine
        # instance to avoid cross-thread state issues)
        self._engine_kwargs = engram_engine_kwargs(engram_cfg)

    # ------------------------------------------------------------------ #
    # Processed-sessions bookkeeping                                      #
    # ------------------------------------------------------------------ #

    def _load_processed(self) -> set:
        if not self._processed_marker.exists():
            return set()
        return set(self._processed_marker.read_text(encoding="utf-8").splitlines())

    def _is_processed(self, cache_key: str) -> bool:
        with self._processed_lock:
            return cache_key in self._processed_cache

    def _mark_processed(self, cache_key: str) -> None:
        with self._processed_lock:
            if cache_key not in self._processed_cache:
                self._processed_cache.add(cache_key)
                with self._processed_marker.open("a", encoding="utf-8") as f:
                    f.write(cache_key + "\n")

    def _cleanup_processed_marker(self) -> None:
        """
        Prune the processed-sessions marker file, keeping only records from
        the last 7 days.  Records are expected to be in the form
        ``{session_id}:{mtime}`` where mtime is a Unix timestamp (integer).
        Lines that don't match the format are dropped.
        """
        if not self._processed_marker.exists():
            return
        cutoff = time.time() - 7 * 24 * 3600  # 7 days ago
        kept: List[str] = []
        for line in self._processed_marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                try:
                    if float(parts[1]) >= cutoff:
                        kept.append(line)
                    # else: drop (older than 7 days)
                except ValueError:
                    kept.append(line)  # unknown format — keep to be safe
            else:
                kept.append(line)
        with self._processed_lock:
            self._processed_marker.write_text("\n".join(kept) + "\n", encoding="utf-8")
            self._processed_cache = set(kept)

    def _report_error_once(
        self,
        error_type: str,
        message: str,
        run_id: Optional[str] = None,
    ) -> None:
        """Log an error once per type per run, while keeping per-type counters."""
        with self._reported_errors_lock:
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
            if error_type in self._reported_errors:
                return
            self._reported_errors.add(error_type)

        if run_id:
            logger.error("[run_id=%s] %s", run_id, message)
        else:
            logger.error(message)

    def _extract_ingest_error(self, status: object) -> Optional[str]:
        """
        Best-effort extraction of error text from batch_ingest() status.

        Backward-compatible with multiple status shapes:
          - {"error": "..."}
          - {"status": {"error": "..."}}
          - object.error
          - object.status.error

        Missing fields are treated as "no explicit error".
        """
        if status is None:
            return None

        def _from_obj(obj: object) -> Optional[str]:
            if obj is None:
                return None

            if isinstance(obj, dict):
                err = obj.get("error")
                if err:
                    return str(err)
                return _from_obj(obj.get("status"))

            err_attr = getattr(obj, "error", None)
            if err_attr:
                return str(err_attr)

            return _from_obj(getattr(obj, "status", None))

        try:
            return _from_obj(status)
        except Exception:  # noqa: BLE001
            # Unknown status shape: keep backward-compatible behavior.
            return None

    def _status_looks_stalled(self, status: object, observer_threshold: int) -> bool:
        """
        Detect suspicious "silent failure" states where observer should have run
        but status shows no progress and no explicit error.
        """
        if not isinstance(status, dict):
            return False

        if status.get("error"):
            return False

        try:
            pending_tokens = int(status.get("pending_tokens", 0))
        except (TypeError, ValueError):
            return False

        observed = status.get("observed")
        return pending_tokens >= observer_threshold and observed is False

    # ------------------------------------------------------------------ #
    # Session discovery                                                   #
    # ------------------------------------------------------------------ #

    def find_sessions(self) -> List[Path]:
        """Return session JSONL files modified within max_age_hours."""
        if not self.scan_dir.exists():
            logger.warning("Sessions dir not found: %s", self.scan_dir)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        sessions: List[Path] = []

        for p in sorted(self.scan_dir.rglob("*.jsonl")):
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                if mtime >= cutoff:
                    sessions.append(p)
            except OSError:
                pass

        logger.info("Found %d recent session file(s) in %s", len(sessions), self.scan_dir)
        return sessions

    # ------------------------------------------------------------------ #
    # Per-session processing                                              #
    # ------------------------------------------------------------------ #

    def _process_session(
        self,
        session_file: Path,
        tmp_dir: Path,
        thread_id_hint: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[str, str, int, str]:
        """
        Process a single session file.

        Returns:
            (session_id, thread_id, messages_ingested, outcome)
            outcome ∈ {"processed", "skipped", "failed"}
        """
        session_id = session_file.stem
        mtime = int(session_file.stat().st_mtime)
        cache_key = f"{session_id}:{mtime}"
        run_label = run_id or "n/a"

        if self._is_processed(cache_key):
            logger.debug("[run_id=%s] Skip (unchanged): %s", run_label, session_id)
            return session_id, "", 0, "skipped"

        # Detect channel (with thread-map cache)
        thread_id = thread_id_hint or detect_thread_id(
            session_file,
            thread_map_path=self._thread_map_path,
        )
        logger.info("[run_id=%s] Session %s → thread '%s'", run_label, session_id, thread_id)

        if self.dry_run:
            self._mark_processed(cache_key)
            return session_id, thread_id, 0, "processed"

        # Convert
        tmp_out = tmp_dir / f"{session_id}.jsonl"
        msg_count = convert_session(session_file, tmp_out)
        if msg_count == 0:
            logger.info("[run_id=%s]   No messages extracted from %s", run_label, session_id)
            self._mark_processed(cache_key)
            return session_id, thread_id, 0, "processed"

        # Ingest (with per-thread lock to protect file writes)
        lock = self._lock_reg.get(thread_id)

        # Compute workspace root from storage base:
        # storage_base = {workspace}/memory/engram  OR an absolute custom path.
        # EngramEngine wants workspace_path such that it appends memory/engram/ itself.
        # So workspace = storage_base.parent.parent if using default layout,
        # but to be safe we always pass self.workspace.
        engine = EngramEngine(workspace_path=self.workspace, **self._engine_kwargs)

        # Read messages
        messages: List[Dict] = []
        try:
            with tmp_out.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if raw:
                        try:
                            messages.append(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
        except OSError as exc:
            self._report_error_once(
                "read_converted",
                f"  Cannot read converted file {tmp_out}: {exc}",
                run_id=run_label,
            )
            return session_id, thread_id, 0, "failed"

        # Write to storage under lock using batch_ingest for efficiency
        ingest_status: object = None
        with lock:
            valid_messages = [m for m in messages if m.get("content")]
            if valid_messages:
                ingest_status = engine.batch_ingest(thread_id, valid_messages)
            ingested = len(valid_messages)

        ingest_error = self._extract_ingest_error(ingest_status)
        if ingest_error:
            logger.error(
                "[run_id=%s]   ✗ Ingest failed (not marking processed): "
                "session=%s file=%s error=%s",
                run_label,
                session_id,
                session_file,
                ingest_error,
            )
            return session_id, thread_id, 0, "failed"

        if self._status_looks_stalled(ingest_status, observer_threshold=engine.observer_threshold):
            logger.error(
                "[run_id=%s]   ✗ Ingest status looks stalled (not marking processed): "
                "session=%s thread=%s pending_tokens=%s pending_tokens_after=%s observed=%s",
                run_label,
                session_id,
                thread_id,
                ingest_status.get("pending_tokens") if isinstance(ingest_status, dict) else "?",
                ingest_status.get("pending_tokens_after") if isinstance(ingest_status, dict) else "?",
                ingest_status.get("observed") if isinstance(ingest_status, dict) else "?",
            )
            return session_id, thread_id, 0, "failed"

        self._mark_processed(cache_key)
        logger.info(
            "[run_id=%s]   ✓ Ingested %d messages into thread '%s'",
            run_label,
            ingested,
            thread_id,
        )
        return session_id, thread_id, ingested, "processed"

    # ------------------------------------------------------------------ #
    # Run                                                                 #
    # ------------------------------------------------------------------ #

    def run_once(self) -> Dict[str, int]:
        """
        Process pending sessions concurrently, respecting rate limit and deadline.

        Rate limiting:
          - At most *max_sessions_per_run* sessions are started per call.
            Sessions are sorted by priority (main > discord > cron > subagent).
            Deferred sessions are counted in the structured summary as
            remaining_estimate.

        Soft deadline:
          - When *max_run_seconds* elapses, no new sessions are submitted to
            the executor; in-flight ones are allowed to finish gracefully.
            Deferred sessions are included in remaining_estimate.

        Returns a dict mapping thread_id → total messages ingested.
        Also prints a structured summary via _print_summary().
        """
        run_id = _make_run_id()
        run_start = time.monotonic()
        logger.info(
            "Engram auto run started (run_id=%s, max_sessions=%d, max_run_seconds=%d)",
            run_id,
            self.max_sessions_per_run,
            self.max_run_seconds,
        )

        sessions = self.find_sessions()

        # Reset per-run error dedup and counters
        with self._reported_errors_lock:
            self._reported_errors.clear()
            self._error_counts.clear()

        # Prune old processed-session records (keep last 7 days)
        self._cleanup_processed_marker()

        summary = RunSummary()
        summary.total_sessions = len(sessions)

        if not sessions:
            logger.info("[run_id=%s] No recent sessions to process.", run_id)
            result = summary.to_dict(run_id)
            self._print_summary(result, run_id)
            return {}

        totals: Dict[str, int] = {}
        totals_lock = threading.Lock()

        with tempfile.TemporaryDirectory(prefix="engram_auto_") as tmp_str:
            tmp_dir = Path(tmp_str)

            # Pre-detect thread for prioritisation and skip unchanged sessions
            jobs: List[Tuple[Path, str]] = []
            pre_skipped = 0
            for sf in sessions:
                session_id = sf.stem
                try:
                    cache_key = f"{session_id}:{int(sf.stat().st_mtime)}"
                except OSError as exc:
                    self._report_error_once(
                        "session_stat",
                        f"Cannot stat session file {sf}: {exc}",
                        run_id=run_id,
                    )
                    summary.inc_failed()
                    continue

                if self._is_processed(cache_key):
                    summary.inc_skipped()
                    pre_skipped += 1
                    continue

                tid = detect_thread_id(sf, thread_map_path=self._thread_map_path)
                jobs.append((sf, tid))

            # Sort by priority
            jobs.sort(key=lambda item: (thread_priority(item[1]), item[0].name))

            # --- Rate limiting: cap at max_sessions_per_run ---
            if len(jobs) > self.max_sessions_per_run:
                deferred = len(jobs) - self.max_sessions_per_run
                summary.set_remaining(deferred)
                logger.info(
                    "[run_id=%s] Rate limit: processing %d of %d pending sessions "
                    "(%d deferred to next run)",
                    run_id,
                    self.max_sessions_per_run,
                    len(jobs),
                    deferred,
                )
                jobs = jobs[: self.max_sessions_per_run]
            else:
                summary.set_remaining(0)

            # --- Submit to executor with soft-deadline enforcement ---
            deadline_hit = False
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="engram"
            ) as executor:
                futures: Dict[concurrent.futures.Future, Path] = {}

                for sf, tid in jobs:
                    elapsed = time.monotonic() - run_start
                    if elapsed >= self.max_run_seconds:
                        # Soft deadline reached — count remaining jobs as deferred
                        deferred_now = len(jobs) - len(futures)
                        with totals_lock:
                            current_remaining = summary.remaining_estimate
                            summary.set_remaining(current_remaining + deferred_now)
                        deadline_hit = True
                        logger.warning(
                            "[run_id=%s] Soft deadline reached after %.1fs; "
                            "%d session(s) deferred to next run.",
                            run_id,
                            elapsed,
                            deferred_now,
                        )
                        break

                    fut = executor.submit(self._process_session, sf, tmp_dir, tid, run_id)
                    futures[fut] = sf

                for fut in concurrent.futures.as_completed(futures):
                    sf = futures[fut]
                    try:
                        _, thread_id, count, outcome = fut.result()
                        if outcome == "processed":
                            summary.inc_processed()
                        elif outcome == "skipped":
                            summary.inc_skipped()
                        else:
                            summary.inc_failed()
                        if outcome == "processed" and thread_id and count > 0:
                            with totals_lock:
                                totals[thread_id] = totals.get(thread_id, 0) + count
                    except Exception as exc:  # noqa: BLE001
                        err_type = type(exc).__name__
                        self._report_error_once(
                            f"process_session:{err_type}",
                            f"Error processing {sf}: {exc}",
                            run_id=run_id,
                        )
                        summary.inc_failed()

        # --- Build and print structured summary ---
        result = summary.to_dict(run_id)
        self._print_summary(result, run_id, totals=totals)

        with self._reported_errors_lock:
            error_counts_snapshot = dict(self._error_counts)
        if error_counts_snapshot:
            ordered = ", ".join(
                f"{k}={error_counts_snapshot[k]}" for k in sorted(error_counts_snapshot)
            )
            print(f"Error summary [{run_id}]: {ordered}")

        return totals

    def _print_summary(
        self,
        summary: Dict,
        run_id: str,
        totals: Optional[Dict[str, int]] = None,
    ) -> None:
        """Print structured run summary to stdout."""
        print(
            f"Run summary [{run_id}]: "
            f"processed={summary['processed']} "
            f"skipped={summary['skipped']} "
            f"failed={summary['failed']} "
            f"remaining_estimate={summary['remaining_estimate']}"
        )

        if totals:
            # Show pending token counts per thread
            storage = EngramStorage(self.workspace)
            from claw_compactor.engram_utils import count_messages_tokens as _count_messages_tokens
            print("Thread pending tokens:")
            for tid in sorted(totals.keys()):
                pending = storage.read_pending(tid)
                pt = _count_messages_tokens(pending)
                print(f"  {tid}: {totals[tid]} new msgs ingested, {pt} pending tokens")

    def run_daemon(self, interval_seconds: int = 900) -> None:
        """Run run_once() in a loop, sleeping *interval_seconds* between runs."""
        logger.info("Engram daemon started (interval=%ds)", interval_seconds)
        while True:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("run_once error: %s", exc)
            logger.info("Sleeping %ds until next run…", interval_seconds)
            time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Engram status helper
# ---------------------------------------------------------------------------

def print_status(workspace: Path, engram_cfg: Dict) -> None:
    """Print Engram status for all known threads."""
    from claw_compactor.engram_storage import EngramStorage
    storage = EngramStorage(workspace)
    threads = storage.list_threads()
    if not threads:
        print("No Engram threads found.")
        return
    print(f"{'Thread':<28} {'Pending':>7} {'Obs tok':>8} {'Ref tok':>8} {'Total':>8}")
    print("─" * 65)
    from claw_compactor.tokens import estimate_tokens
    for tid in threads:
        pending = storage.read_pending(tid)
        obs = storage.read_observations(tid)
        ref = storage.read_reflection(tid)
        from claw_compactor.engram_utils import count_messages_tokens as _count_messages_tokens
        pt = _count_messages_tokens(pending)
        ot = estimate_tokens(obs)
        rt = estimate_tokens(ref)
        print(f"{tid:<28} {len(pending):>7} {ot:>8,} {rt:>8,} {pt+ot+rt:>8,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engram_auto.py",
        description="Engram Auto-Runner — multi-channel concurrent session ingestion",
    )
    p.add_argument(
        "--workspace",
        default=None,
        help="Workspace root (default: auto-detected from config storage.base_dir)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to engram.yaml / engram.json (default: auto-detect)",
    )
    p.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously (every 15 minutes)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Daemon sleep interval in seconds (default: 900)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Detect channels and convert but do not ingest",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print Engram thread status and exit",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS_PER_RUN,
        dest="max_sessions",
        help=f"Max sessions to process per run (default: {DEFAULT_MAX_SESSIONS_PER_RUN})",
    )
    p.add_argument(
        "--max-run-seconds",
        type=int,
        default=DEFAULT_MAX_RUN_SECONDS,
        dest="max_run_seconds",
        help=f"Soft deadline in seconds for a single run (default: {DEFAULT_MAX_RUN_SECONDS})",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg_path = Path(args.config).expanduser() if args.config else None
    engram_cfg = load_engram_config(cfg_path)

    # Derive workspace from storage base dir
    # storage.base_dir = {workspace}/memory/engram  (by convention)
    # So workspace = storage_base.parent.parent
    storage_base = Path(engram_cfg["storage"]["base_dir"])
    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
    else:
        # If storage base follows the convention, go up two levels; otherwise use cwd
        if storage_base.name == "engram" and storage_base.parent.name == "memory":
            workspace = storage_base.parent.parent
        else:
            workspace = Path.cwd()

    if args.status:
        print_status(workspace, engram_cfg)
        return

    runner = EngramAutoRunner(
        workspace=workspace,
        engram_cfg=engram_cfg,
        dry_run=args.dry_run,
        max_sessions_per_run=args.max_sessions,
        max_run_seconds=args.max_run_seconds,
    )

    if args.daemon:
        runner.run_daemon(interval_seconds=args.interval)
    else:
        totals = runner.run_once()
        if totals:
            print("Ingestion summary:")
            for tid, count in sorted(totals.items()):
                print(f"  {tid}: {count} messages")
        else:
            print("Nothing to ingest (all sessions up to date).")


if __name__ == "__main__":
    main()
