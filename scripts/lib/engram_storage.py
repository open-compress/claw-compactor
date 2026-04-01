"""
engram_storage.py — File-system storage backend for Engram (Observational Memory).

Layout under base_path/memory/engram/{thread_id}/:
    observations.md   — append-only observation log (Observer output)
    reflections.md    — latest reflection (Reflector output, overwritten each run)
    pending.jsonl     — raw pending messages not yet observed (JSONL, append-only)
    meta.json         — per-thread statistics and timestamps

All writes use atomic rename (tempfile + os.replace) to avoid partial-write
corruption even on crash.

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


class EngramStorage:
    """
    File-system storage for Engram's three-layer memory.

    Args:
        base_path: Workspace root directory. Engram data lives at
                   ``{base_path}/memory/engram/{thread_id}/``.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _thread_dir(self, thread_id: str) -> Path:
        """Return (and create) the directory for a thread."""
        if not thread_id or "/" in thread_id or "\\" in thread_id or ".." in thread_id:
            raise ValueError(f"Invalid thread_id: {thread_id!r}")
        d = self.base_path / "memory" / "engram" / thread_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _obs_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "observations.md"

    def _ref_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "reflections.md"

    def _pending_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "pending.jsonl"

    def _meta_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "meta.json"

    # ------------------------------------------------------------------
    # Observations (append-only Markdown)
    # ------------------------------------------------------------------

    def append_observation(
        self,
        thread_id: str,
        observation: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Append a new observation block to the thread's observation log.

        A separator header is prepended so multiple Observer runs are
        distinguishable.

        Args:
            thread_id:   Thread identifier.
            observation: Observation text from the Observer LLM.
            timestamp:   Optional ISO timestamp; defaults to UTC now.
        """
        ts = timestamp or _now_utc()
        header = f"\n<!-- observed_at: {ts} -->\n"
        content = header + observation.strip() + "\n"

        path = self._obs_path(thread_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)

        self._update_meta(thread_id, last_observed_at=ts)

    def read_observations(self, thread_id: str) -> str:
        """Read the full observation log for a thread (empty string if none)."""
        path = self._obs_path(thread_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Reflections (overwrite each run)
    # ------------------------------------------------------------------

    def write_reflection(
        self,
        thread_id: str,
        reflection: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Write (overwrite) the reflection for a thread using atomic rename.

        Args:
            thread_id:  Thread identifier.
            reflection: Reflection text from the Reflector LLM.
            timestamp:  Optional ISO timestamp; defaults to UTC now.
        """
        ts = timestamp or _now_utc()
        header = f"<!-- reflected_at: {ts} -->\n"
        content = header + reflection.strip() + "\n"

        path = self._ref_path(thread_id)
        _atomic_write(path, content)
        self._update_meta(thread_id, last_reflected_at=ts)

    def read_reflection(self, thread_id: str) -> str:
        """Read the latest reflection for a thread (empty string if none)."""
        path = self._ref_path(thread_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Pending messages (JSONL, append-only)
    # ------------------------------------------------------------------

    def append_message(self, thread_id: str, message: dict) -> None:
        """
        Append a raw message dict to the pending queue.

        Args:
            thread_id: Thread identifier.
            message:   Dict with at least ``"role"`` and ``"content"``.
        """
        path = self._pending_path(thread_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def read_pending(self, thread_id: str) -> List[dict]:
        """
        Read all pending messages for a thread.

        Returns:
            List of message dicts in append order.
        """
        path = self._pending_path(thread_id)
        if not path.exists():
            return []
        messages: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip corrupted lines
        return messages

    def clear_pending(self, thread_id: str) -> None:
        """
        Truncate the pending queue (called after a successful observe run).

        Args:
            thread_id: Thread identifier.
        """
        path = self._pending_path(thread_id)
        if path.exists():
            path.write_text("", encoding="utf-8")
        self._update_meta(thread_id, pending_count=0)

    def pending_count(self, thread_id: str) -> int:
        """Return the number of pending messages (counts lines without parsing)."""
        path = self._pending_path(thread_id)
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def read_meta(self, thread_id: str) -> dict:
        """
        Read thread metadata.

        Returns:
            Metadata dict, or a minimal default dict if none exists.
        """
        path = self._meta_path(thread_id)
        if not path.exists():
            return {"thread_id": thread_id, "created_at": None}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"thread_id": thread_id}

    def _update_meta(self, thread_id: str, **kwargs: object) -> None:
        """Merge *kwargs* into thread metadata and persist atomically."""
        meta = self.read_meta(thread_id)
        if not meta.get("created_at"):
            meta["created_at"] = datetime.now(timezone.utc).isoformat()
        meta["thread_id"] = thread_id
        meta.update(kwargs)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(
            self._meta_path(thread_id),
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    # ------------------------------------------------------------------
    # Thread discovery
    # ------------------------------------------------------------------

    def list_threads(self) -> List[str]:
        """Return sorted list of all known thread IDs."""
        engram_dir = self.base_path / "memory" / "engram"
        if not engram_dir.exists():
            return []
        return sorted(
            d.name
            for d in engram_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via tempfile + os.replace."""
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp_engram_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
