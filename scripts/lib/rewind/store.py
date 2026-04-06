"""Rewind store: hash-addressed LRU cache enabling reversible compression.

When a compression stage (e.g. Ionizer) discards significant content, it
stores the original in RewindStore and embeds a hash marker in the compressed
output.  If the LLM later needs the full original, it calls the Rewind tool
with the marker ID — the store returns the original text from its LRU cache.

This gives the best of both worlds: aggressive compression for token savings,
with on-demand retrieval when the LLM determines it needs more detail.

Storage is bounded by max_entries (LRU eviction) and ttl_seconds (time-based
expiry).  Hash IDs are 24-char hex SHA-256 prefixes — collision probability
is negligible for the expected cache sizes.

Part of claw-compactor v7. License: MIT.
"""
from __future__ import annotations
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CacheEntry:
    original: str
    compressed: str
    stored_at: float
    original_tokens: int
    compressed_tokens: int


class RewindStore:
    """LRU store mapping hash IDs to original text for later retrieval."""

    def __init__(self, max_entries: int = 500, ttl_seconds: int = 600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

    def store(self, original: str, compressed: str,
              original_tokens: int = 0, compressed_tokens: int = 0) -> str:
        """Store original text and return a 24-char hex hash ID."""
        hash_id = hashlib.sha256(original.encode("utf-8")).hexdigest()[:24]
        entry = CacheEntry(
            original=original,
            compressed=compressed,
            stored_at=time.monotonic(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
        )
        # Move to end (most recently used)
        if hash_id in self._cache:
            self._cache.move_to_end(hash_id)
        self._cache[hash_id] = entry
        # Evict oldest if over limit
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return hash_id

    def retrieve(self, hash_id: str) -> Optional[str]:
        """Retrieve original text by hash ID. Returns None if expired or missing."""
        entry = self._cache.get(hash_id)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > self.ttl_seconds:
            del self._cache[hash_id]
            return None
        self._cache.move_to_end(hash_id)
        return entry.original

    def search(self, hash_id: str, keywords: list[str]) -> Optional[str]:
        """Retrieve and filter original text by keywords. Returns matching lines."""
        original = self.retrieve(hash_id)
        if original is None:
            return None
        if not keywords:
            return original
        lines = original.split("\n")
        matched = [l for l in lines if any(kw.lower() in l.lower() for kw in keywords)]
        return "\n".join(matched) if matched else original

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> int:
        """Persist the current cache to a JSON file.

        Entries that have exceeded their TTL are not saved.  Returns the
        number of entries written.

        Args:
            path: Destination file path (will be created or overwritten).

        Returns:
            Number of entries persisted.
        """
        now = time.monotonic()
        entries: dict[str, dict] = {}
        for hash_id, entry in self._cache.items():
            age = now - entry.stored_at
            if age <= self.ttl_seconds:
                entries[hash_id] = {
                    "original": entry.original,
                    "compressed": entry.compressed,
                    "remaining_ttl": round(self.ttl_seconds - age, 1),
                    "original_tokens": entry.original_tokens,
                    "compressed_tokens": entry.compressed_tokens,
                }
        Path(path).write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(entries)

    def load(self, path: str | Path) -> int:
        """Restore cache entries from a previously saved JSON file.

        Loaded entries are merged into the current cache (existing entries
        are preserved).  Each loaded entry's TTL is set to its
        ``remaining_ttl`` from the save file — entries that would already
        be expired are skipped.

        Args:
            path: Source file path.

        Returns:
            Number of entries loaded.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded = 0
        now = time.monotonic()
        for hash_id, blob in data.items():
            remaining = blob.get("remaining_ttl", 0)
            if remaining <= 0:
                continue
            if hash_id in self._cache:
                continue  # don't overwrite live entries
            entry = CacheEntry(
                original=blob["original"],
                compressed=blob.get("compressed", ""),
                stored_at=now - (self.ttl_seconds - remaining),
                original_tokens=blob.get("original_tokens", 0),
                compressed_tokens=blob.get("compressed_tokens", 0),
            )
            self._cache[hash_id] = entry
            loaded += 1
        # Enforce max_entries after load
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return loaded
