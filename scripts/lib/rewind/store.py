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
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CacheEntry:
    original: str
    compressed: str
    stored_at: float
    original_tokens: int
    compressed_tokens: int
    ttl_override: int | None = None


class RewindStore:
    """LRU store mapping hash IDs to original text for later retrieval."""

    def __init__(self, max_entries: int = 500, ttl_seconds: int = 600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

    def store(self, original: str, compressed: str,
              original_tokens: int = 0, compressed_tokens: int = 0,
              ttl: int | None = None) -> str:
        """Store original text and return a 24-char hex hash ID."""
        hash_id = hashlib.sha256(original.encode("utf-8")).hexdigest()[:24]
        entry = CacheEntry(
            original=original,
            compressed=compressed,
            stored_at=time.monotonic(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ttl_override=ttl,
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
        effective_ttl = entry.ttl_override if entry.ttl_override is not None else self.ttl_seconds
        if time.monotonic() - entry.stored_at > effective_ttl:
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
