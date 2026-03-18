"""Ionizer — JSON/structured data compression FusionStage.

Performs statistical sampling on JSON arrays of dicts or strings.
Error/exception items are always preserved. Uses RewindStore for reversibility.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import json
import random
from typing import Any

from lib.fusion.base import FusionStage, FusionContext, FusionResult
from lib.rewind.marker import embed_marker
from lib.rewind.store import RewindStore
from lib.tokens import estimate_tokens

# Minimum array length before sampling is considered worthwhile.
_MIN_ARRAY_LEN = 5

# Number of items to keep from the front and back of a dict array.
_FRONT_BACK_K = 3

# Maximum array length before we start sampling dict arrays.
_SAMPLE_THRESHOLD = 20

# For large string arrays, keep at most this many unique entries.
_MAX_UNIQUE_STRINGS = 30

# Keywords that flag an item as an error record that must be preserved.
_ERROR_KEYWORDS = frozenset({"error", "exception", "failed", "failure", "fatal"})


def _item_is_error(item: dict) -> bool:
    """Return True if any value in *item* contains an error keyword."""
    for v in item.values():
        if isinstance(v, str):
            lowered = v.lower()
            if any(kw in lowered for kw in _ERROR_KEYWORDS):
                return True
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            continue
    # Also check keys
    for k in item:
        if any(kw in k.lower() for kw in _ERROR_KEYWORDS):
            return True
    return False


def _detect_id_fields(items: list[dict]) -> list[str]:
    """Heuristically detect ID-like field names in a list of dicts."""
    if not items:
        return []
    candidate_keys: list[str] = []
    first = items[0]
    for k in first:
        lower_k = k.lower()
        if lower_k in {"id", "uuid", "key", "name", "index", "seq", "sequence", "num", "number"}:
            candidate_keys.append(k)
        elif lower_k.endswith("_id") or lower_k.endswith("_key") or lower_k.endswith("_uuid"):
            candidate_keys.append(k)
    return candidate_keys


def _discover_schema(items: list[dict]) -> list[str]:
    """Return the union of all keys seen across items."""
    seen: dict[str, None] = {}
    for item in items:
        for k in item:
            seen[k] = None
    return list(seen.keys())


def _sample_dict_array(items: list[dict], k: int) -> list[dict]:
    """
    Sample a dict array:
      1. Always keep error items.
      2. Keep first K + last K items.
      3. Fill remaining budget with uniform random sample from the middle.
    """
    n = len(items)
    error_indices = {i for i, item in enumerate(items) if _item_is_error(item)}
    front_indices = set(range(min(k, n)))
    back_indices = set(range(max(0, n - k), n))

    protected = error_indices | front_indices | back_indices
    middle_indices = [i for i in range(n) if i not in protected]

    # Determine how many middle items to sample.
    total_budget = min(n, k * 2 + len(error_indices) + max(0, _SAMPLE_THRESHOLD - k * 2))
    middle_budget = max(0, total_budget - len(protected))

    if middle_indices and middle_budget > 0:
        sampled_middle = sorted(random.sample(middle_indices, min(middle_budget, len(middle_indices))))
    else:
        sampled_middle = []

    kept_indices = sorted(protected | set(sampled_middle))
    return [items[i] for i in kept_indices]


def _compress_dict_array(items: list[dict], rewind_store: RewindStore | None) -> tuple[str, str, int, int]:
    """
    Compress a JSON array of dicts. Returns (original_json, compressed_json,
    original_count, compressed_count).
    """
    original_json = json.dumps(items, indent=2)
    schema = _discover_schema(items)
    id_fields = _detect_id_fields(items)
    sampled = _sample_dict_array(items, _FRONT_BACK_K)

    compressed_json = json.dumps(sampled, indent=2)

    schema_comment = f"// Schema fields: {', '.join(schema)}"
    if id_fields:
        schema_comment += f" | ID fields: {', '.join(id_fields)}"

    header = f"{schema_comment}\n// Showing {len(sampled)} of {len(items)} items"
    result_text = f"{header}\n{compressed_json}"

    if rewind_store is not None:
        hash_id = rewind_store.store(
            original=original_json,
            compressed=result_text,
            original_tokens=estimate_tokens(original_json),
            compressed_tokens=estimate_tokens(result_text),
        )
        result_text = embed_marker(result_text, len(items), len(sampled), hash_id)

    return original_json, result_text, len(items), len(sampled)


def _compress_string_array(items: list[str], rewind_store: RewindStore | None) -> tuple[str, str, int, int]:
    """
    Compress a JSON array of strings via deduplication + sampling.
    """
    original_json = json.dumps(items, indent=2)

    # Deduplicate while preserving order.
    seen: dict[str, None] = {}
    for s in items:
        seen[s] = None
    unique = list(seen.keys())

    if len(unique) > _MAX_UNIQUE_STRINGS:
        kept = unique[:_FRONT_BACK_K] + unique[-_FRONT_BACK_K:]
        middle = unique[_FRONT_BACK_K: len(unique) - _FRONT_BACK_K]
        budget = max(0, _MAX_UNIQUE_STRINGS - _FRONT_BACK_K * 2)
        if middle and budget > 0:
            kept += random.sample(middle, min(budget, len(middle)))
        kept_sorted = sorted(set(range(len(unique))),
                             key=lambda i: unique[i] not in kept)
        unique = [u for u in unique if u in set(kept)]

    compressed_json = json.dumps(unique, indent=2)
    header = f"// {len(items) - len(unique)} duplicates removed. Showing {len(unique)} of {len(items)} items."
    result_text = f"{header}\n{compressed_json}"

    if rewind_store is not None:
        hash_id = rewind_store.store(
            original=original_json,
            compressed=result_text,
            original_tokens=estimate_tokens(original_json),
            compressed_tokens=estimate_tokens(result_text),
        )
        result_text = embed_marker(result_text, len(items), len(unique), hash_id)

    return original_json, result_text, len(items), len(unique)


class Ionizer(FusionStage):
    """JSON array statistical sampling compressor."""

    name = "ionizer"
    order = 15

    def __init__(self, rewind_store: RewindStore | None = None) -> None:
        self._rewind_store = rewind_store

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "json"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        content = ctx.content.strip()

        # Attempt to parse the JSON.
        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError as exc:
            return FusionResult(
                content=ctx.content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                warnings=[f"Ionizer: JSON parse error — {exc}"],
                skipped=True,
            )

        # Only operate on arrays.
        if not isinstance(data, list):
            return FusionResult(
                content=ctx.content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                skipped=True,
            )

        # Skip small arrays.
        if len(data) < _MIN_ARRAY_LEN:
            return FusionResult(
                content=ctx.content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                skipped=True,
            )

        markers: list[str] = []

        # Dispatch based on element type.
        if data and all(isinstance(item, dict) for item in data):
            _, compressed, orig_count, comp_count = _compress_dict_array(data, self._rewind_store)
            markers.append(f"ionizer:dict_array:{orig_count}->{comp_count}")
        elif data and all(isinstance(item, str) for item in data):
            _, compressed, orig_count, comp_count = _compress_string_array(data, self._rewind_store)
            markers.append(f"ionizer:string_array:{orig_count}->{comp_count}")
        else:
            # Mixed or unsupported array — skip.
            return FusionResult(
                content=ctx.content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                skipped=True,
            )

        compressed_tokens = estimate_tokens(compressed)
        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )
