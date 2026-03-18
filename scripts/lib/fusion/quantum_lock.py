"""Quantum Lock — KV Cache alignment as a FusionStage.

Runs at order=3, just before Cortex (order=5), so that downstream stages
always receive a prefix-stable system message.

The Anthropic prompt cache keys on the first N tokens of the system prompt.
Any dynamic content (dates, UUIDs, API keys, JWTs, timestamps) that appears
near the top of a system message will bust the cache on every request.

QuantumLock solves this by:
  1. Detecting all dynamic fragments using regex patterns.
  2. Replacing each occurrence with a stable placeholder token.
  3. Appending a clearly delimited "dynamic context" block at the END of the
     message so the model still has access to the real values.

The "quantum" metaphor: dynamic values are collapsed into a deterministic
tail section so the wavefunction of the prefix stays locked (stable).

Part of claw-compactor Phase 5. License: MIT.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from lib.fusion.base import FusionContext, FusionResult, FusionStage
from lib.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Dynamic content patterns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicPattern:
    """A compiled pattern that identifies dynamic content."""
    name: str
    regex: re.Pattern
    placeholder: str


_RAW_PATTERNS: list[tuple[str, str, str]] = [
    # ISO 8601 date/datetime
    (
        "iso_date",
        r"\b\d{4}-\d{2}-\d{2}"
        r"(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b",
        "<date>",
    ),
    # Plain HH:MM:SS times
    (
        "time",
        r"\b\d{2}:\d{2}:\d{2}\b",
        "<time>",
    ),
    # JWTs (eyJ...) — header.payload.signature
    (
        "jwt",
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
        "<jwt>",
    ),
    # API keys: sk-..., rk-... OR pk_live_..., pk_test_... (Stripe-style underscore separator)
    (
        "api_key",
        r"\b(?:(?:sk|rk)-[A-Za-z0-9_-]{16,}|(?:pk_live|pk_test)_[A-Za-z0-9_-]{16,})\b",
        "<api_key>",
    ),
    # UUIDs (case-insensitive)
    (
        "uuid",
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "<uuid>",
    ),
    # Unix timestamps: 10-digit (seconds since ~2001) or 13-digit (ms)
    (
        "unix_ts",
        r"\b(?:1[5-9]\d{8}|[2-9]\d{9}|\d{13})\b",
        "<timestamp>",
    ),
    # High-entropy hex strings: 32–64 hex chars (request/trace/session IDs)
    (
        "hex_id",
        r"\b[0-9a-fA-F]{32,64}\b",
        "<id>",
    ),
]

DYNAMIC_PATTERNS: list[DynamicPattern] = [
    DynamicPattern(
        name=name,
        regex=re.compile(pattern),
        placeholder=placeholder,
    )
    for name, pattern, placeholder in _RAW_PATTERNS
]

APPENDIX_START = "<!-- quantum-lock: dynamic context -->"
APPENDIX_END = "<!-- end quantum-lock -->"
APPENDIX_SEPARATOR = "---"


# ---------------------------------------------------------------------------
# Public functions (usable standalone, not only as a FusionStage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicFragment:
    """A single dynamic fragment extracted from content."""
    name: str
    original: str
    placeholder: str
    indices: tuple[int, ...]  # positions in the original string


def extract_dynamic(content: str) -> list[DynamicFragment]:
    """Return all dynamic fragments found in *content*, sorted by first index.

    De-duplicates by original value: the same UUID appearing multiple times
    is reported once with all its positions.
    """
    seen: dict[str, DynamicFragment] = {}

    for dp in DYNAMIC_PATTERNS:
        for match in dp.regex.finditer(content):
            val = match.group(0)
            if val in seen:
                frag = seen[val]
                seen[val] = DynamicFragment(
                    name=frag.name,
                    original=frag.original,
                    placeholder=frag.placeholder,
                    indices=(*frag.indices, match.start()),
                )
            else:
                seen[val] = DynamicFragment(
                    name=dp.name,
                    original=val,
                    placeholder=dp.placeholder,
                    indices=(match.start(),),
                )

    return sorted(seen.values(), key=lambda f: f.indices[0])


def stabilize(content: str) -> str:
    """Stabilise *content* for KV cache alignment.

    Replaces dynamic fragments with placeholders and appends a
    "dynamic context" appendix at the end so the model still has
    access to the real values.

    Returns *content* unchanged if no dynamic fragments are found.
    """
    fragments = extract_dynamic(content)
    if not fragments:
        return content

    stabilized = content
    # Process longest originals first to avoid partial substitution
    for frag in sorted(fragments, key=lambda f: len(f.original), reverse=True):
        stabilized = stabilized.replace(frag.original, frag.placeholder)

    appendix_lines = [
        "",
        APPENDIX_SEPARATOR,
        APPENDIX_START,
    ]
    for frag in fragments:
        appendix_lines.append(f"{frag.name}: {frag.original}")
    appendix_lines.append(APPENDIX_END)

    return stabilized + "\n".join(appendix_lines)


def get_prefix_hash(content: str) -> str:
    """Return a SHA-256 hex digest of the stable prefix of *content*.

    The stable prefix is the portion before the quantum-lock appendix
    delimiter.  Identical hashes across requests indicate a likely
    prompt-cache hit.
    """
    stabilized = stabilize(content)
    marker = f"\n{APPENDIX_SEPARATOR}\n{APPENDIX_START}"
    idx = stabilized.find(marker)
    prefix = stabilized[:idx] if idx != -1 else stabilized
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FusionStage implementation
# ---------------------------------------------------------------------------

class QuantumLock(FusionStage):
    """KV cache alignment stage for the Fusion Pipeline.

    Runs at order=3 (before Cortex at order=5) so that every downstream
    stage receives a prefix-stable version of the content.

    Only applies to system-role content; user/assistant/tool messages are
    passed through unchanged (they are not cached by Anthropic).
    """

    name = "quantum_lock"
    order = 3  # runs before Cortex (order=5)

    def should_apply(self, ctx: FusionContext) -> bool:
        """Apply only to system messages that contain dynamic content."""
        if ctx.role != "system":
            return False
        return bool(extract_dynamic(ctx.content))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        stabilized = stabilize(ctx.content)
        compressed_tokens = estimate_tokens(stabilized)

        fragments = extract_dynamic(ctx.content)
        markers = [
            f"quantum_lock:{frag.name}={frag.placeholder}"
            for frag in fragments
        ]

        warnings: list[str] = []
        if compressed_tokens > original_tokens:
            warnings.append(
                f"quantum_lock: stabilized content is larger than original "
                f"({compressed_tokens} > {original_tokens} tokens) — "
                f"dynamic appendix overhead"
            )

        return FusionResult(
            content=stabilized,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
            warnings=warnings,
        )
