"""FusionEngine — unified entry point for all Claw Compactor compression.

Constructs the full 14-stage Fusion Pipeline and exposes two public methods:

    engine.compress(text, ...)         — compress a single string
    engine.compress_messages(messages) — compress a list of OpenAI-format messages

The pipeline chains 14 stages in a fixed execution order:

    QuantumLock(3) -> Cortex(5) -> Photon(8) -> RLE(10) -> SemanticDedup(12)
    -> Ionizer(15) -> LogCrunch(16) -> SearchCrunch(17) -> DiffCrunch(18)
    -> StructuralCollapse(20) -> Neurosyntax(25) -> Nexus(35) -> TokenOpt(40)
    -> Abbrev(45)

Each stage receives an immutable FusionContext and returns an immutable
FusionResult.  The pipeline threads the compressed output forward — each
stage's result becomes the next stage's input context.  Stages that don't
apply to the current content type are skipped at zero cost via should_apply().

Three legacy modules (RLE, TokenizerOptimizer, CompressedContext) are wrapped
as adapter FusionStages so they participate in the same pipeline and metrics
infrastructure.

Achieves 54% weighted-average compression across six content types (code, JSON,
logs, diffs, search results, agent conversations) — a 5.9x improvement over
the legacy regex-only path.

Part of claw-compactor v7. License: MIT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from any cwd; project root is three levels up
# from this file (scripts/lib/fusion/engine.py → scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from claw_compactor.fusion.base import FusionContext, FusionResult, FusionStage
from claw_compactor.fusion.pipeline import FusionPipeline
from claw_compactor.fusion.cortex import Cortex
from claw_compactor.fusion.quantum_lock import QuantumLock
from claw_compactor.fusion.photon import PhotonStage
from claw_compactor.fusion.ionizer import Ionizer
from claw_compactor.fusion.log_crunch import LogCrunch
from claw_compactor.fusion.search_crunch import SearchCrunch
from claw_compactor.fusion.diff_crunch import DiffCrunch
from claw_compactor.fusion.semantic_dedup import SemanticDedup, dedup_across_messages
from claw_compactor.fusion.structural_collapse import StructuralCollapse
from claw_compactor.fusion.neurosyntax import Neurosyntax
from claw_compactor.fusion.nexus import NexusStage
from claw_compactor.rewind.store import RewindStore
from claw_compactor.tokens import estimate_tokens

# Legacy modules wrapped as adapter stages
import claw_compactor.rle as _rle
from claw_compactor.tokenizer_optimizer import optimize_tokens as _optimize_tokens

# compressed_context lives in scripts/, not scripts/lib/ — optional import
_CC_DIR = _SCRIPTS_DIR
if str(_CC_DIR) not in sys.path:
    sys.path.insert(0, str(_CC_DIR))

try:
    from compressed_context import (  # type: ignore[import]
        compress_ultra as _compress_ultra,
        ULTRA_ABBREVS as _ULTRA_ABBREVS,
        ULTRA_FILLERS as _ULTRA_FILLERS,
    )
except ImportError:
    _compress_ultra = None
    _ULTRA_ABBREVS = {}
    _ULTRA_FILLERS = set()


# ---------------------------------------------------------------------------
# Adapter FusionStages wrapping legacy modules
# ---------------------------------------------------------------------------

class RLEStage(FusionStage):
    """Wraps lib.rle.compress() — path, IP, and enum compression.

    Applies to all content types (the RLE transforms are structural-pattern
    aware and safe on any text).  Order 10 — runs after Photon (8) and before
    Ionizer (15).
    """

    name = "rle"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return bool(ctx.content)

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        compressed = _rle.compress(ctx.content)
        compressed_tokens = estimate_tokens(compressed)
        markers: list[str] = []
        if compressed_tokens < original_tokens:
            markers.append(f"rle:{original_tokens}->{compressed_tokens}")
        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )


class TokenOptStage(FusionStage):
    """Wraps lib.tokenizer_optimizer.optimize_tokens(aggressive=True).

    Cleans up formatting (bold/italic, excess whitespace, tables, bullets) to
    reduce tokenizer overhead.  Order 40 — runs after most semantic stages,
    before AbbrevStage.
    """

    name = "token_opt"
    order = 40

    def should_apply(self, ctx: FusionContext) -> bool:
        return bool(ctx.content)

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        compressed = _optimize_tokens(ctx.content, aggressive=True)
        compressed_tokens = estimate_tokens(compressed)
        markers: list[str] = []
        if compressed_tokens < original_tokens:
            markers.append(f"token_opt:{original_tokens}->{compressed_tokens}")
        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )


class AbbrevStage(FusionStage):
    """Wraps abbreviation + filler removal from compressed_context.compress_ultra().

    Only applied to natural language text (content_type == "text"), never to
    code, JSON, logs, diffs, or search results where abbreviations would corrupt
    structured data.  Order 45 — final aggressive pass before Nexus (35) has
    already run, but after TokenOpt cleans whitespace.
    """

    name = "abbrev"
    order = 45

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "text" and bool(ctx.content) and _compress_ultra is not None

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        compressed = _compress_ultra(ctx.content)  # type: ignore[misc]
        compressed_tokens = estimate_tokens(compressed)
        markers: list[str] = []
        if compressed_tokens < original_tokens:
            markers.append(f"abbrev:{original_tokens}->{compressed_tokens}")
        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )


# ---------------------------------------------------------------------------
# Stage ordering summary (for documentation)
# ---------------------------------------------------------------------------
#
#   3   QuantumLock     — KV-cache alignment (system messages only)
#   5   Cortex          — content-type detection
#   8   Photon          — image/base64 compression
#  10   RLEStage        — path / IP / enum compression        [adapter]
#  12   SemanticDedup   — near-duplicate block deduplication
#  15   Ionizer         — JSON array sampling
#  16   LogCrunch       — build/test log compression
#  17   SearchCrunch    — search result compression
#  18   DiffCrunch      — diff/patch compression
#  20   StructuralCollapse — import/repeated-line collapse
#  25   Neurosyntax     — AST-aware code compression
#  35   NexusStage      — ML token-level compressor (fallback: stopword removal)
#  40   TokenOptStage   — tokenizer format optimisation        [adapter]
#  45   AbbrevStage     — ultra-abbreviation (text only)       [adapter]


def _build_pipeline(rewind_store: RewindStore | None) -> FusionPipeline:
    """Construct the full pipeline with every stage, in order."""
    stages: list[FusionStage] = [
        QuantumLock(),
        Cortex(),
        PhotonStage(),
        RLEStage(),
        SemanticDedup(),
        Ionizer(rewind_store=rewind_store),
        LogCrunch(),
        SearchCrunch(),
        DiffCrunch(),
        StructuralCollapse(),
        Neurosyntax(),
        NexusStage(),
        TokenOptStage(),
        AbbrevStage(),
    ]
    return FusionPipeline(stages)


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Unified compression engine.  Single entry point for all compression.

    Parameters
    ----------
    enable_rewind:
        Maintain a RewindStore so compressed JSON arrays can be reversed.
        Default True.
    aggressive:
        Reserved for future per-stage aggressiveness knob.  Currently all
        adapter stages run at maximum aggressiveness.  Default True.
    """

    def __init__(
        self,
        enable_rewind: bool = True,
        aggressive: bool = True,
    ) -> None:
        self._rewind_store: RewindStore | None = (
            RewindStore() if enable_rewind else None
        )
        self._aggressive = aggressive
        self._pipeline: FusionPipeline = _build_pipeline(self._rewind_store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        text: str,
        content_type: str = "text",
        role: str = "user",
        language: str | None = None,
        model: str | None = None,
        token_budget: int | None = None,
        query: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Compress a single text string through the full pipeline.

        Parameters
        ----------
        text:
            The text to compress.
        content_type:
            Hint for the pipeline: "text", "code", "json", "log", "diff",
            "search".  Cortex will auto-detect if left as "text".
        role:
            Message role — "system", "user", "assistant", "tool".
        language:
            Optional programming language hint (e.g. "python", "go").
        model, token_budget, query, metadata:
            Additional context passed into FusionContext.

        Returns
        -------
        dict with keys:
            compressed  — the compressed string
            original    — the original string
            stats       — per-stage and aggregate stats dict
            markers     — list of compression marker strings
            warnings    — list of warning strings
        """
        if not text:
            return {
                "compressed": text,
                "original": text,
                "stats": _empty_stats(text),
                "markers": [],
                "warnings": [],
            }

        ctx = FusionContext(
            content=text,
            content_type=content_type,
            role=role,
            language=language,
            model=model,
            token_budget=token_budget,
            query=query,
            metadata=metadata or {},
        )

        pipeline_result = self._pipeline.run(ctx)
        stats = _build_stats(text, pipeline_result.content, pipeline_result)

        return {
            "compressed": pipeline_result.content,
            "original": text,
            "stats": stats,
            "markers": pipeline_result.markers,
            "warnings": pipeline_result.warnings,
        }

    def compress_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Compress a list of OpenAI-format chat messages.

        Each message must have at minimum ``role`` and ``content`` keys.
        Content may be a string or a list (OpenAI multipart format — only the
        text parts are compressed; image_url parts are passed through the
        normal Photon path).

        Parameters
        ----------
        messages:
            List of dicts, each with "role" and "content".

        Returns
        -------
        dict with keys:
            messages        — list of compressed message dicts (same structure
                              as input, content replaced with compressed text)
            stats           — aggregate stats across all messages
            per_message     — list of per-message stat dicts
            markers         — all markers from all messages combined
            warnings        — all warnings from all messages combined
        """
        if not messages:
            return {
                "messages": [],
                "stats": _empty_aggregate_stats(),
                "per_message": [],
                "markers": [],
                "warnings": [],
            }

        # Phase 0: cross-message semantic dedup
        deduped_messages, dedup_stats = dedup_across_messages(messages)

        compressed_messages: list[dict[str, Any]] = []
        per_message_stats: list[dict[str, Any]] = []
        all_markers: list[str] = []
        all_warnings: list[str] = []

        if dedup_stats.get("messages_deduped", 0) > 0:
            all_markers.append(
                f"cross_msg_dedup:{dedup_stats['messages_deduped']}_msgs_deduped"
            )

        total_original_tokens = 0
        total_compressed_tokens = 0
        total_original_chars = 0
        total_compressed_chars = 0
        total_timing_ms = 0.0

        for msg in deduped_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle multipart content (OpenAI list format).
            if isinstance(content, list):
                result_msg, msg_stats, msg_markers, msg_warnings = (
                    self._compress_multipart_message(role, content, msg)
                )
            else:
                result_msg, msg_stats, msg_markers, msg_warnings = (
                    self._compress_text_message(role, str(content), msg)
                )

            compressed_messages.append(result_msg)
            per_message_stats.append(msg_stats)
            all_markers.extend(msg_markers)
            all_warnings.extend(msg_warnings)

            total_original_tokens += msg_stats["original_tokens"]
            total_compressed_tokens += msg_stats["compressed_tokens"]
            total_original_chars += msg_stats["original_chars"]
            total_compressed_chars += msg_stats["compressed_chars"]
            total_timing_ms += msg_stats["timing_ms"]

        aggregate_stats = _aggregate_stats(
            original_tokens=total_original_tokens,
            compressed_tokens=total_compressed_tokens,
            original_chars=total_original_chars,
            compressed_chars=total_compressed_chars,
            timing_ms=total_timing_ms,
            message_count=len(messages),
        )

        return {
            "messages": compressed_messages,
            "stats": aggregate_stats,
            "per_message": per_message_stats,
            "markers": all_markers,
            "warnings": all_warnings,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_text_message(
        self,
        role: str,
        content: str,
        original_msg: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
        """Compress a plain-text message.  Returns (msg, stats, markers, warnings)."""
        t0 = time.monotonic()
        result = self.compress(text=content, role=role)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Build the output message preserving all keys from the original.
        out_msg = {**original_msg, "content": result["compressed"]}

        original_tokens = estimate_tokens(content)
        compressed_tokens = estimate_tokens(result["compressed"])

        msg_stats = {
            "role": role,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "original_chars": len(content),
            "compressed_chars": len(result["compressed"]),
            "reduction_pct": _reduction_pct(original_tokens, compressed_tokens),
            "timing_ms": round(elapsed_ms, 2),
            "stages_run": result["stats"].get("stages_run", 0),
        }

        return out_msg, msg_stats, result["markers"], result["warnings"]

    def _compress_multipart_message(
        self,
        role: str,
        parts: list[Any],
        original_msg: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
        """Compress a multipart (list-content) message.

        Text parts are run through the full pipeline.  Other part types
        (image_url, etc.) are passed through unchanged — Photon handles
        base64 images at the string level, but multipart image_url objects
        are left alone here.
        """
        t0 = time.monotonic()
        compressed_parts: list[Any] = []
        all_markers: list[str] = []
        all_warnings: list[str] = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_original_chars = 0
        total_compressed_chars = 0

        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                result = self.compress(text=text, role=role)
                compressed_parts.append({**part, "text": result["compressed"]})
                all_markers.extend(result["markers"])
                all_warnings.extend(result["warnings"])
                total_original_tokens += estimate_tokens(text)
                total_compressed_tokens += estimate_tokens(result["compressed"])
                total_original_chars += len(text)
                total_compressed_chars += len(result["compressed"])
            else:
                # Non-text part — pass through unchanged.
                compressed_parts.append(part)

        elapsed_ms = (time.monotonic() - t0) * 1000
        out_msg = {**original_msg, "content": compressed_parts}

        msg_stats = {
            "role": role,
            "original_tokens": total_original_tokens,
            "compressed_tokens": total_compressed_tokens,
            "original_chars": total_original_chars,
            "compressed_chars": total_compressed_chars,
            "reduction_pct": _reduction_pct(total_original_tokens, total_compressed_tokens),
            "timing_ms": round(elapsed_ms, 2),
            "stages_run": 0,  # aggregated across parts
        }

        return out_msg, msg_stats, all_markers, all_warnings

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> FusionPipeline:
        """The underlying FusionPipeline instance."""
        return self._pipeline

    @property
    def rewind_store(self) -> RewindStore | None:
        """The RewindStore instance (None if enable_rewind=False)."""
        return self._rewind_store

    @property
    def stage_names(self) -> list[str]:
        """Ordered list of stage names in the pipeline."""
        return [t.name for t in self._pipeline.transforms]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _reduction_pct(original: int, compressed: int) -> float:
    if original == 0:
        return 0.0
    return round((original - compressed) / original * 100, 2)


def _build_stats(
    original_text: str,
    compressed_text: str,
    pipeline_result: Any,
) -> dict[str, Any]:
    """Build a rich stats dict from a single-text pipeline result."""
    original_tokens = estimate_tokens(original_text)
    compressed_tokens = estimate_tokens(compressed_text)

    stages_run = sum(
        1 for step in pipeline_result.steps if not step.result.skipped
    )
    stages_skipped = sum(
        1 for step in pipeline_result.steps if step.result.skipped
    )

    per_stage = [
        {
            "name": step.transform_name,
            "skipped": step.result.skipped,
            "original_tokens": step.result.original_tokens,
            "compressed_tokens": step.result.compressed_tokens,
            "timing_ms": round(step.result.timing_ms, 3),
        }
        for step in pipeline_result.steps
    ]

    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "original_chars": len(original_text),
        "compressed_chars": len(compressed_text),
        "reduction_pct": _reduction_pct(original_tokens, compressed_tokens),
        "total_timing_ms": round(pipeline_result.total_timing_ms, 3),
        "stages_run": stages_run,
        "stages_skipped": stages_skipped,
        "per_stage": per_stage,
    }


def _empty_stats(text: str) -> dict[str, Any]:
    tokens = estimate_tokens(text)
    return {
        "original_tokens": tokens,
        "compressed_tokens": tokens,
        "original_chars": len(text),
        "compressed_chars": len(text),
        "reduction_pct": 0.0,
        "total_timing_ms": 0.0,
        "stages_run": 0,
        "stages_skipped": 0,
        "per_stage": [],
    }


def _empty_aggregate_stats() -> dict[str, Any]:
    return {
        "original_tokens": 0,
        "compressed_tokens": 0,
        "original_chars": 0,
        "compressed_chars": 0,
        "reduction_pct": 0.0,
        "total_timing_ms": 0.0,
        "message_count": 0,
    }


def _aggregate_stats(
    original_tokens: int,
    compressed_tokens: int,
    original_chars: int,
    compressed_chars: int,
    timing_ms: float,
    message_count: int,
) -> dict[str, Any]:
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "reduction_pct": _reduction_pct(original_tokens, compressed_tokens),
        "total_timing_ms": round(timing_ms, 3),
        "message_count": message_count,
    }
