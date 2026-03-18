"""Cortex — intelligent content router for the Fusion Pipeline.

Cortex is the pipeline's "brain" — it runs at order=5 (before all compressor
stages) and auto-detects content type (code, json, log, diff, search, text)
and programming language (16 languages supported) by analyzing structural
patterns, keywords, and syntax markers.  Detection results are propagated
into FusionContext via context_updates, so every downstream stage can make
type-aware compression decisions without redundant analysis.

Part of claw-compactor v7. License: MIT.
"""
from __future__ import annotations

from claw_compactor.fusion.base import FusionContext, FusionResult, FusionStage
from claw_compactor.fusion.content_detector import ContentDetector
from claw_compactor.tokens import estimate_tokens


class Cortex(FusionStage):
    """Intelligent content router. Detects content type and routes to appropriate compressors."""

    name = "cortex"
    order = 5  # must run before all compressor stages

    def __init__(self) -> None:
        self.detector = ContentDetector()

    def should_apply(self, ctx: FusionContext) -> bool:
        # Skip if a caller has already made an explicit type decision (non-default value).
        return ctx.content_type == "text"

    def apply(self, ctx: FusionContext) -> FusionResult:
        detection = self.detector.detect(ctx.content)
        tokens = estimate_tokens(ctx.content)

        context_updates: dict[str, object] = {
            "content_type": detection.content_type,
        }
        if detection.language is not None:
            context_updates["language"] = detection.language

        return FusionResult(
            content=ctx.content,
            original_tokens=tokens,
            compressed_tokens=tokens,  # Cortex never modifies content
            skipped=False,
            context_updates=context_updates,
        )
