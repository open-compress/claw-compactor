"""Fusion pipeline engine: ordered chain of FusionStages with immutable data flow.

Stages are sorted by their ``order`` attribute at construction time.  At runtime,
each stage's timed_apply() is called sequentially — the compressed output from
stage N becomes the input FusionContext for stage N+1.  Stages may propagate
context_updates (e.g. Cortex setting content_type="code") that modify the
context for all downstream stages.

The pipeline is immutable: add() returns a new FusionPipeline instance.

Part of claw-compactor v7. License: MIT.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionStepResult:
    """Result from a single fusion pipeline step."""
    transform_name: str
    result: FusionResult


@dataclass(frozen=True)
class FusionPipelineResult:
    """Aggregated result from running all fusion stages."""
    content: str
    steps: list[FusionStepResult] = field(default_factory=list)
    total_timing_ms: float = 0.0
    markers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FusionPipeline:
    """Ordered chain of FusionStages."""

    def __init__(self, transforms: list[FusionStage] | None = None):
        self._transforms: list[FusionStage] = sorted(
            transforms or [], key=lambda t: t.order
        )

    def add(self, transform: FusionStage) -> FusionPipeline:
        """Return a new FusionPipeline with the fusion stage added (immutable)."""
        new_transforms = sorted(
            [*self._transforms, transform], key=lambda t: t.order
        )
        return FusionPipeline(new_transforms)

    @property
    def transforms(self) -> list[FusionStage]:
        return list(self._transforms)

    def run(self, ctx: FusionContext) -> FusionPipelineResult:
        """Run all fusion stages sequentially. Each stage's output feeds the next."""
        steps: list[FusionStepResult] = []
        all_markers: list[str] = []
        all_warnings: list[str] = []
        total_ms = 0.0
        current_ctx = ctx

        for transform in self._transforms:
            result = transform.timed_apply(current_ctx)
            steps.append(FusionStepResult(
                transform_name=transform.name,
                result=result,
            ))
            total_ms += result.timing_ms

            if not result.skipped:
                updates = {"content": result.content, **result.context_updates}
                current_ctx = current_ctx.evolve(**updates)
                all_markers.extend(result.markers)
                all_warnings.extend(result.warnings)
                logger.debug(
                    "%s: %d→%d tokens (%.1fms)",
                    transform.name,
                    result.original_tokens,
                    result.compressed_tokens,
                    result.timing_ms,
                )
            else:
                logger.debug("%s: skipped", transform.name)

        return FusionPipelineResult(
            content=current_ctx.content,
            steps=steps,
            total_timing_ms=total_ms,
            markers=all_markers,
            warnings=all_warnings,
        )
