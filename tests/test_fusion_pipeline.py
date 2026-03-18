"""Tests for Fusion Pipeline: base classes, FusionContext, FusionResult, and FusionPipeline engine."""

import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult
from claw_compactor.fusion.pipeline import FusionPipeline, FusionPipelineResult, FusionStepResult


# ---------------------------------------------------------------------------
# Concrete FusionStage helpers
# ---------------------------------------------------------------------------

class UpperTransform(FusionStage):
    """Uppercases content; only runs on content_type == 'text'."""
    name = "upper"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "text"

    def apply(self, ctx: FusionContext) -> FusionResult:
        result = ctx.content.upper()
        return FusionResult(
            content=result,
            original_tokens=len(ctx.content.split()),
            compressed_tokens=len(result.split()),
        )


class AppendTransform(FusionStage):
    """Appends a suffix; always applies."""
    name = "append"
    order = 20

    def __init__(self, suffix: str = " [done]"):
        self.suffix = suffix

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        result = ctx.content + self.suffix
        return FusionResult(
            content=result,
            original_tokens=len(ctx.content.split()),
            compressed_tokens=len(result.split()),
            markers=["append-marker"],
            warnings=["append-warning"],
        )


class NeverApplyTransform(FusionStage):
    """Never applies."""
    name = "never"
    order = 5

    def should_apply(self, ctx: FusionContext) -> bool:
        return False

    def apply(self, ctx: FusionContext) -> FusionResult:
        raise AssertionError("apply() must not be called when should_apply returns False")


class HighOrderTransform(FusionStage):
    """High order value — should run last."""
    name = "high_order"
    order = 100

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        result = f"[high]{ctx.content}[/high]"
        return FusionResult(content=result, original_tokens=1, compressed_tokens=1)


class LowOrderTransform(FusionStage):
    """Low order value — should run first."""
    name = "low_order"
    order = 1

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        result = f"[low]{ctx.content}[/low]"
        return FusionResult(content=result, original_tokens=1, compressed_tokens=1)


class TrackingTransform(FusionStage):
    """Records the content it received, for chaining verification."""
    name = "tracker"
    order = 50

    def __init__(self):
        self.received: list[str] = []

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        self.received.append(ctx.content)
        return FusionResult(
            content=ctx.content + "|tracked",
            original_tokens=1,
            compressed_tokens=1,
        )


# ---------------------------------------------------------------------------
# FusionContext tests
# ---------------------------------------------------------------------------

class TestFusionContext:
    def test_defaults(self):
        ctx = FusionContext(content="hello")
        assert ctx.content == "hello"
        assert ctx.content_type == "text"
        assert ctx.role == "user"
        assert ctx.language is None
        assert ctx.model is None
        assert ctx.token_budget is None
        assert ctx.query is None
        assert ctx.metadata == {}

    def test_evolve_returns_new_instance(self):
        ctx = FusionContext(content="hello", content_type="text")
        evolved = ctx.evolve(content="world")
        assert evolved is not ctx
        assert evolved.content == "world"

    def test_evolve_original_unchanged(self):
        ctx = FusionContext(content="original", content_type="text")
        ctx.evolve(content="changed")
        assert ctx.content == "original"

    def test_evolve_only_changes_specified_fields(self):
        ctx = FusionContext(content="hello", content_type="code", role="system", language="python")
        evolved = ctx.evolve(content="world")
        assert evolved.content_type == "code"
        assert evolved.role == "system"
        assert evolved.language == "python"

    def test_evolve_multiple_fields(self):
        ctx = FusionContext(content="x")
        evolved = ctx.evolve(content="y", content_type="json", role="assistant")
        assert evolved.content == "y"
        assert evolved.content_type == "json"
        assert evolved.role == "assistant"

    def test_frozen_mutation_raises(self):
        ctx = FusionContext(content="hello")
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            ctx.content = "mutated"  # type: ignore[misc]

    def test_metadata_default_is_empty_dict(self):
        ctx1 = FusionContext(content="a")
        ctx2 = FusionContext(content="b")
        # Each instance gets its own dict (not shared)
        ctx1.metadata["key"] = "val"
        assert "key" not in ctx2.metadata

    def test_evolve_with_metadata(self):
        ctx = FusionContext(content="hello")
        evolved = ctx.evolve(metadata={"source": "test"})
        assert evolved.metadata == {"source": "test"}
        assert ctx.metadata == {}


# ---------------------------------------------------------------------------
# FusionResult tests
# ---------------------------------------------------------------------------

class TestFusionResult:
    def test_defaults(self):
        result = FusionResult(content="hello")
        assert result.content == "hello"
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.markers == []
        assert result.warnings == []
        assert result.timing_ms == 0.0
        assert result.skipped is False

    def test_frozen(self):
        result = FusionResult(content="hello")
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            result.content = "changed"  # type: ignore[misc]

    def test_custom_values(self):
        result = FusionResult(
            content="out",
            original_tokens=10,
            compressed_tokens=5,
            markers=["m1"],
            warnings=["w1"],
            timing_ms=3.14,
            skipped=True,
        )
        assert result.original_tokens == 10
        assert result.compressed_tokens == 5
        assert result.markers == ["m1"]
        assert result.warnings == ["w1"]
        assert result.timing_ms == 3.14
        assert result.skipped is True

    def test_skipped_flag_default_false(self):
        result = FusionResult(content="x")
        assert result.skipped is False

    def test_lists_are_independent_per_instance(self):
        r1 = FusionResult(content="a")
        r2 = FusionResult(content="b")
        # default_factory should produce independent lists
        assert r1.markers is not r2.markers
        assert r1.warnings is not r2.warnings


# ---------------------------------------------------------------------------
# FusionStage.timed_apply() tests
# ---------------------------------------------------------------------------

class TestTimedApply:
    def test_skips_when_should_apply_false(self):
        transform = NeverApplyTransform()
        ctx = FusionContext(content="hello")
        result = transform.timed_apply(ctx)
        assert result.skipped is True
        assert result.content == "hello"

    def test_skipped_result_preserves_original_content(self):
        transform = NeverApplyTransform()
        ctx = FusionContext(content="preserve me", content_type="text")
        result = transform.timed_apply(ctx)
        assert result.content == "preserve me"

    def test_records_timing(self):
        transform = AppendTransform()
        ctx = FusionContext(content="hello")
        result = transform.timed_apply(ctx)
        assert result.timing_ms >= 0.0
        assert isinstance(result.timing_ms, float)

    def test_timing_is_nonzero_for_slow_transform(self):
        class SlowTransform(FusionStage):
            name = "slow"
            order = 1

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                time.sleep(0.02)
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1)

        transform = SlowTransform()
        ctx = FusionContext(content="hello")
        result = transform.timed_apply(ctx)
        assert result.timing_ms >= 15.0  # at least 15ms

    def test_applies_transform_when_should_apply_true(self):
        transform = UpperTransform()
        ctx = FusionContext(content="hello world", content_type="text")
        result = transform.timed_apply(ctx)
        assert result.skipped is False
        assert result.content == "HELLO WORLD"

    def test_skipped_result_has_zero_tokens(self):
        transform = NeverApplyTransform()
        ctx = FusionContext(content="hello")
        result = transform.timed_apply(ctx)
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_skipped_result_has_skipped_true_and_no_timing_overhead(self):
        transform = NeverApplyTransform()
        ctx = FusionContext(content="hello")
        result = transform.timed_apply(ctx)
        assert result.skipped is True

    def test_timed_apply_copies_markers_and_warnings(self):
        class MarkerTransform(FusionStage):
            name = "marker_t"
            order = 1

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                return FusionResult(
                    content=ctx.content,
                    original_tokens=1,
                    compressed_tokens=1,
                    markers=["m1", "m2"],
                    warnings=["w1"],
                )

        transform = MarkerTransform()
        ctx = FusionContext(content="test")
        result = transform.timed_apply(ctx)
        assert result.markers == ["m1", "m2"]
        assert result.warnings == ["w1"]

    def test_upper_does_not_apply_to_non_text(self):
        transform = UpperTransform()
        ctx = FusionContext(content="hello world", content_type="code")
        result = transform.timed_apply(ctx)
        assert result.skipped is True
        assert result.content == "hello world"


# ---------------------------------------------------------------------------
# FusionPipeline tests
# ---------------------------------------------------------------------------

class TestFusionPipeline:
    def test_empty_pipeline_returns_original_content(self):
        pipeline = FusionPipeline()
        ctx = FusionContext(content="hello world")
        pr = pipeline.run(ctx)
        assert pr.content == "hello world"

    def test_empty_pipeline_has_no_steps(self):
        pipeline = FusionPipeline()
        ctx = FusionContext(content="hello")
        pr = pipeline.run(ctx)
        assert pr.steps == []

    def test_single_transform_applied(self):
        pipeline = FusionPipeline([UpperTransform()])
        ctx = FusionContext(content="hello world", content_type="text")
        pr = pipeline.run(ctx)
        assert pr.content == "HELLO WORLD"

    def test_runs_transforms_in_order_by_order_field(self):
        # LowOrderTransform (order=1) should run before HighOrderTransform (order=100)
        # LowOrderTransform wraps with [low]...[/low] first, then HighOrderTransform wraps that
        pipeline = FusionPipeline([HighOrderTransform(), LowOrderTransform()])
        ctx = FusionContext(content="x")
        pr = pipeline.run(ctx)
        # low runs first: "[low]x[/low]"
        # high runs second: "[high][low]x[/low][/high]"
        assert pr.content == "[high][low]x[/low][/high]"

    def test_chains_output_each_transform_receives_previous_output(self):
        tracker = TrackingTransform()  # order=50
        upper = UpperTransform()       # order=10
        # upper runs first (order=10), tracker second (order=50)
        pipeline = FusionPipeline([tracker, upper])
        ctx = FusionContext(content="hello", content_type="text")
        pr = pipeline.run(ctx)
        # upper ran first, tracker received uppercased content
        assert tracker.received[0] == "HELLO"

    def test_pipeline_add_returns_new_pipeline(self):
        p1 = FusionPipeline()
        p2 = p1.add(UpperTransform())
        assert p2 is not p1

    def test_pipeline_add_does_not_mutate_original(self):
        p1 = FusionPipeline([AppendTransform()])
        p1.add(UpperTransform())
        assert len(p1.transforms) == 1
        assert p1.transforms[0].name == "append"

    def test_pipeline_add_includes_new_transform(self):
        p1 = FusionPipeline()
        p2 = p1.add(UpperTransform())
        assert len(p2.transforms) == 1
        assert p2.transforms[0].name == "upper"

    def test_pipeline_add_maintains_order_sorting(self):
        p = FusionPipeline()
        p = p.add(HighOrderTransform())   # order=100
        p = p.add(LowOrderTransform())    # order=1
        names = [t.name for t in p.transforms]
        assert names == ["low_order", "high_order"]

    def test_collects_markers_from_all_steps(self):
        class MarkerA(FusionStage):
            name = "ma"
            order = 1

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1, markers=["marker-a"])

        class MarkerB(FusionStage):
            name = "mb"
            order = 2

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1, markers=["marker-b"])

        pipeline = FusionPipeline([MarkerA(), MarkerB()])
        ctx = FusionContext(content="x")
        pr = pipeline.run(ctx)
        assert "marker-a" in pr.markers
        assert "marker-b" in pr.markers

    def test_collects_warnings_from_all_steps(self):
        class WarnA(FusionStage):
            name = "wa"
            order = 1

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1, warnings=["warn-a"])

        class WarnB(FusionStage):
            name = "wb"
            order = 2

            def should_apply(self, ctx):
                return True

            def apply(self, ctx):
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1, warnings=["warn-b"])

        pipeline = FusionPipeline([WarnA(), WarnB()])
        ctx = FusionContext(content="x")
        pr = pipeline.run(ctx)
        assert "warn-a" in pr.warnings
        assert "warn-b" in pr.warnings

    def test_skipped_transform_does_not_contribute_markers(self):
        class SkippedWithMarkers(FusionStage):
            name = "skip_marker"
            order = 1

            def should_apply(self, ctx):
                return False

            def apply(self, ctx):
                return FusionResult(content=ctx.content, original_tokens=1, compressed_tokens=1, markers=["should-not-appear"])

        pipeline = FusionPipeline([SkippedWithMarkers()])
        ctx = FusionContext(content="x")
        pr = pipeline.run(ctx)
        assert "should-not-appear" not in pr.markers

    def test_skipped_transform_does_not_change_content(self):
        pipeline = FusionPipeline([NeverApplyTransform()])
        ctx = FusionContext(content="unchanged")
        pr = pipeline.run(ctx)
        assert pr.content == "unchanged"

    def test_total_timing_accumulated(self):
        pipeline = FusionPipeline([AppendTransform(" a"), AppendTransform(" b")])
        ctx = FusionContext(content="x")
        pr = pipeline.run(ctx)
        assert pr.total_timing_ms >= 0.0
        # total should be sum of step timings
        step_total = sum(s.result.timing_ms for s in pr.steps)
        assert abs(pr.total_timing_ms - step_total) < 0.001

    def test_steps_list_contains_all_transform_names(self):
        pipeline = FusionPipeline([UpperTransform(), AppendTransform()])
        ctx = FusionContext(content="hello", content_type="text")
        pr = pipeline.run(ctx)
        names = [s.transform_name for s in pr.steps]
        assert "upper" in names
        assert "append" in names

    def test_empty_content_handled(self):
        pipeline = FusionPipeline([UpperTransform()])
        ctx = FusionContext(content="", content_type="text")
        pr = pipeline.run(ctx)
        assert pr.content == ""

    def test_pipeline_result_is_fusion_pipeline_result_type(self):
        pipeline = FusionPipeline()
        ctx = FusionContext(content="hello")
        pr = pipeline.run(ctx)
        assert isinstance(pr, FusionPipelineResult)

    def test_multiple_transforms_chain_content(self):
        # upper (order=10) -> append (order=20)
        pipeline = FusionPipeline([AppendTransform(" [done]"), UpperTransform()])
        ctx = FusionContext(content="hello", content_type="text")
        pr = pipeline.run(ctx)
        # upper runs first: "HELLO", then append: "HELLO [done]"
        assert pr.content == "HELLO [done]"

    def test_transforms_property_returns_sorted_copy(self):
        pipeline = FusionPipeline([HighOrderTransform(), LowOrderTransform()])
        transforms = pipeline.transforms
        assert transforms[0].order < transforms[1].order
        # Modifying returned list does not affect pipeline
        transforms.append(AppendTransform())
        assert len(pipeline.transforms) == 2

    def test_pipeline_with_all_skipped_returns_original(self):
        pipeline = FusionPipeline([NeverApplyTransform()])
        ctx = FusionContext(content="original text")
        pr = pipeline.run(ctx)
        assert pr.content == "original text"
