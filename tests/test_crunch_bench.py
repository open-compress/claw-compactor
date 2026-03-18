"""Tests for CrunchBench: multi-dimensional compression evaluation engine.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.crunch_bench import BenchmarkResult, CrunchBench, _approx_tokens, _fmt_stats
from lib.fusion.base import FusionContext, FusionResult, FusionStage
from lib.fusion.pipeline import FusionPipeline
from lib.rewind.store import RewindStore


# ---------------------------------------------------------------------------
# Minimal concrete FusionStage helpers
# ---------------------------------------------------------------------------

class HalfStage(FusionStage):
    """Drops every other word — halves approximate token count."""
    name = "half"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        words = ctx.content.split()
        compressed = " ".join(words[::2])
        return FusionResult(
            content=compressed,
            original_tokens=len(words),
            compressed_tokens=len(compressed.split()),
        )


class IdentityStage(FusionStage):
    """Returns content unchanged."""
    name = "identity"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        words = ctx.content.split()
        return FusionResult(
            content=ctx.content,
            original_tokens=len(words),
            compressed_tokens=len(words),
        )


class NeverStage(FusionStage):
    """Never applies — pipeline passes content through unchanged."""
    name = "never"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return False

    def apply(self, ctx: FusionContext) -> FusionResult:  # pragma: no cover
        raise AssertionError("should not be called")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def short_text() -> str:
    return "the quick brown fox jumps over the lazy dog"


@pytest.fixture()
def long_text() -> str:
    return " ".join([f"word{i}" for i in range(200)])


@pytest.fixture()
def half_pipeline() -> FusionPipeline:
    return FusionPipeline([HalfStage()])


@pytest.fixture()
def identity_pipeline() -> FusionPipeline:
    return FusionPipeline([IdentityStage()])


@pytest.fixture()
def empty_pipeline() -> FusionPipeline:
    return FusionPipeline()


@pytest.fixture()
def rewind() -> RewindStore:
    return RewindStore(max_entries=100, ttl_seconds=600)


@pytest.fixture()
def bench_with_rewind(half_pipeline, rewind) -> CrunchBench:
    return CrunchBench(half_pipeline, rewind_store=rewind)


@pytest.fixture()
def bench_no_rewind(half_pipeline) -> CrunchBench:
    return CrunchBench(half_pipeline)


@pytest.fixture()
def base_ctx() -> FusionContext:
    return FusionContext(content="placeholder", content_type="text")


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_is_frozen(self):
        r = BenchmarkResult(
            compression_ratio=2.0,
            accuracy_score=None,
            reversibility=1.0,
            latency_ms=5.0,
            cost_savings=0.001,
        )
        with pytest.raises((TypeError, AttributeError)):
            r.compression_ratio = 3.0  # type: ignore[misc]

    def test_fields_stored_correctly(self):
        r = BenchmarkResult(
            compression_ratio=1.5,
            accuracy_score=0.9,
            reversibility=0.8,
            latency_ms=12.3,
            cost_savings=0.0042,
        )
        assert r.compression_ratio == 1.5
        assert r.accuracy_score == 0.9
        assert r.reversibility == 0.8
        assert r.latency_ms == 12.3
        assert r.cost_savings == pytest.approx(0.0042, rel=1e-6)

    def test_accuracy_score_none_allowed(self):
        r = BenchmarkResult(
            compression_ratio=2.0,
            accuracy_score=None,
            reversibility=1.0,
            latency_ms=1.0,
            cost_savings=0.0,
        )
        assert r.accuracy_score is None


# ---------------------------------------------------------------------------
# _approx_tokens helper
# ---------------------------------------------------------------------------

class TestApproxTokens:
    def test_minimum_is_one(self):
        assert _approx_tokens("") == 1

    def test_four_chars_per_token(self):
        # 8-char string → 2 tokens
        assert _approx_tokens("abcdefgh") == 2

    def test_longer_text(self):
        text = "a" * 400
        assert _approx_tokens(text) == 100


# ---------------------------------------------------------------------------
# CrunchBench.evaluate_single
# ---------------------------------------------------------------------------

class TestEvaluateSingle:
    def test_returns_benchmark_result(self, bench_with_rewind, short_text, base_ctx):
        result = bench_with_rewind.evaluate_single(short_text, base_ctx)
        assert isinstance(result, BenchmarkResult)

    def test_compression_ratio_greater_than_one_when_compressed(
        self, bench_with_rewind, long_text, base_ctx
    ):
        result = bench_with_rewind.evaluate_single(long_text, base_ctx)
        assert result.compression_ratio > 1.0

    def test_compression_ratio_one_for_identity_pipeline(
        self, identity_pipeline, rewind, short_text, base_ctx
    ):
        bench = CrunchBench(identity_pipeline, rewind_store=rewind)
        result = bench.evaluate_single(short_text, base_ctx)
        # identity keeps same word count → ratio ≈ 1.0
        assert result.compression_ratio == pytest.approx(1.0, rel=0.01)

    def test_latency_ms_is_non_negative(self, bench_with_rewind, short_text, base_ctx):
        result = bench_with_rewind.evaluate_single(short_text, base_ctx)
        assert result.latency_ms >= 0.0

    def test_reversibility_one_when_rewind_store_present(
        self, bench_with_rewind, short_text, base_ctx
    ):
        result = bench_with_rewind.evaluate_single(short_text, base_ctx)
        assert result.reversibility == 1.0

    def test_reversibility_zero_when_no_rewind_store(
        self, bench_no_rewind, short_text, base_ctx
    ):
        result = bench_no_rewind.evaluate_single(short_text, base_ctx)
        assert result.reversibility == 0.0

    def test_accuracy_score_none_without_llm_judge(
        self, bench_with_rewind, short_text, base_ctx
    ):
        result = bench_with_rewind.evaluate_single(short_text, base_ctx)
        assert result.accuracy_score is None

    def test_cost_savings_positive_when_compressed(
        self, bench_with_rewind, long_text, base_ctx
    ):
        result = bench_with_rewind.evaluate_single(long_text, base_ctx)
        assert result.cost_savings > 0.0

    def test_cost_savings_zero_for_identity(
        self, identity_pipeline, rewind, short_text, base_ctx
    ):
        bench = CrunchBench(identity_pipeline, rewind_store=rewind)
        result = bench.evaluate_single(short_text, base_ctx)
        # identity: no tokens saved → cost_savings == 0
        assert result.cost_savings == pytest.approx(0.0, abs=1e-9)

    def test_ctx_content_is_overridden_by_text_arg(
        self, bench_with_rewind, base_ctx
    ):
        # base_ctx has content="placeholder"; evaluate_single must use text arg
        actual_text = "alpha beta gamma delta epsilon zeta"
        result = bench_with_rewind.evaluate_single(actual_text, base_ctx)
        # If content override works, ratio reflects actual_text not "placeholder"
        orig_tokens = _approx_tokens(actual_text)
        assert result.compression_ratio == pytest.approx(
            orig_tokens / _approx_tokens(" ".join(actual_text.split()[::2])),
            rel=0.05,
        )

    def test_unknown_model_falls_back_to_sonnet_pricing(
        self, bench_with_rewind, long_text, base_ctx
    ):
        # Should not raise; uses fallback pricing
        result = bench_with_rewind.evaluate_single(
            long_text, base_ctx, model="non-existent-model-xyz"
        )
        assert result.cost_savings >= 0.0

    def test_known_model_pricing_used(self, half_pipeline, rewind, long_text, base_ctx):
        bench_opus = CrunchBench(half_pipeline, rewind_store=rewind)
        bench_gpt4o = CrunchBench(half_pipeline, rewind_store=rewind)

        result_opus = bench_opus.evaluate_single(long_text, base_ctx, model="claude-opus-4-6")
        result_gpt4o = bench_gpt4o.evaluate_single(long_text, base_ctx, model="gpt-4o")

        # opus input = $15/M, gpt-4o input = $2.5/M → opus savings > gpt-4o savings
        assert result_opus.cost_savings > result_gpt4o.cost_savings

    def test_empty_text_does_not_raise(self, bench_with_rewind, base_ctx):
        result = bench_with_rewind.evaluate_single("", base_ctx)
        assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# CrunchBench.evaluate_dataset
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    def test_returns_list_of_same_length(self, bench_with_rewind, base_ctx):
        samples = [
            {"text": "hello world foo bar", "ctx": base_ctx},
            {"text": "another sample text here", "ctx": base_ctx},
            {"text": "third entry in the dataset", "ctx": base_ctx},
        ]
        results = bench_with_rewind.evaluate_dataset(samples)
        assert len(results) == 3

    def test_each_result_is_benchmark_result(self, bench_with_rewind, base_ctx):
        samples = [{"text": "sample text for testing", "ctx": base_ctx}]
        results = bench_with_rewind.evaluate_dataset(samples)
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_empty_dataset_returns_empty_list(self, bench_with_rewind):
        assert bench_with_rewind.evaluate_dataset([]) == []

    def test_per_sample_model_override(self, bench_with_rewind, base_ctx, long_text):
        samples = [
            {"text": long_text, "ctx": base_ctx, "model": "claude-opus-4-6"},
            {"text": long_text, "ctx": base_ctx, "model": "gpt-4o"},
        ]
        results = bench_with_rewind.evaluate_dataset(samples)
        # opus ($15/M input) vs gpt-4o ($2.5/M input)
        assert results[0].cost_savings > results[1].cost_savings


# ---------------------------------------------------------------------------
# CrunchBench.summary
# ---------------------------------------------------------------------------

class TestSummary:
    def _make_results(self, ratios, reversibilities=None, latencies=None, savings=None):
        n = len(ratios)
        reversibilities = reversibilities or [1.0] * n
        latencies = latencies or [5.0] * n
        savings = savings or [0.001] * n
        return [
            BenchmarkResult(
                compression_ratio=ratios[i],
                accuracy_score=None,
                reversibility=reversibilities[i],
                latency_ms=latencies[i],
                cost_savings=savings[i],
            )
            for i in range(n)
        ]

    def test_empty_results_returns_empty_dict(self, bench_with_rewind):
        assert bench_with_rewind.summary([]) == {}

    def test_summary_contains_required_keys(self, bench_with_rewind):
        results = self._make_results([1.5, 2.0, 2.5])
        summary = bench_with_rewind.summary(results)
        assert "compression_ratio" in summary
        assert "reversibility" in summary
        assert "latency_ms" in summary
        assert "cost_savings" in summary
        assert "sample_count" in summary

    def test_sample_count_is_correct(self, bench_with_rewind):
        results = self._make_results([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = bench_with_rewind.summary(results)
        assert summary["sample_count"] == 5

    def test_mean_is_correct(self, bench_with_rewind):
        results = self._make_results([1.0, 2.0, 3.0])
        summary = bench_with_rewind.summary(results)
        assert summary["compression_ratio"]["mean"] == pytest.approx(2.0, rel=1e-6)

    def test_median_is_correct(self, bench_with_rewind):
        results = self._make_results([1.0, 3.0, 2.0])
        summary = bench_with_rewind.summary(results)
        assert summary["compression_ratio"]["median"] == pytest.approx(2.0, rel=1e-6)

    def test_p95_is_within_range(self, bench_with_rewind):
        values = list(range(1, 21))  # 20 items; p95 at index 19 = value 20
        results = self._make_results([float(v) for v in values])
        summary = bench_with_rewind.summary(results)
        assert summary["compression_ratio"]["p95"] == pytest.approx(20.0, rel=0.1)

    def test_min_max_correct(self, bench_with_rewind):
        results = self._make_results([3.0, 1.0, 5.0, 2.0, 4.0])
        summary = bench_with_rewind.summary(results)
        assert summary["compression_ratio"]["min"] == pytest.approx(1.0)
        assert summary["compression_ratio"]["max"] == pytest.approx(5.0)

    def test_accuracy_score_absent_when_all_none(self, bench_with_rewind):
        results = self._make_results([2.0])
        summary = bench_with_rewind.summary(results)
        assert "accuracy_score" not in summary

    def test_accuracy_score_present_when_some_set(self, bench_with_rewind):
        results = [
            BenchmarkResult(2.0, accuracy_score=0.9, reversibility=1.0, latency_ms=5.0, cost_savings=0.001),
            BenchmarkResult(2.0, accuracy_score=None, reversibility=1.0, latency_ms=5.0, cost_savings=0.001),
        ]
        summary = bench_with_rewind.summary(results)
        assert "accuracy_score" in summary
        assert summary["accuracy_score"]["mean"] == pytest.approx(0.9)

    def test_single_result_summary(self, bench_with_rewind):
        results = self._make_results([2.5])
        summary = bench_with_rewind.summary(results)
        assert summary["compression_ratio"]["mean"] == pytest.approx(2.5)
        assert summary["compression_ratio"]["min"] == pytest.approx(2.5)
        assert summary["compression_ratio"]["max"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CrunchBench.report
# ---------------------------------------------------------------------------

class TestReport:
    def _make_single_result(self):
        return [BenchmarkResult(
            compression_ratio=2.0,
            accuracy_score=None,
            reversibility=1.0,
            latency_ms=10.0,
            cost_savings=0.005,
        )]

    def test_empty_results_returns_no_results_message(self, bench_with_rewind):
        report = bench_with_rewind.report([])
        assert "No results" in report

    def test_report_is_string(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert isinstance(report, str)

    def test_report_contains_title(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "CrunchBench" in report

    def test_report_contains_compression_ratio_section(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "Compression Ratio" in report

    def test_report_contains_reversibility_section(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "Reversibility" in report

    def test_report_contains_latency_section(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "Latency" in report

    def test_report_contains_cost_savings_section(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "Cost Savings" in report

    def test_report_no_accuracy_section_when_none(self, bench_with_rewind):
        report = bench_with_rewind.report(self._make_single_result())
        assert "Accuracy Score" not in report

    def test_report_has_accuracy_section_when_provided(self, bench_with_rewind):
        results = [BenchmarkResult(
            compression_ratio=2.0,
            accuracy_score=0.85,
            reversibility=1.0,
            latency_ms=5.0,
            cost_savings=0.001,
        )]
        report = bench_with_rewind.report(results)
        assert "Accuracy Score" in report

    def test_report_contains_sample_count(self, bench_with_rewind):
        results = self._make_single_result() * 3
        report = bench_with_rewind.report(results)
        assert "3" in report


# ---------------------------------------------------------------------------
# MODEL_PRICING class attribute
# ---------------------------------------------------------------------------

class TestModelPricing:
    def test_all_expected_models_present(self):
        expected = {"claude-opus-4-6", "claude-sonnet-4-6", "gpt-4o", "gpt-5.4"}
        assert expected.issubset(set(CrunchBench.MODEL_PRICING.keys()))

    def test_each_model_has_input_and_output(self):
        for model, pricing in CrunchBench.MODEL_PRICING.items():
            assert "input" in pricing, f"Missing 'input' for {model}"
            assert "output" in pricing, f"Missing 'output' for {model}"

    def test_pricing_values_are_positive(self):
        for model, pricing in CrunchBench.MODEL_PRICING.items():
            assert pricing["input"] > 0, f"Non-positive input price for {model}"
            assert pricing["output"] > 0, f"Non-positive output price for {model}"
