"""CrunchBench: multi-dimensional evaluation engine for compression quality.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claw_compactor.fusion.base import FusionContext
from claw_compactor.fusion.pipeline import FusionPipeline
from claw_compactor.rewind.store import RewindStore

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable result from evaluating a single compression run."""
    compression_ratio: float      # original_tokens / compressed_tokens; >1 means savings
    accuracy_score: float | None  # LLM judge score (0-1), None if no LLM judge used
    reversibility: float          # Rewind retrieval exact-match rate (0-1)
    latency_ms: float             # total pipeline execution time in milliseconds
    cost_savings: float           # estimated dollar savings based on model pricing


# ---------------------------------------------------------------------------
# Pricing table (USD per 1 million tokens)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":    {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":  {"input": 3.0,   "output": 15.0},
    "gpt-4o":             {"input": 2.5,   "output": 10.0},
    "gpt-5.4":            {"input": 5.0,   "output": 15.0},
}

_DEFAULT_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Token counting helper (character-based approximation — no external deps)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Return a rough token count (4 chars per token, minimum 1)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# CrunchBench
# ---------------------------------------------------------------------------

class CrunchBench:
    """Evaluation engine for compression quality.

    Measures four dimensions for every run:
      - compression_ratio   — how much smaller the output is
      - accuracy_score      — optional LLM-judge faithfulness score
      - reversibility       — whether Rewind can restore the original
      - latency_ms          — wall-clock pipeline time
      - cost_savings        — estimated dollar savings at model pricing
    """

    MODEL_PRICING: dict[str, dict[str, float]] = _MODEL_PRICING

    def __init__(
        self,
        pipeline: FusionPipeline,
        rewind_store: RewindStore | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._rewind = rewind_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        text: str,
        ctx: FusionContext,
        model: str = _DEFAULT_MODEL,
    ) -> BenchmarkResult:
        """Compress *text* through the pipeline and return a BenchmarkResult.

        Args:
            text:  The original text to compress (used to build the context).
            ctx:   Base FusionContext; its *content* field is overridden with
                   *text* so the caller does not need to duplicate it.
            model: Model name used for cost-savings calculation.
        """
        # Override content so callers can pass a pre-built context template.
        ctx = ctx.evolve(content=text)

        original_tokens = _approx_tokens(text)

        # --- run pipeline and time it ---
        t0 = time.monotonic()
        pipeline_result = self._pipeline.run(ctx)
        latency_ms = (time.monotonic() - t0) * 1000.0

        compressed_text = pipeline_result.content
        compressed_tokens = _approx_tokens(compressed_text)

        # --- compression ratio ---
        compression_ratio = (
            original_tokens / compressed_tokens
            if compressed_tokens > 0
            else 1.0
        )

        # --- reversibility via RewindStore ---
        reversibility = self._score_reversibility(text, compressed_text)

        # --- cost savings ---
        cost_savings = self._estimate_cost_savings(
            original_tokens, compressed_tokens, model
        )

        # accuracy_score: None unless a subclass or caller wires in an LLM judge
        accuracy_score: float | None = None

        return BenchmarkResult(
            compression_ratio=compression_ratio,
            accuracy_score=accuracy_score,
            reversibility=reversibility,
            latency_ms=latency_ms,
            cost_savings=cost_savings,
        )

    def evaluate_dataset(
        self,
        samples: list[dict],
    ) -> list[BenchmarkResult]:
        """Evaluate a list of sample dicts.

        Each dict must contain:
          - ``text``  — the original text (str)
          - ``ctx``   — a FusionContext
        Optional keys:
          - ``model`` — override model for cost calculations (str)

        Returns a list of BenchmarkResult in the same order as *samples*.
        """
        results: list[BenchmarkResult] = []
        for sample in samples:
            text: str = sample["text"]
            ctx: FusionContext = sample["ctx"]
            model: str = sample.get("model", _DEFAULT_MODEL)
            result = self.evaluate_single(text, ctx, model=model)
            results.append(result)
        return results

    def summary(self, results: list[BenchmarkResult]) -> dict:
        """Return mean / median / p95 statistics for each numeric dimension.

        Returns an empty dict when *results* is empty.
        """
        if not results:
            return {}

        def _stats(values: list[float]) -> dict[str, float]:
            sorted_v = sorted(values)
            n = len(sorted_v)
            p95_idx = min(int(n * 0.95), n - 1)
            return {
                "mean":   statistics.mean(sorted_v),
                "median": statistics.median(sorted_v),
                "p95":    sorted_v[p95_idx],
                "min":    sorted_v[0],
                "max":    sorted_v[-1],
                "count":  float(n),
            }

        compression_ratios = [r.compression_ratio for r in results]
        reversibilities    = [r.reversibility     for r in results]
        latencies          = [r.latency_ms         for r in results]
        cost_savings_vals  = [r.cost_savings        for r in results]

        accuracy_vals = [
            r.accuracy_score for r in results if r.accuracy_score is not None
        ]

        out: dict = {
            "compression_ratio": _stats(compression_ratios),
            "reversibility":     _stats(reversibilities),
            "latency_ms":        _stats(latencies),
            "cost_savings":      _stats(cost_savings_vals),
            "sample_count":      len(results),
        }
        if accuracy_vals:
            out["accuracy_score"] = _stats(accuracy_vals)

        return out

    def report(self, results: list[BenchmarkResult]) -> str:
        """Return a human-readable Markdown report for *results*."""
        if not results:
            return "# CrunchBench Report\n\n_No results to display._\n"

        stats = self.summary(results)
        lines: list[str] = [
            "# CrunchBench Report",
            "",
            f"**Samples evaluated:** {stats['sample_count']}",
            "",
            "## Compression Ratio  (original_tokens / compressed_tokens)",
            _fmt_stats(stats["compression_ratio"]),
            "",
            "## Reversibility  (Rewind exact-match rate)",
            _fmt_stats(stats["reversibility"]),
            "",
            "## Latency  (ms)",
            _fmt_stats(stats["latency_ms"]),
            "",
            "## Estimated Cost Savings  (USD)",
            _fmt_stats(stats["cost_savings"]),
            "",
        ]
        if "accuracy_score" in stats:
            lines += [
                "## Accuracy Score  (LLM judge, 0-1)",
                _fmt_stats(stats["accuracy_score"]),
                "",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_reversibility(self, original: str, compressed: str) -> float:
        """Return 1.0 if Rewind can restore the original exactly, 0.0 otherwise.

        When no RewindStore is attached the score is 0.0 (cannot verify).
        """
        if self._rewind is None:
            return 0.0
        hash_id = self._rewind.store(
            original=original,
            compressed=compressed,
            original_tokens=_approx_tokens(original),
            compressed_tokens=_approx_tokens(compressed),
        )
        retrieved = self._rewind.retrieve(hash_id)
        return 1.0 if retrieved == original else 0.0

    def _estimate_cost_savings(
        self,
        original_tokens: int,
        compressed_tokens: int,
        model: str,
    ) -> float:
        """Estimate input-side dollar savings from token reduction.

        Uses the model's input price per 1M tokens.  Falls back to
        claude-sonnet-4-6 pricing when the model is unknown.
        """
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING[_DEFAULT_MODEL])
        price_per_million = pricing["input"]
        saved_tokens = max(0, original_tokens - compressed_tokens)
        return (saved_tokens / 1_000_000) * price_per_million


# ---------------------------------------------------------------------------
# Formatting helper (module-private)
# ---------------------------------------------------------------------------

def _fmt_stats(stats: dict[str, float]) -> str:
    """Format a stats dict as a compact Markdown table row."""
    return (
        "| mean | median | p95 | min | max |\n"
        "|------|--------|-----|-----|-----|\n"
        "| {mean:.4f} | {median:.4f} | {p95:.4f} | {min:.4f} | {max:.4f} |".format(**stats)
    )
