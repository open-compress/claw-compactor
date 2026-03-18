"""Tests for FeedbackLoop: retrieval event tracking and compression adjustment.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.feedback import FeedbackLoop, RetrievalEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    stage: str,
    was_retrieved: bool,
    compression_ratio: float = 2.0,
    hash_id: str = "abc123",
) -> RetrievalEvent:
    """Convenience factory for RetrievalEvent with a fixed timestamp."""
    return RetrievalEvent(
        hash_id=hash_id,
        stage_name=stage,
        compression_ratio=compression_ratio,
        was_retrieved=was_retrieved,
        timestamp=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# RetrievalEvent dataclass
# ---------------------------------------------------------------------------

class TestRetrievalEvent:
    def test_is_frozen(self):
        ev = _event("compress", True)
        with pytest.raises((TypeError, AttributeError)):
            ev.was_retrieved = False  # type: ignore[misc]

    def test_fields_stored(self):
        ev = RetrievalEvent(
            hash_id="deadbeef",
            stage_name="half",
            compression_ratio=3.5,
            was_retrieved=True,
            timestamp=1234.5,
        )
        assert ev.hash_id == "deadbeef"
        assert ev.stage_name == "half"
        assert ev.compression_ratio == 3.5
        assert ev.was_retrieved is True
        assert ev.timestamp == 1234.5


# ---------------------------------------------------------------------------
# FeedbackLoop construction
# ---------------------------------------------------------------------------

class TestFeedbackLoopConstruction:
    def test_default_window_size(self):
        loop = FeedbackLoop()
        assert loop._window_size == 100

    def test_custom_window_size(self):
        loop = FeedbackLoop(window_size=50)
        assert loop._window_size == 50

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            FeedbackLoop(window_size=0)

    def test_negative_window_size_raises(self):
        with pytest.raises(ValueError):
            FeedbackLoop(window_size=-5)


# ---------------------------------------------------------------------------
# FeedbackLoop.record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_single_event(self):
        loop = FeedbackLoop()
        loop.record(_event("compress", True))
        assert loop.retrieval_rate() == 1.0

    def test_record_multiple_events(self):
        loop = FeedbackLoop()
        loop.record(_event("compress", True))
        loop.record(_event("compress", False))
        assert loop.retrieval_rate() == pytest.approx(0.5)

    def test_record_non_retrieved_events(self):
        loop = FeedbackLoop()
        for _ in range(5):
            loop.record(_event("compress", False))
        assert loop.retrieval_rate() == 0.0

    def test_record_all_retrieved_events(self):
        loop = FeedbackLoop()
        for _ in range(5):
            loop.record(_event("compress", True))
        assert loop.retrieval_rate() == 1.0


# ---------------------------------------------------------------------------
# FeedbackLoop.retrieval_rate — overall
# ---------------------------------------------------------------------------

class TestRetrievalRateOverall:
    def test_empty_loop_returns_zero(self):
        loop = FeedbackLoop()
        assert loop.retrieval_rate() == 0.0

    def test_half_retrieved(self):
        loop = FeedbackLoop()
        loop.record(_event("a", True))
        loop.record(_event("a", False))
        loop.record(_event("b", True))
        loop.record(_event("b", False))
        assert loop.retrieval_rate() == pytest.approx(0.5)

    def test_one_of_four_retrieved(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True))
        loop.record(_event("s", False))
        loop.record(_event("s", False))
        loop.record(_event("s", False))
        assert loop.retrieval_rate() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# FeedbackLoop.retrieval_rate — per-stage
# ---------------------------------------------------------------------------

class TestRetrievalRatePerStage:
    def test_per_stage_ignores_other_stages(self):
        loop = FeedbackLoop()
        # stage_a: 100% retrieval
        loop.record(_event("stage_a", True))
        loop.record(_event("stage_a", True))
        # stage_b: 0% retrieval
        loop.record(_event("stage_b", False))
        loop.record(_event("stage_b", False))

        assert loop.retrieval_rate("stage_a") == pytest.approx(1.0)
        assert loop.retrieval_rate("stage_b") == pytest.approx(0.0)

    def test_unknown_stage_returns_zero(self):
        loop = FeedbackLoop()
        loop.record(_event("known_stage", True))
        assert loop.retrieval_rate("nonexistent_stage") == 0.0

    def test_per_stage_mixed(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True))
        loop.record(_event("s", False))
        loop.record(_event("s", True))
        # 2 out of 3 retrieved
        assert loop.retrieval_rate("s") == pytest.approx(2 / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# FeedbackLoop — window overflow
# ---------------------------------------------------------------------------

class TestWindowOverflow:
    def test_window_evicts_oldest_when_full(self):
        loop = FeedbackLoop(window_size=5)
        # Fill window: all retrieved
        for _ in range(5):
            loop.record(_event("s", True))
        assert loop.retrieval_rate() == 1.0

        # Push one non-retrieved event → oldest retrieved event evicted
        loop.record(_event("s", False))
        # 4 retrieved, 1 not → 4/5
        assert loop.retrieval_rate() == pytest.approx(4 / 5)

    def test_window_does_not_grow_beyond_max(self):
        loop = FeedbackLoop(window_size=10)
        for i in range(50):
            loop.record(_event("s", i % 2 == 0))
        assert len(loop._events) == 10

    def test_oldest_events_are_gone_after_overflow(self):
        loop = FeedbackLoop(window_size=3)
        # Record 3 retrieved events
        for _ in range(3):
            loop.record(_event("s", True))
        # Now add 3 non-retrieved events — all original events evicted
        for _ in range(3):
            loop.record(_event("s", False))
        assert loop.retrieval_rate() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FeedbackLoop.suggest_adjustments
# ---------------------------------------------------------------------------

class TestSuggestAdjustments:
    def test_no_events_returns_empty(self):
        loop = FeedbackLoop()
        assert loop.suggest_adjustments() == {}

    def test_low_retrieval_rate_no_adjustment(self):
        loop = FeedbackLoop()
        # 20% retrieval rate — below 0.3 threshold → no suggestion
        loop.record(_event("compress", True))
        loop.record(_event("compress", False))
        loop.record(_event("compress", False))
        loop.record(_event("compress", False))
        loop.record(_event("compress", False))
        adjustments = loop.suggest_adjustments()
        assert "compress" not in adjustments

    def test_high_retrieval_rate_triggers_adjustment(self):
        loop = FeedbackLoop()
        # 80% retrieval rate — above 0.3 → should suggest
        for _ in range(8):
            loop.record(_event("compress", True))
        for _ in range(2):
            loop.record(_event("compress", False))
        adjustments = loop.suggest_adjustments()
        assert "compress" in adjustments
        assert adjustments["compress"] > 0.0

    def test_adjustment_scales_with_retrieval_rate(self):
        loop_high = FeedbackLoop()
        loop_low = FeedbackLoop()

        # 90% retrieval
        for _ in range(9):
            loop_high.record(_event("s", True))
        loop_high.record(_event("s", False))

        # 40% retrieval — just above threshold
        for _ in range(4):
            loop_low.record(_event("s", True))
        for _ in range(6):
            loop_low.record(_event("s", False))

        adj_high = loop_high.suggest_adjustments().get("s", 0.0)
        adj_low = loop_low.suggest_adjustments().get("s", 0.0)

        assert adj_high > adj_low

    def test_independent_adjustments_per_stage(self):
        loop = FeedbackLoop()
        # stage_a: 90% retrieval → adjustment
        for _ in range(9):
            loop.record(_event("stage_a", True))
        loop.record(_event("stage_a", False))
        # stage_b: 10% retrieval → no adjustment
        loop.record(_event("stage_b", True))
        for _ in range(9):
            loop.record(_event("stage_b", False))

        adjustments = loop.suggest_adjustments()
        assert "stage_a" in adjustments
        assert "stage_b" not in adjustments

    def test_adjustment_value_is_positive(self):
        loop = FeedbackLoop()
        for _ in range(10):
            loop.record(_event("s", True))
        adj = loop.suggest_adjustments()
        assert adj["s"] > 0.0


# ---------------------------------------------------------------------------
# FeedbackLoop.export_stats
# ---------------------------------------------------------------------------

class TestExportStats:
    def test_empty_loop_stats(self):
        loop = FeedbackLoop()
        stats = loop.export_stats()
        assert stats["total_events"] == 0
        assert stats["total_retrieved"] == 0
        assert stats["overall_retrieval_rate"] == 0.0
        assert stats["per_stage"] == {}

    def test_export_stats_has_required_keys(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True))
        stats = loop.export_stats()
        required = {
            "window_size", "total_events", "total_retrieved",
            "overall_retrieval_rate", "per_stage", "adjustments",
        }
        assert required.issubset(stats.keys())

    def test_total_events_count(self):
        loop = FeedbackLoop()
        for _ in range(7):
            loop.record(_event("s", False))
        stats = loop.export_stats()
        assert stats["total_events"] == 7

    def test_total_retrieved_count(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True))
        loop.record(_event("s", True))
        loop.record(_event("s", False))
        stats = loop.export_stats()
        assert stats["total_retrieved"] == 2

    def test_overall_retrieval_rate_in_stats(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True))
        loop.record(_event("s", False))
        stats = loop.export_stats()
        assert stats["overall_retrieval_rate"] == pytest.approx(0.5)

    def test_per_stage_keys_present(self):
        loop = FeedbackLoop()
        loop.record(_event("alpha", True))
        loop.record(_event("beta", False))
        stats = loop.export_stats()
        assert "alpha" in stats["per_stage"]
        assert "beta" in stats["per_stage"]

    def test_per_stage_event_count(self):
        loop = FeedbackLoop()
        loop.record(_event("alpha", True))
        loop.record(_event("alpha", False))
        loop.record(_event("beta", True))
        stats = loop.export_stats()
        assert stats["per_stage"]["alpha"]["event_count"] == 2
        assert stats["per_stage"]["beta"]["event_count"] == 1

    def test_per_stage_avg_compression_ratio(self):
        loop = FeedbackLoop()
        loop.record(_event("s", True, compression_ratio=2.0))
        loop.record(_event("s", False, compression_ratio=4.0))
        stats = loop.export_stats()
        assert stats["per_stage"]["s"]["avg_compression_ratio"] == pytest.approx(3.0)

    def test_window_size_in_stats(self):
        loop = FeedbackLoop(window_size=42)
        stats = loop.export_stats()
        assert stats["window_size"] == 42

    def test_adjustments_in_stats_match_suggest_adjustments(self):
        loop = FeedbackLoop()
        for _ in range(9):
            loop.record(_event("s", True))
        loop.record(_event("s", False))
        stats = loop.export_stats()
        assert stats["adjustments"] == loop.suggest_adjustments()
