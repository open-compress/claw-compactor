"""FeedbackLoop: track Rewind retrieval events and auto-adjust compression rates.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# If retrieval rate for a stage exceeds this fraction, suggest backing off.
_HIGH_RETRIEVAL_THRESHOLD = 0.3

# Default suggested compression-rate reduction when threshold is exceeded.
_DEFAULT_REDUCTION = 0.1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalEvent:
    """Immutable record of one Rewind retrieval observation."""
    hash_id: str
    stage_name: str        # which FusionStage produced this compressed chunk
    compression_ratio: float
    was_retrieved: bool    # True if the LLM actually called rewind_retrieve
    timestamp: float       # monotonic seconds


# ---------------------------------------------------------------------------
# FeedbackLoop
# ---------------------------------------------------------------------------

class FeedbackLoop:
    """Track Rewind retrieval events and auto-adjust compression rates.

    Maintains a sliding window of the last *window_size* events.  When the
    retrieval rate for a stage exceeds the high-retrieval threshold (0.3) the
    loop recommends reducing that stage's compression rate by
    ``_DEFAULT_REDUCTION`` (10 percentage points).
    """

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._events: deque[RetrievalEvent] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event: RetrievalEvent) -> None:
        """Append *event* to the sliding window.

        Once the window is full, the oldest event is automatically evicted.
        """
        self._events.append(event)

    def retrieval_rate(self, stage_name: str | None = None) -> float:
        """Return the fraction of events where ``was_retrieved`` is True.

        Args:
            stage_name: When provided, only events for that stage are
                        considered.  When ``None``, all events are included.

        Returns:
            A float in [0.0, 1.0].  Returns 0.0 when there are no matching
            events (avoids division-by-zero).
        """
        events = self._filter(stage_name)
        if not events:
            return 0.0
        retrieved_count = sum(1 for e in events if e.was_retrieved)
        return retrieved_count / len(events)

    def suggest_adjustments(self) -> dict[str, float]:
        """Return per-stage suggested compression-rate reductions.

        For each stage whose retrieval rate exceeds the high-retrieval
        threshold (0.3), the suggested reduction is:

            suggested_reduction = _DEFAULT_REDUCTION
                                  * (retrieval_rate / _HIGH_RETRIEVAL_THRESHOLD)

        Stages below the threshold are omitted from the result dict.

        Returns:
            ``{stage_name: reduction_amount}`` where the reduction is a
            positive float (e.g. 0.1 means "reduce compression rate by 10%").
        """
        stage_names = {e.stage_name for e in self._events}
        adjustments: dict[str, float] = {}
        for stage in stage_names:
            rate = self.retrieval_rate(stage)
            if rate > _HIGH_RETRIEVAL_THRESHOLD:
                # Scale reduction proportionally to how far over threshold we are
                reduction = _DEFAULT_REDUCTION * (rate / _HIGH_RETRIEVAL_THRESHOLD)
                adjustments[stage] = round(reduction, 6)
        return adjustments

    def export_stats(self) -> dict:
        """Return summary statistics for monitoring / dashboards."""
        stage_names = sorted({e.stage_name for e in self._events})
        per_stage: dict[str, dict] = {}
        for stage in stage_names:
            events = self._filter(stage)
            retrieved = sum(1 for e in events if e.was_retrieved)
            ratios = [e.compression_ratio for e in events]
            per_stage[stage] = {
                "event_count":    len(events),
                "retrieved_count": retrieved,
                "retrieval_rate": retrieved / len(events) if events else 0.0,
                "avg_compression_ratio": (
                    sum(ratios) / len(ratios) if ratios else 0.0
                ),
            }

        total_events = len(self._events)
        total_retrieved = sum(1 for e in self._events if e.was_retrieved)

        return {
            "window_size":       self._window_size,
            "total_events":      total_events,
            "total_retrieved":   total_retrieved,
            "overall_retrieval_rate": (
                total_retrieved / total_events if total_events else 0.0
            ),
            "per_stage":         per_stage,
            "adjustments":       self.suggest_adjustments(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(self, stage_name: str | None) -> list[RetrievalEvent]:
        """Return events matching *stage_name*, or all events when None."""
        if stage_name is None:
            return list(self._events)
        return [e for e in self._events if e.stage_name == stage_name]
