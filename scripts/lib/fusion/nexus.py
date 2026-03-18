"""Nexus — ML-powered token compressor FusionStage (order=35).

Uses a dual-head ModernBERT-style classifier (CrunchModel) to make
keep/discard decisions for each token in a text passage.

When torch is unavailable the stage falls back to a rule-based heuristic
compressor (stopword removal + repetition detection) so the pipeline stays
functional without heavy ML dependencies.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re
from typing import Any

from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult

try:
    from claw_compactor.tokens import estimate_tokens  # type: ignore[import]
except ImportError:  # pragma: no cover — tokens module may not exist yet
    def estimate_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text.split()))

# ---------------------------------------------------------------------------
# Optional torch / transformers import
# ---------------------------------------------------------------------------
TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Import CrunchModel regardless — it has its own graceful stub when torch
# is absent.  We gate actual instantiation on TORCH_AVAILABLE.
from claw_compactor.fusion.nexus_model import CrunchModel  # noqa: E402

# ---------------------------------------------------------------------------
# Rule-based fallback constants
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "which", "who", "whom", "what", "where", "when", "how",
})

# Minimum word count before NexusStage runs.
_MIN_WORDS = 20

# Fusion thresholds (also used to expose for testing).
TOKEN_PROB_THRESHOLD = 0.5
SPAN_SCORE_THRESHOLD = 0.6
UNCERTAIN_LOW = 0.3


# ---------------------------------------------------------------------------
# NexusModel — thin wrapper around CrunchModel
# ---------------------------------------------------------------------------

class NexusModel:
    """Dual-head ModernBERT token classifier for keep/discard decisions.

    Wraps CrunchModel with configurable fusion thresholds.

    Fusion rule applied per token t_i with keep-class probability p_i and
    span importance score s_i:

      keep  ← p_i > TOKEN_PROB_THRESHOLD
      keep  ← UNCERTAIN_LOW < p_i ≤ TOKEN_PROB_THRESHOLD AND s_i > SPAN_SCORE_THRESHOLD
      discard otherwise
    """

    def __init__(
        self,
        token_prob_threshold: float = TOKEN_PROB_THRESHOLD,
        span_score_threshold: float = SPAN_SCORE_THRESHOLD,
        uncertain_low: float = UNCERTAIN_LOW,
        **model_kwargs: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "NexusModel requires torch. Install it with: pip install torch"
            )
        self._model = CrunchModel(**model_kwargs)
        self._token_prob_threshold = token_prob_threshold
        self._span_score_threshold = span_score_threshold
        self._uncertain_low = uncertain_low

    def compress(self, tokens: list[str]) -> list[str]:
        """Return the subset of *tokens* that the model decides to keep."""
        return self._model.compress(
            tokens,
            token_prob_threshold=self._token_prob_threshold,
            span_score_threshold=self._span_score_threshold,
            uncertain_low=self._uncertain_low,
        )


# ---------------------------------------------------------------------------
# NexusStage
# ---------------------------------------------------------------------------

class NexusStage(FusionStage):
    """ML-powered token compressor FusionStage.

    - Uses NexusModel (CrunchModel) when torch is available.
    - Falls back to rule-based compression (stopword removal + repetition
      detection) when torch is absent, so the pipeline still runs.
    - Skips entirely (should_apply → False) when torch is absent AND
      the caller has set require_torch=True in the constructor.

    Ordering: 35 (after Cortex=5, Neurosyntax=25; before later dedup stages).
    """

    name = "nexus"
    order = 35

    def __init__(
        self,
        require_torch: bool = False,
        token_prob_threshold: float = TOKEN_PROB_THRESHOLD,
        span_score_threshold: float = SPAN_SCORE_THRESHOLD,
        uncertain_low: float = UNCERTAIN_LOW,
    ) -> None:
        self._require_torch = require_torch
        self._token_prob_threshold = token_prob_threshold
        self._span_score_threshold = span_score_threshold
        self._uncertain_low = uncertain_low
        self._model: NexusModel | None = None

        if TORCH_AVAILABLE:
            self._model = NexusModel(
                token_prob_threshold=token_prob_threshold,
                span_score_threshold=span_score_threshold,
                uncertain_low=uncertain_low,
            )

    # ------------------------------------------------------------------
    # FusionStage interface
    # ------------------------------------------------------------------

    def should_apply(self, ctx: FusionContext) -> bool:
        """Return True when the stage should run.

        Conditions:
          1. content_type must be "text"
          2. content must contain at least _MIN_WORDS words
          3. If require_torch=True, torch must be available.
             If require_torch=False (default), falls back gracefully.
        """
        if ctx.content_type != "text":
            return False
        if len(ctx.content.split()) < _MIN_WORDS:
            return False
        if self._require_torch and not TORCH_AVAILABLE:
            return False
        return True

    def apply(self, ctx: FusionContext) -> FusionResult:
        """Apply ML or rule-based token compression."""
        original_tokens = estimate_tokens(ctx.content)
        words = ctx.content.split()

        warnings: list[str] = []

        if TORCH_AVAILABLE and self._model is not None:
            kept_words, method = self._ml_compress(words)
        else:
            kept_words, method = self._fallback_compress(words)
            warnings.append(
                "nexus: torch unavailable — used rule-based fallback compression"
            )

        compressed = " ".join(kept_words)
        compressed_tokens = estimate_tokens(compressed)

        return FusionResult(
            content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=[f"nexus:{method}"],
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # ML compression path
    # ------------------------------------------------------------------

    def _ml_compress(self, words: list[str]) -> tuple[list[str], str]:
        """Run CrunchModel inference and return (kept_words, method_label)."""
        assert self._model is not None
        kept = self._model.compress(words)
        # Always keep at least one word to avoid empty output.
        if not kept and words:
            kept = [words[0]]
        return kept, "ml"

    # ------------------------------------------------------------------
    # Rule-based fallback compression
    # ------------------------------------------------------------------

    def _fallback_compress(self, words: list[str]) -> tuple[list[str], str]:
        """Simple heuristic compression: stopword removal + repetition detection."""
        # Phase 1: Remove stop-words (case-insensitive) but keep words that are
        # purely stopwords if the whole sentence would collapse.
        after_stopwords = [
            w for w in words
            if _clean(w) not in _STOPWORDS or not _clean(w)
        ]

        # Ensure we did not over-compress (keep at least 40% of original words)
        if len(after_stopwords) < max(1, len(words) * 0.4):
            after_stopwords = words[:]

        # Phase 2: Remove exact-duplicate consecutive tokens (repetition).
        deduplicated = _deduplicate_consecutive(after_stopwords)

        # Phase 3: Remove repeated n-grams (bigrams that appear 3+ times).
        compressed = _remove_repeated_ngrams(deduplicated, n=2, min_count=3)

        # Guarantee non-empty output.
        if not compressed and words:
            compressed = [words[0]]

        return compressed, "fallback"


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _clean(word: str) -> str:
    """Lowercase and strip punctuation from a word for stopword lookup."""
    return re.sub(r"[^\w]", "", word).lower()


def _deduplicate_consecutive(words: list[str]) -> list[str]:
    """Remove consecutive duplicate tokens (case-insensitive)."""
    if not words:
        return []
    result: list[str] = [words[0]]
    for word in words[1:]:
        if word.lower() != result[-1].lower():
            result.append(word)
    return result


def _remove_repeated_ngrams(
    words: list[str],
    n: int = 2,
    min_count: int = 3,
) -> list[str]:
    """Drop tokens that belong to an n-gram repeated >= min_count times."""
    if len(words) < n:
        return words[:]

    # Count n-gram occurrences.
    ngram_counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(w.lower() for w in words[i : i + n])
        ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    # Find n-grams that exceed the threshold.
    repeated: set[tuple[str, ...]] = {
        gram for gram, count in ngram_counts.items() if count >= min_count
    }
    if not repeated:
        return words[:]

    # Mark positions that are part of a repeated n-gram (keep first occurrence).
    seen_repeated: set[tuple[str, ...]] = set()
    drop_positions: set[int] = set()

    for i in range(len(words) - n + 1):
        gram = tuple(w.lower() for w in words[i : i + n])
        if gram in repeated:
            if gram in seen_repeated:
                for j in range(i, i + n):
                    drop_positions.add(j)
            else:
                seen_repeated.add(gram)

    return [w for i, w in enumerate(words) if i not in drop_positions]
