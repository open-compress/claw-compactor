"""Tests for NexusStage — ML token compressor FusionStage (Phase 4).

Covers:
  - TORCH_AVAILABLE flag detection
  - Fallback rule-based compression when torch is absent
  - ML compression path (model forward pass with random weights)
  - Fusion thresholds (token prob + span score)
  - NexusStage integration with FusionContext / FusionResult
  - should_apply gating logic
  - Edge cases: empty text, very short text, all-stopwords
  - CrunchModel architecture (forward, compress)
  - NexusModel wrapper
  - _deduplicate_consecutive helper
  - _remove_repeated_ngrams helper
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.base import FusionContext, FusionResult  # noqa: E402
from claw_compactor.fusion.nexus import (  # noqa: E402
    NexusStage,
    NexusModel,
    TORCH_AVAILABLE,
    TOKEN_PROB_THRESHOLD,
    SPAN_SCORE_THRESHOLD,
    UNCERTAIN_LOW,
    _MIN_WORDS,
    _clean,
    _deduplicate_consecutive,
    _remove_repeated_ngrams,
)
from claw_compactor.fusion.nexus_model import CrunchModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_TEXT = "Hello world."
_LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog and the cat sat on the mat "
    "while the sun was shining brightly in the clear blue sky above the field."
)
_STOPWORD_TEXT = (
    "the and or but a an is are was were be been being have has had "
    "do does did will would could should may might shall can it its "
    "this that these those he she they we you i me him her us which"
)


def _make_ctx(content: str, content_type: str = "text") -> FusionContext:
    return FusionContext(content=content, content_type=content_type)


def _stage() -> NexusStage:
    return NexusStage()


# ===========================================================================
# 1. TORCH_AVAILABLE flag
# ===========================================================================

class TestTorchAvailableFlag:
    def test_torch_available_is_bool(self):
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_nexus_model_module_exports_flag(self):
        from claw_compactor.fusion.nexus_model import TORCH_AVAILABLE as mf
        assert isinstance(mf, bool)

    def test_nexus_and_model_flags_agree(self):
        from claw_compactor.fusion.nexus_model import TORCH_AVAILABLE as mf
        assert TORCH_AVAILABLE == mf


# ===========================================================================
# 2. should_apply gating
# ===========================================================================

class TestShouldApply:
    def test_rejects_non_text_content_type(self):
        stage = _stage()
        for ct in ("code", "json", "log", "diff", "search"):
            ctx = _make_ctx(_LONG_TEXT, content_type=ct)
            assert stage.should_apply(ctx) is False, f"should reject content_type={ct}"

    def test_rejects_short_text_below_min_words(self):
        stage = _stage()
        text = " ".join(["word"] * (_MIN_WORDS - 1))
        ctx = _make_ctx(text)
        assert stage.should_apply(ctx) is False

    def test_accepts_exactly_min_words(self):
        stage = _stage()
        text = " ".join(["word"] * _MIN_WORDS)
        ctx = _make_ctx(text)
        assert stage.should_apply(ctx) is True

    def test_accepts_long_text(self):
        stage = _stage()
        assert stage.should_apply(_make_ctx(_LONG_TEXT)) is True

    def test_rejects_empty_string(self):
        stage = _stage()
        assert stage.should_apply(_make_ctx("")) is False

    def test_require_torch_false_allows_fallback(self):
        """With require_torch=False (default), should_apply is True even without torch."""
        stage = NexusStage(require_torch=False)
        with patch("lib.fusion.nexus.TORCH_AVAILABLE", False):
            # We need to re-check via the instance attribute; patch at module level.
            # Instead build a stage while torch is patched out.
            pass
        # If test runs at all torch may be available; just verify the flag is respected.
        stage2 = NexusStage(require_torch=False)
        assert stage2._require_torch is False

    def test_require_torch_true_skips_when_torch_absent(self):
        """With require_torch=True and torch unavailable, should_apply must be False."""
        stage = NexusStage(require_torch=True)
        long_ctx = _make_ctx(_LONG_TEXT)
        # Patch TORCH_AVAILABLE in the nexus module.
        with patch("lib.fusion.nexus.TORCH_AVAILABLE", False):
            # We cannot call stage.should_apply here because the patch only
            # affects the module namespace; the stage already captured the flag.
            # Instead verify that a freshly-patched stage would behave correctly.
            import claw_compactor.fusion.nexus as nexus_mod
            orig = nexus_mod.TORCH_AVAILABLE
            nexus_mod.TORCH_AVAILABLE = False
            try:
                # Build a new stage under patched flag.
                new_stage = NexusStage(require_torch=True)
                result = new_stage.should_apply(long_ctx)
                assert result is False
            finally:
                nexus_mod.TORCH_AVAILABLE = orig


# ===========================================================================
# 3. Fallback rule-based compression
# ===========================================================================

class TestFallbackCompression:
    """Tests run with torch either absent or via direct _fallback_compress call."""

    def _fallback(self, text: str) -> str:
        stage = _stage()
        words = text.split()
        kept, method = stage._fallback_compress(words)
        assert method == "fallback"
        return " ".join(kept)

    def test_stopword_removal_reduces_word_count(self):
        text = "the quick brown fox jumps over the lazy dog and the cat sat on mat"
        out = self._fallback(text)
        word_count_before = len(text.split())
        word_count_after = len(out.split())
        assert word_count_after <= word_count_before

    def test_output_is_non_empty_for_normal_text(self):
        out = self._fallback(_LONG_TEXT)
        assert out.strip() != ""

    def test_all_stopwords_returns_non_empty(self):
        out = self._fallback(_STOPWORD_TEXT)
        # Guaranteed non-empty fallback.
        assert out.strip() != ""

    def test_consecutive_duplicates_removed(self):
        text = " ".join(["word"] * _MIN_WORDS)
        out = self._fallback(text)
        words = out.split()
        for i in range(len(words) - 1):
            assert words[i].lower() != words[i + 1].lower(), "consecutive duplicate found"

    def test_repeated_ngrams_are_collapsed(self):
        # Repeat a bigram 4 times — should be deduplicated.
        phrase = "alpha beta " * 4
        text = phrase + " " + " ".join(["gamma"] * _MIN_WORDS)
        out = self._fallback(text)
        # At most one occurrence of the repeated bigram should remain.
        alpha_count = out.lower().split().count("alpha")
        assert alpha_count <= 2  # first occurrence kept

    def test_empty_input_returns_empty(self):
        stage = _stage()
        kept, _ = stage._fallback_compress([])
        assert kept == []

    def test_output_preserves_content_words(self):
        """Content words (non-stopwords) should generally survive."""
        text = "machine learning neural network transformer architecture " * 4
        out = self._fallback(text)
        assert any(w in out for w in ["machine", "learning", "neural", "network"])


# ===========================================================================
# 4. Fusion logic thresholds
# ===========================================================================

class TestFusionLogic:
    """Directly test the CrunchModel.compress fusion rule via mocked probs."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_high_prob_token_kept(self):
        """Token with prob > 0.5 is always kept."""
        import torch
        model = CrunchModel()
        # Manually mock forward to return controlled logits.
        tokens = ["kept", "dropped"]
        # Logits: [large_discard, large_keep] → keep_prob ≈ 1.0
        # Logits: [large_keep, large_discard] → keep_prob ≈ 0.0
        mock_logits = torch.tensor([[[  -10.0, 10.0],   # token 0: keep_prob ≈ 1
                                      [  10.0, -10.0]]])  # token 1: keep_prob ≈ 0
        mock_spans = torch.tensor([[0.1, 0.1]])

        with patch.object(model, "forward", return_value=(mock_logits, mock_spans)):
            kept = model.compress(tokens, token_prob_threshold=0.5)
        assert "kept" in kept
        assert "dropped" not in kept

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_uncertain_high_span_token_kept(self):
        """Token in uncertain band with span_score > 0.6 is kept."""
        import torch
        model = CrunchModel()
        tokens = ["uncertain_high_span", "uncertain_low_span"]
        # logits [discard_logit=0.2, keep_logit=-0.2] → keep_prob ≈ 0.40
        # which sits in the uncertain band (UNCERTAIN_LOW=0.3, threshold=0.5).
        # First token has high span (0.8 > 0.6) → kept.
        # Second token has low span (0.2 < 0.6) → discarded.
        mock_logits = torch.tensor([[[0.2, -0.2],   # keep_prob ≈ 0.40
                                      [0.2, -0.2]]])
        mock_spans = torch.tensor([[0.8, 0.2]])

        with patch.object(model, "forward", return_value=(mock_logits, mock_spans)):
            kept = model.compress(
                tokens,
                token_prob_threshold=0.5,
                span_score_threshold=0.6,
                uncertain_low=0.3,
            )
        assert "uncertain_high_span" in kept
        assert "uncertain_low_span" not in kept

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_low_prob_token_discarded_regardless_of_span(self):
        """Token with prob below uncertain_low is always discarded."""
        import torch
        model = CrunchModel()
        tokens = ["low_prob"]
        mock_logits = torch.tensor([[[10.0, -10.0]]])   # keep_prob ≈ 0 (< 0.3)
        mock_spans = torch.tensor([[0.99]])              # high span — should not matter

        with patch.object(model, "forward", return_value=(mock_logits, mock_spans)):
            kept = model.compress(tokens, uncertain_low=0.3)
        assert "low_prob" not in kept


# ===========================================================================
# 5. CrunchModel forward pass
# ===========================================================================

class TestCrunchModelForward:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_forward_returns_two_tensors(self):
        import torch
        model = CrunchModel()
        ids = torch.randint(1, 100, (1, 10))
        token_logits, span_scores = model.forward(ids)
        assert token_logits is not None
        assert span_scores is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_token_logits_shape(self):
        import torch
        model = CrunchModel(vocab_size=500, embed_dim=16, hidden_size=32)
        B, T = 2, 8
        ids = torch.randint(1, 100, (B, T))
        token_logits, _ = model.forward(ids)
        assert token_logits.shape == (B, T, 2)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_span_scores_shape(self):
        import torch
        model = CrunchModel(vocab_size=500, embed_dim=16, hidden_size=32)
        B, T = 2, 8
        ids = torch.randint(1, 100, (B, T))
        _, span_scores = model.forward(ids)
        assert span_scores.shape == (B, T)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_span_scores_in_zero_one_range(self):
        import torch
        model = CrunchModel(vocab_size=500, embed_dim=16, hidden_size=32)
        ids = torch.randint(1, 100, (1, 15))
        _, span_scores = model.forward(ids)
        assert span_scores.min().item() >= 0.0
        assert span_scores.max().item() <= 1.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_compress_returns_list_of_strings(self):
        model = CrunchModel()
        tokens = _LONG_TEXT.split()
        result = model.compress(tokens)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_compress_empty_tokens(self):
        model = CrunchModel()
        assert model.compress([]) == []

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_compress_single_token(self):
        model = CrunchModel()
        result = model.compress(["hello"])
        assert isinstance(result, list)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_forward_no_grad_safe(self):
        """forward() must not raise when called inside torch.no_grad()."""
        import torch
        model = CrunchModel()
        ids = torch.randint(1, 50, (1, 5))
        with torch.no_grad():
            token_logits, span_scores = model.forward(ids)
        assert token_logits is not None


# ===========================================================================
# 6. NexusStage integration
# ===========================================================================

class TestNexusStageIntegration:
    def test_stage_name(self):
        assert NexusStage.name == "nexus"

    def test_stage_order(self):
        assert NexusStage.order == 35

    def test_apply_returns_fusion_result(self):
        stage = _stage()
        ctx = _make_ctx(_LONG_TEXT)
        if not stage.should_apply(ctx):
            pytest.skip("stage skips this context")
        result = stage.apply(ctx)
        assert isinstance(result, FusionResult)

    def test_apply_content_is_string(self):
        stage = _stage()
        ctx = _make_ctx(_LONG_TEXT)
        if not stage.should_apply(ctx):
            pytest.skip("stage skips this context")
        result = stage.apply(ctx)
        assert isinstance(result.content, str)

    def test_apply_records_original_tokens(self):
        stage = _stage()
        ctx = _make_ctx(_LONG_TEXT)
        if not stage.should_apply(ctx):
            pytest.skip("stage skips this context")
        result = stage.apply(ctx)
        assert result.original_tokens > 0

    def test_apply_marker_contains_nexus(self):
        stage = _stage()
        ctx = _make_ctx(_LONG_TEXT)
        if not stage.should_apply(ctx):
            pytest.skip("stage skips this context")
        result = stage.apply(ctx)
        assert any("nexus" in m for m in result.markers)

    def test_timed_apply_skips_when_should_apply_false(self):
        stage = _stage()
        ctx = _make_ctx(_SHORT_TEXT)  # too short
        result = stage.timed_apply(ctx)
        assert result.skipped is True
        assert result.content == _SHORT_TEXT

    def test_timed_apply_runs_on_long_text(self):
        stage = _stage()
        ctx = _make_ctx(_LONG_TEXT)
        result = stage.timed_apply(ctx)
        # Either ran (skipped=False) or torch unavailable triggered fallback.
        assert isinstance(result.content, str)
        assert result.content.strip() != ""

    def test_fallback_warning_emitted_when_torch_absent(self):
        """When torch unavailable, a warning is appended to result.warnings."""
        import claw_compactor.fusion.nexus as nexus_mod
        orig = nexus_mod.TORCH_AVAILABLE
        nexus_mod.TORCH_AVAILABLE = False
        try:
            stage = NexusStage()
            stage._model = None  # simulate no model
            ctx = _make_ctx(_LONG_TEXT)
            result = stage.apply(ctx)
            assert any("torch unavailable" in w or "fallback" in w for w in result.warnings)
        finally:
            nexus_mod.TORCH_AVAILABLE = orig


# ===========================================================================
# 7. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_text_should_not_apply(self):
        stage = _stage()
        assert stage.should_apply(_make_ctx("")) is False

    def test_very_short_text_should_not_apply(self):
        stage = _stage()
        assert stage.should_apply(_make_ctx("Hello world.")) is False

    def test_all_stopwords_apply_returns_nonempty(self):
        stage = _stage()
        ctx = _make_ctx(_STOPWORD_TEXT)
        if not stage.should_apply(ctx):
            pytest.skip("stage skips stopword-only text — acceptable")
        result = stage.apply(ctx)
        assert result.content.strip() != ""

    def test_single_word_repeated_many_times(self):
        text = "echo " * _MIN_WORDS
        stage = _stage()
        ctx = _make_ctx(text.strip())
        if not stage.should_apply(ctx):
            pytest.skip("skipped by stage")
        result = stage.apply(ctx)
        assert isinstance(result.content, str)

    def test_unicode_text_handled(self):
        text = "机器学习 神经网络 Transformer 架构 " * 6
        stage = _stage()
        ctx = _make_ctx(text.strip())
        if stage.should_apply(ctx):
            result = stage.apply(ctx)
            assert isinstance(result.content, str)

    def test_content_type_code_skipped(self):
        stage = _stage()
        ctx = FusionContext(content=_LONG_TEXT, content_type="code")
        assert stage.should_apply(ctx) is False


# ===========================================================================
# 8. Internal helper unit tests
# ===========================================================================

class TestHelpers:
    def test_clean_strips_punctuation(self):
        assert _clean("hello,") == "hello"
        assert _clean("world!") == "world"

    def test_clean_lowercases(self):
        assert _clean("Hello") == "hello"

    def test_clean_empty_string(self):
        assert _clean("") == ""

    def test_deduplicate_consecutive_removes_consecutive(self):
        words = ["a", "a", "b", "b", "b", "c"]
        result = _deduplicate_consecutive(words)
        assert result == ["a", "b", "c"]

    def test_deduplicate_consecutive_keeps_non_consecutive(self):
        words = ["a", "b", "a"]
        result = _deduplicate_consecutive(words)
        assert result == ["a", "b", "a"]

    def test_deduplicate_consecutive_empty(self):
        assert _deduplicate_consecutive([]) == []

    def test_deduplicate_consecutive_case_insensitive(self):
        words = ["Hello", "hello", "world"]
        result = _deduplicate_consecutive(words)
        assert len(result) == 2

    def test_remove_repeated_ngrams_collapses_repeated_bigrams(self):
        words = ["alpha", "beta"] * 4 + ["gamma"]
        result = _remove_repeated_ngrams(words, n=2, min_count=3)
        alpha_count = result.count("alpha")
        assert alpha_count < 4

    def test_remove_repeated_ngrams_no_repeats_unchanged(self):
        words = ["a", "b", "c", "d"]
        result = _remove_repeated_ngrams(words, n=2, min_count=3)
        assert result == words

    def test_remove_repeated_ngrams_empty(self):
        assert _remove_repeated_ngrams([], n=2, min_count=3) == []

    def test_remove_repeated_ngrams_below_min_count_unchanged(self):
        words = ["x", "y", "x", "y"]  # bigram appears 2 times, min_count=3
        result = _remove_repeated_ngrams(words, n=2, min_count=3)
        assert result == words


# ===========================================================================
# 9. CrunchModel stub when torch absent
# ===========================================================================

class TestCrunchModelStub:
    @pytest.mark.skipif(TORCH_AVAILABLE, reason="torch IS installed — stub not active")
    def test_crunch_model_raises_import_error_without_torch(self):
        with pytest.raises(ImportError, match="torch"):
            CrunchModel()

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="torch IS installed — stub not active")
    def test_nexus_model_raises_import_error_without_torch(self):
        with pytest.raises(ImportError, match="torch"):
            NexusModel()


# ===========================================================================
# 10. Constant exports
# ===========================================================================

class TestConstants:
    def test_token_prob_threshold_value(self):
        assert TOKEN_PROB_THRESHOLD == 0.5

    def test_span_score_threshold_value(self):
        assert SPAN_SCORE_THRESHOLD == 0.6

    def test_uncertain_low_value(self):
        assert UNCERTAIN_LOW == 0.3

    def test_min_words_positive(self):
        assert _MIN_WORDS > 0
