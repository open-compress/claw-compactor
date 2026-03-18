"""Nexus ML model architecture — CrunchModel dual-head token classifier.

Provides:
  - CrunchModel(nn.Module): backbone + token_head + span_head
  - forward() returning token_logits and span_scores
  - compress() running inference and filtering tokens

When torch is unavailable the module exports stub classes that raise
ImportError on instantiation, so callers can guard with TORCH_AVAILABLE.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
_HIDDEN_SIZE = 128   # lightweight backbone hidden dim (mock/test-safe)
_SPAN_KERNEL = 3     # 1-D CNN kernel size for span head
_NUM_LABELS = 2      # keep / discard


# ---------------------------------------------------------------------------
# CrunchModel — only defined when torch is present
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class CrunchModel(nn.Module):  # type: ignore[misc]
        """Dual-head ModernBERT-style token classifier.

        Architecture:
          backbone  — 2-layer bidirectional GRU over token embeddings
          token_head — linear → 2-class logits (keep / discard) per token
          span_head  — 1-D CNN → scalar importance score per token position

        The backbone is intentionally small so tests run on CPU with random
        weights in milliseconds.  In production the backbone would be replaced
        by a pretrained ModernBERT encoder.
        """

        def __init__(
            self,
            vocab_size: int = 30522,   # default BERT vocab size
            embed_dim: int = 64,
            hidden_size: int = _HIDDEN_SIZE,
            num_labels: int = _NUM_LABELS,
            span_kernel: int = _SPAN_KERNEL,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.backbone = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            backbone_out = hidden_size * 2  # bidirectional

            # Token head: per-token binary classification
            self.token_head = nn.Linear(backbone_out, num_labels)

            # Span head: 1-D CNN over backbone output → importance scalar
            self.span_conv = nn.Conv1d(
                in_channels=backbone_out,
                out_channels=1,
                kernel_size=span_kernel,
                padding=span_kernel // 2,
            )

        def forward(
            self,
            input_ids: "torch.Tensor",  # (B, T)
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            """Return (token_logits, span_scores).

            token_logits : (B, T, num_labels)  — raw logits for keep/discard
            span_scores  : (B, T)              — importance score in [0, 1]
            """
            emb = self.embedding(input_ids)           # (B, T, E)
            hidden, _ = self.backbone(emb)             # (B, T, 2*H)

            token_logits = self.token_head(hidden)     # (B, T, 2)

            # Span head needs (B, C, T) channel-first layout
            hidden_t = hidden.transpose(1, 2)          # (B, 2*H, T)
            span_raw = self.span_conv(hidden_t)        # (B, 1, T)
            span_scores = torch.sigmoid(
                span_raw.squeeze(1)                    # (B, T)
            )

            return token_logits, span_scores

        def compress(
            self,
            tokens: list[str],
            token_prob_threshold: float = 0.5,
            span_score_threshold: float = 0.6,
            uncertain_low: float = 0.3,
        ) -> list[str]:
            """Run inference and return the filtered token list.

            Fusion rule:
              keep if token_prob > token_prob_threshold
              OR (uncertain_low < token_prob < token_prob_threshold
                  AND span_score > span_score_threshold)

            Args:
                tokens: whitespace-split word tokens.
                token_prob_threshold: minimum keep-class probability to keep
                    a token outright.
                span_score_threshold: span importance threshold applied in the
                    uncertain band.
                uncertain_low: lower bound of the uncertain probability band.

            Returns:
                Filtered list of kept tokens (same strings, no modifications).
            """
            if not tokens:
                return []

            # Encode tokens as simple char-hash indices (mock tokenizer).
            # In production this would use a real BPE tokenizer.
            input_ids = torch.tensor(
                [[_char_hash(t) for t in tokens]],
                dtype=torch.long,
            )  # (1, T)

            self.eval()
            with torch.no_grad():
                token_logits, span_scores = self.forward(input_ids)

            # token_logits: (1, T, 2) → probabilities
            probs = F.softmax(token_logits, dim=-1)   # (1, T, 2)
            keep_probs = probs[0, :, 1].tolist()       # (T,) — prob of keep class
            span_vals = span_scores[0, :].tolist()     # (T,)

            kept: list[str] = []
            for token, kp, sv in zip(tokens, keep_probs, span_vals):
                if kp > token_prob_threshold:
                    kept.append(token)
                elif uncertain_low < kp and sv > span_score_threshold:
                    kept.append(token)
                # else: discard

            return kept

else:
    # Stub so `from lib.fusion.nexus_model import CrunchModel` always works.
    class CrunchModel:  # type: ignore[no-redef]
        """Stub — torch is not installed."""

        def __init__(self, *args, **kwargs):  # noqa: ANN204
            raise ImportError(
                "CrunchModel requires torch. Install it with: pip install torch"
            )


# ---------------------------------------------------------------------------
# Utility: simple hash-based mock tokenizer
# ---------------------------------------------------------------------------

def _char_hash(token: str, vocab_size: int = 30522) -> int:
    """Map a token string to a vocabulary index via its string hash."""
    return (hash(token) & 0x7FFF_FFFF) % max(1, vocab_size - 1) + 1
