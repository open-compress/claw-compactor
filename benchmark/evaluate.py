"""evaluate.py — Benchmark evaluation metrics for claw-compactor compression comparison.

Metrics:
1. Token compression ratio = compressed_tokens / original_tokens
2. ROUGE-L (pure Python implementation — no external deps)
3. Information retention F1 (keyword-based)
4. Latency (ms)
5. LLM call count

Python 3.9+ / no external dependencies required.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Token estimation (mirrors lib/tokens.py heuristic — no tiktoken required)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]')

def estimate_tokens(text: str) -> int:
    """CJK-aware token estimator (4 chars/token for ASCII, 1.5 for CJK)."""
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return max(1, int(cjk_chars / 1.5 + other_chars / 4))


# ---------------------------------------------------------------------------
# ROUGE-L — pure Python, no external deps
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute LCS length using DP (O(m*n) time, O(min(m,n)) space)."""
    if not a or not b:
        return 0
    if a == b:
        return len(a)
    m, n = len(a), len(b)
    # Use shorter list as columns for space efficiency
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for ai in a:
        curr = [0] * (n + 1)
        for j, bj in enumerate(b, 1):
            if ai == bj:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(reference: str, hypothesis: str, beta: float = 1.2) -> dict[str, float]:
    """
    Compute ROUGE-L between reference and hypothesis.

    Args:
        reference:   Original (ground-truth) text.
        hypothesis:  Compressed / generated text.
        beta:        F-measure beta (default 1.2 weights recall slightly higher).

    Returns:
        dict with keys 'precision', 'recall', 'f1'.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Information retention (keyword-based F1)
# ---------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Extract top-N significant keywords from text using TF-style scoring.

    Filters common stopwords. Returns lowercase tokens.
    """
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "can", "may", "might", "shall", "this",
        "that", "these", "those", "i", "you", "we", "they", "it", "he", "she",
        "my", "your", "our", "their", "its", "which", "who", "what",
        "when", "where", "how", "why", "if", "as", "so", "not", "no", "also",
        "just", "then", "than", "more", "most", "very", "too", "all", "any",
        "each", "few", "more", "both", "only", "same", "other", "such",
        "into", "after", "before", "about", "above", "through", "during",
        "s", "t", "re", "ll", "ve", "d", "m"
    }
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-\.]*[a-zA-Z0-9]|[a-zA-Z0-9]", text.lower())
    freq: dict[str, int] = {}
    for tok in tokens:
        if tok not in STOPWORDS and len(tok) > 2:
            freq[tok] = freq.get(tok, 0) + 1
    sorted_by_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_by_freq[:top_n]]


def information_retention_f1(original: str, compressed: str, top_n: int = 30) -> dict[str, float]:
    """
    Compute information retention F1.

    Extracts top-N keywords from original text, checks how many appear in compressed.

    Returns:
        dict with 'precision', 'recall', 'f1', 'keywords_original', 'keywords_found'
    """
    orig_keywords = set(extract_keywords(original, top_n=top_n))
    comp_lower = compressed.lower()

    if not orig_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "keywords_original": 0, "keywords_found": 0}

    found = sum(1 for kw in orig_keywords if kw in comp_lower)

    recall = found / len(orig_keywords)

    # Precision: what fraction of compressed content keywords are from original?
    comp_keywords = set(extract_keywords(compressed, top_n=top_n))
    if not comp_keywords:
        precision = 0.0
    else:
        shared = len(orig_keywords & comp_keywords)
        precision = shared / len(comp_keywords)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "keywords_original": len(orig_keywords),
        "keywords_found": found,
    }


# ---------------------------------------------------------------------------
# Main EvaluationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """All metrics for one (sample, compressor) pair."""

    sample_id: str
    compressor_name: str

    # Raw sizes
    original_tokens: int = 0
    compressed_tokens: int = 0

    # Derived
    compression_ratio: float = 0.0   # compressed / original (lower = better compression)
    space_saving_pct: float = 0.0    # (1 - ratio) * 100

    # Quality
    rouge_l: dict = field(default_factory=dict)
    info_retention: dict = field(default_factory=dict)

    # Performance
    latency_ms: float = 0.0
    llm_calls: int = 0

    # Raw text (optional, for debugging)
    compressed_text_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "compressor": self.compressor_name,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "space_saving_pct": round(self.space_saving_pct, 1),
            "rouge_l_f1": self.rouge_l.get("f1", 0.0),
            "rouge_l_precision": self.rouge_l.get("precision", 0.0),
            "rouge_l_recall": self.rouge_l.get("recall", 0.0),
            "info_retention_f1": self.info_retention.get("f1", 0.0),
            "info_retention_recall": self.info_retention.get("recall", 0.0),
            "keywords_original": self.info_retention.get("keywords_original", 0),
            "keywords_found": self.info_retention.get("keywords_found", 0),
            "latency_ms": round(self.latency_ms, 1),
            "llm_calls": self.llm_calls,
            "compressed_preview": self.compressed_text_preview[:200],
        }


def evaluate(
    sample_id: str,
    compressor_name: str,
    original_text: str,
    compressed_text: str,
    latency_ms: float,
    llm_calls: int = 0,
) -> EvaluationResult:
    """Run all metrics and return an EvaluationResult."""
    orig_tokens = estimate_tokens(original_text)
    comp_tokens = estimate_tokens(compressed_text)

    ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 1.0
    saving = (1 - ratio) * 100

    rl = rouge_l(original_text, compressed_text)
    ir = information_retention_f1(original_text, compressed_text)

    return EvaluationResult(
        sample_id=sample_id,
        compressor_name=compressor_name,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        compression_ratio=ratio,
        space_saving_pct=saving,
        rouge_l=rl,
        info_retention=ir,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        compressed_text_preview=compressed_text[:300],
    )


# ---------------------------------------------------------------------------
# Utility: convert messages list to plain text
# ---------------------------------------------------------------------------

def messages_to_text(messages: list[dict]) -> str:
    """Flatten a messages list to a plain text conversation string."""
    parts = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        ts = m.get("ts", "")
        if ts:
            parts.append(f"[{ts}] {role}: {content}")
        else:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Smoke test
    ref = "The quick brown fox jumps over the lazy dog. Dogs are great animals. The fox ran away quickly."
    hyp = "The fox jumped over the dog and ran away."

    rl = rouge_l(ref, hyp)
    ir = information_retention_f1(ref, hyp)
    print(f"ROUGE-L: {rl}")
    print(f"Info retention: {ir}")
    print(f"Original tokens: {estimate_tokens(ref)}")
    print(f"Hypothesis tokens: {estimate_tokens(hyp)}")
    print("evaluate.py smoke test passed ✓")
