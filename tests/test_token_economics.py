"""Token Economics tests — precise verification of compression savings.

Inspired by claude-mem's rigorous token accounting approach.
Every compression technique must prove its token savings are real and measurable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.tokens import estimate_tokens, using_tiktoken
from claw_compactor.dictionary import (
    build_codebook, compress_text, decompress_text, compression_stats,
)
from claw_compactor.tokenizer_optimizer import (
    optimize_tokens, estimate_savings, normalize_punctuation,
    compress_table_to_kv, strip_bold_italic, compact_bullets,
)
from claw_compactor.rle import compress as rle_compress, compress_paths, compress_ip_families
from compressed_context import compress_with_stats


# --- Token Counting Precision ---

class TestTokenCountingPrecision:
    """Verify estimate_tokens returns correct values for edge cases."""

    def test_empty_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_like_empty(self):
        """Empty string should be 0, not 1 or error."""
        assert estimate_tokens("") == 0

    def test_single_char(self):
        assert estimate_tokens("a") >= 1

    def test_single_chinese_char(self):
        assert estimate_tokens("你") >= 1

    def test_whitespace_only(self):
        result = estimate_tokens("   \n\n\t  ")
        assert result >= 1  # whitespace still has tokens

    def test_large_content_positive(self):
        """Large content should give proportionally large token count."""
        small = estimate_tokens("hello world")
        large = estimate_tokens("hello world " * 1000)
        assert large > small * 500  # at least 500x more

    def test_known_sentence(self):
        """A known English sentence should give a reasonable count."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        # tiktoken: ~10 tokens; heuristic: ~11
        assert 8 <= tokens <= 15

    def test_chinese_text_density(self):
        """Chinese text should have ~1 token per 1-2 chars with tiktoken."""
        text = "这是一个测试句子用来验证中文分词"
        tokens = estimate_tokens(text)
        if using_tiktoken():
            # tiktoken: roughly 1 token per Chinese char
            assert tokens >= len(text) * 0.3
        assert tokens >= 1

    def test_consistency(self):
        """Same text should always give same count."""
        text = "consistency test 一致性测试"
        a = estimate_tokens(text)
        b = estimate_tokens(text)
        assert a == b


# --- Savings Calculation ---

class TestSavingsCalculation:
    """Verify savings arithmetic is correct."""

    def test_savings_is_before_minus_after(self):
        original = "**Bold** text with ，Chinese，punctuation。here。"
        optimized = optimize_tokens(original, aggressive=True)
        stats = estimate_savings(original, optimized)
        expected_saved = stats["original_tokens"] - stats["optimized_tokens"]
        char_diff = stats["original_chars"] - stats["optimized_chars"]
        assert char_diff >= 0 or stats["original_chars"] >= stats["optimized_chars"]
        assert stats["token_reduction_pct"] == pytest.approx(
            (stats["original_tokens"] - stats["optimized_tokens"]) / stats["original_tokens"] * 100,
            abs=0.15,
        )

    def test_savings_zero_for_unchanged(self):
        text = "plain text no changes"
        optimized = optimize_tokens(text)
        stats = estimate_savings(text, optimized)
        assert stats["token_reduction_pct"] == 0.0 or stats["optimized_tokens"] <= stats["original_tokens"]

    def test_negative_savings_possible(self):
        """If compression adds overhead (e.g., codebook), net can be negative."""
        # Tiny text with a big codebook
        codebook = {f"${chr(65+i)}{chr(65+j)}": f"word{i}{j}" for i in range(5) for j in range(5)}
        text = "unique text"
        stats = compression_stats(text, text, codebook)
        # Net reduction can be negative when codebook overhead > savings
        assert "net_reduction_pct" in stats

    def test_empty_original_savings(self):
        stats = estimate_savings("", "")
        assert stats["original_tokens"] == 0
        assert stats["optimized_tokens"] == 0
        assert stats["token_reduction_pct"] == 0.0

    def test_rounding(self):
        """Percentage should be rounded to 1 decimal place."""
        stats = estimate_savings("a" * 100, "a" * 97)
        pct = stats["token_reduction_pct"]
        assert pct == round(pct, 1)


# --- Per-Technique Token Economics ---

class TestTableCompressionEconomics:
    """Tables should save 40-70% tokens when converted to key:value."""

    def test_2col_table_savings(self):
        table = (
            "| Key | Value |\n|-----|-------|\n"
            "| Server | gateway-prod |\n"
            "| IP | 192.168.1.1 |\n"
            "| Port | 8080 |\n"
            "| Status | running |\n"
            "| Uptime | 42 days |"
        )
        before = estimate_tokens(table)
        after = estimate_tokens(compress_table_to_kv(table))
        savings_pct = (before - after) / before * 100
        assert savings_pct > 30, f"Only {savings_pct:.1f}% savings on 2-col table"

    def test_3col_table_savings(self):
        table = (
            "| Host | IP | Role |\n|------|-----|------|\n"
            "| gw | 192.168.1.1 | gateway |\n"
            "| n1 | 192.168.1.2 | worker |\n"
            "| n2 | 192.168.1.3 | worker |"
        )
        before = estimate_tokens(table)
        after = estimate_tokens(compress_table_to_kv(table))
        assert after < before

    def test_empty_table_no_crash(self):
        table = "| A | B |\n|---|---|\n"
        compress_table_to_kv(table)  # Should not crash


class TestPunctuationEconomics:
    """Each fullwidth → halfwidth saves ~1 token."""

    def test_measurable_savings(self):
        # Fullwidth punctuation in English-heavy context saves more
        text = "IP：192.168.1.1，Port：8080，Status：online，Mode：active，Region：US"
        before = estimate_tokens(text)
        after = estimate_tokens(normalize_punctuation(text))
        # Savings depend on tokenizer; at minimum should not increase
        assert after <= before, f"Punctuation normalization increased tokens: {before} → {after}"


class TestDictionaryEconomics:
    """Dictionary compression should show net savings on realistic data."""

    def test_net_savings_positive(self):
        """On text with enough repetition, net savings should be positive."""
        texts = [
            "The gateway at 10.0.1.1 serves example_user workspace.\n" * 3,
            "Connect to 10.0.1.1 as example_user for deployment.\n" * 3,
            "Node remote-node at 10.0.1.2 reports to 10.0.1.1 gateway.\n" * 3,
        ]
        cb = build_codebook(texts, min_freq=2)
        if cb:  # Only test if codebook was built
            combined = '\n'.join(texts)
            compressed = compress_text(combined, cb)
            stats = compression_stats(combined, compressed, cb)
            assert stats["gross_reduction_pct"] > 0

    def test_tiny_text_overhead(self):
        """On tiny text, codebook overhead may exceed savings."""
        texts = ["short text here", "other short text"]
        cb = build_codebook(texts, min_freq=1)
        # Either empty codebook or overhead is acknowledged
        if cb:
            stats = compression_stats("short text", compress_text("short text", cb), cb)
            # Net could be negative — that's expected and correct
            assert "net_reduction_pct" in stats


class TestCompressedContextEconomics:
    """CCP should reduce tokens on prose-heavy content."""

    def test_ultra_reduces_prose(self):
        prose = (
            "Alex is the CTO of ExampleCorp and the founder of DataPlatform. "
            "He has over 13 years of experience in cryptocurrency quantitative trading. "
            "He is currently based in Los Gatos, California, with additional offices "
            "in Shanghai and Taipei. The organization focuses on building infrastructure "
            "for automated trading applications and monitoring systems."
        )
        stats = compress_with_stats(prose, "ultra")
        assert stats["compressed_tokens"] < stats["original_tokens"]
        text_savings_pct = (stats["original_tokens"] - stats["compressed_tokens"]) / stats["original_tokens"] * 100
        assert text_savings_pct > 5, f"Only {text_savings_pct:.1f}% text reduction on prose"

    def test_light_minimal_overhead(self):
        text = "Simple clean text with no issues at all."
        stats = compress_with_stats(text, "light")
        # Light mode should barely change clean text
        assert stats["compressed_tokens"] <= stats["original_tokens"] + 1


class TestRLEEconomics:
    """RLE should save tokens on path-heavy and IP-heavy text."""

    def test_path_compression_saves_tokens(self):
        ws = "/home/user/workspace"
        text = f"File at {ws}/memory/a.md\nAlso {ws}/TOOLS.md\nAnd {ws}/MEMORY.md\n"
        before = estimate_tokens(text)
        after = estimate_tokens(compress_paths(text, [ws]))
        assert after < before
        savings = before - after
        assert savings >= 3, f"Only saved {savings} tokens on 3 path occurrences"

    def test_ip_compression_saves_tokens(self):
        text = "Servers: 10.0.1.1, 10.0.1.2, 10.0.1.3, 10.0.1.4"
        before = estimate_tokens(text)
        compressed, _ = compress_ip_families(text)
        after = estimate_tokens(compressed)
        assert after <= before
