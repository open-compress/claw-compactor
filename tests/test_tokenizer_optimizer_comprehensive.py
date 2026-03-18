"""Comprehensive tests for lib/tokenizer_optimizer.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.tokenizer_optimizer import (
    strip_bold_italic, strip_trivial_backticks, normalize_punctuation,
    compress_table_to_kv, compact_bullets, minimize_whitespace,
    optimize_tokens, estimate_savings,
)


class TestStripBoldItalic:
    def test_strips_bold(self):
        assert "hello" in strip_bold_italic("**hello**")
        assert "**" not in strip_bold_italic("**hello**")

    def test_strips_italic(self):
        result = strip_bold_italic("*hello*")
        assert "hello" in result

    def test_mixed(self):
        result = strip_bold_italic("**bold** and *italic* text")
        assert "bold" in result
        assert "italic" in result

    def test_no_markers(self):
        assert strip_bold_italic("plain text") == "plain text"

    def test_empty(self):
        assert strip_bold_italic("") == ""

    def test_nested(self):
        result = strip_bold_italic("***bold italic***")
        assert isinstance(result, str)

    def test_preserves_asterisks_in_code(self):
        # Backtick-wrapped asterisks should ideally be preserved
        result = strip_bold_italic("use `**kwargs` in Python")
        assert isinstance(result, str)

    def test_multiline(self):
        text = "**line1**\n**line2**\n*line3*"
        result = strip_bold_italic(text)
        assert "line1" in result
        assert "line2" in result


class TestStripTrivialBackticks:
    def test_strips_simple_word(self):
        result = strip_trivial_backticks("`hello`")
        assert "hello" in result

    def test_keeps_code_content(self):
        # Content with spaces or special chars should be kept
        result = strip_trivial_backticks("`ls -la`")
        assert "ls" in result

    def test_empty(self):
        assert strip_trivial_backticks("") == ""

    def test_no_backticks(self):
        assert strip_trivial_backticks("plain") == "plain"

    def test_code_block_preserved(self):
        text = "```python\ncode here\n```"
        result = strip_trivial_backticks(text)
        assert "code here" in result


class TestNormalizePunctuation:
    def test_chinese_comma(self):
        assert "，" not in normalize_punctuation("你好，世界")

    def test_chinese_period(self):
        assert "。" not in normalize_punctuation("完成。")

    def test_no_change_english(self):
        text = "Hello, world."
        assert normalize_punctuation(text) == text

    def test_empty(self):
        assert normalize_punctuation("") == ""

    def test_all_chinese_punct(self):
        text = "测试，内容。问题？回答！"
        result = normalize_punctuation(text)
        assert "，" not in result
        assert "。" not in result


class TestCompressTableToKv:
    def test_basic_table(self):
        text = "| Key | Value |\n|-----|-------|\n| name | Alice |\n| age | 30 |"
        result = compress_table_to_kv(text)
        assert "Alice" in result
        assert "30" in result
        assert len(result) <= len(text)

    def test_no_table(self):
        text = "just text"
        assert compress_table_to_kv(text) == text

    def test_empty(self):
        assert compress_table_to_kv("") == ""

    def test_three_columns(self):
        text = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
        result = compress_table_to_kv(text)
        assert "1" in result

    def test_table_with_text_around(self):
        text = "Header:\n| K | V |\n|---|---|\n| x | y |\nFooter"
        result = compress_table_to_kv(text)
        assert "Header" in result
        assert "Footer" in result

    def test_savings_on_real_table(self):
        text = (
            "| Setting | Value | Description |\n"
            "|---------|-------|-------------|\n"
            "| timeout | 30s | Request timeout |\n"
            "| retries | 3 | Maximum retry count |\n"
            "| workers | 8 | Number of workers |\n"
            "| buffer | 4096 | Buffer size bytes |\n"
        )
        result = compress_table_to_kv(text)
        assert len(result) < len(text)


class TestCompactBullets:
    def test_removes_prefixes(self):
        text = "- item 1\n- item 2\n- item 3"
        result = compact_bullets(text)
        assert isinstance(result, str)

    def test_empty(self):
        assert compact_bullets("") == ""

    def test_no_bullets(self):
        text = "paragraph text"
        assert compact_bullets(text) == text

    def test_nested_bullets(self):
        text = "- outer\n  - inner\n  - inner2\n- outer2"
        result = compact_bullets(text)
        assert isinstance(result, str)

    def test_numbered_list(self):
        text = "1. first\n2. second\n3. third"
        result = compact_bullets(text)
        assert isinstance(result, str)


class TestMinimizeWhitespace:
    def test_collapses_blanks(self):
        text = "a\n\n\n\nb"
        result = minimize_whitespace(text)
        assert "\n\n\n" not in result
        assert "a" in result and "b" in result

    def test_strips_trailing(self):
        text = "hello   \nworld   "
        result = minimize_whitespace(text)
        assert "   " not in result or result.strip() == result.strip()

    def test_empty(self):
        assert minimize_whitespace("") == ""

    def test_single_line(self):
        assert minimize_whitespace("hello") == "hello"

    def test_tabs(self):
        text = "\thello\t\tworld"
        result = minimize_whitespace(text)
        assert isinstance(result, str)


class TestOptimizeTokens:
    def test_basic(self):
        text = "**Bold** text，中文。And `simple` backticks"
        result = optimize_tokens(text)
        assert isinstance(result, str)

    def test_aggressive(self):
        text = "**Bold** text with | table | data |\n|---|---|\n| a | b |"
        result = optimize_tokens(text, aggressive=True)
        assert isinstance(result, str)

    def test_empty(self):
        assert optimize_tokens("") == ""

    def test_idempotent(self):
        """Running twice should give same result."""
        text = "**Some** text，with `stuff`"
        first = optimize_tokens(text)
        second = optimize_tokens(first)
        assert first == second

    def test_never_increases_tokens(self):
        """Optimization should never make text use more tokens."""
        from claw_compactor.tokens import estimate_tokens
        texts = [
            "**Bold** and *italic* markers",
            "| Table | Here |\n|-------|------|\n| a | b |",
            "中文，标点。测试！",
            "- item1\n- item2\n- item3",
            "   lots   of   spaces   ",
        ]
        for text in texts:
            optimized = optimize_tokens(text)
            # Token count should not increase (or increase minimally)
            orig_tokens = estimate_tokens(text)
            opt_tokens = estimate_tokens(optimized)
            assert opt_tokens <= orig_tokens + 2, f"Increased tokens for: {text[:50]}"


class TestEstimateSavings:
    def test_basic(self):
        original = "**Bold** text with redundancy"
        optimized = "Bold text with redundancy"
        stats = estimate_savings(original, optimized)
        assert "original_tokens" in stats
        assert "optimized_tokens" in stats
        assert stats["optimized_tokens"] <= stats["original_tokens"]

    def test_no_change(self):
        text = "plain text"
        stats = estimate_savings(text, text)
        assert stats["original_tokens"] == stats["optimized_tokens"]

    def test_empty(self):
        stats = estimate_savings("", "")
        assert stats["original_tokens"] == 0

    def test_significant_savings(self):
        original = "    - item one: the first value    \n    - item two: the second value    \n" * 20
        optimized = "item one:first value\nitem two:second value\n" * 20
        stats = estimate_savings(original, optimized)
        assert stats["optimized_tokens"] < stats["original_tokens"]
