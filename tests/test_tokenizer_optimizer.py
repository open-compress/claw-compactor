"""Tests for tokenizer-aware format optimization."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.tokenizer_optimizer import (
    normalize_punctuation, strip_bold_italic, strip_trivial_backticks,
    minimize_whitespace, compact_bullets, compress_table_to_kv,
    optimize_tokens, estimate_savings,
)
from claw_compactor.tokens import estimate_tokens


class TestNormalizePunctuation:
    def test_chinese_punctuation(self):
        text = "你好，世界。这是：测试！"
        result = normalize_punctuation(text)
        assert "，" not in result
        assert "。" not in result
        assert "：" not in result
        assert "！" not in result
        assert "," in result
        assert "." in result

    def test_brackets(self):
        result = normalize_punctuation("（测试）")
        assert "(" in result
        assert ")" in result

    def test_no_change_ascii(self):
        text = "hello, world!"
        assert normalize_punctuation(text) == text

    def test_mixed(self):
        text = "IP：192.168.1.1，端口：8080"
        result = normalize_punctuation(text)
        assert result == "IP:192.168.1.1,端口:8080"

    def test_token_savings(self):
        text = "这是一个测试，包含很多中文标点。还有冒号：分号；感叹号！"
        before = estimate_tokens(text)
        after = estimate_tokens(normalize_punctuation(text))
        assert after <= before


class TestStripBoldItalic:
    def test_bold(self):
        assert strip_bold_italic("**hello**") == "hello"

    def test_italic(self):
        assert strip_bold_italic("*hello*") == "hello"

    def test_mixed(self):
        text = "This is **bold** and *italic* text"
        result = strip_bold_italic(text)
        assert result == "This is bold and italic text"


class TestStripTrivialBackticks:
    def test_simple_word(self):
        assert strip_trivial_backticks("`hello`") == "hello"

    def test_keeps_complex(self):
        # Backticks with spaces should be kept (actual code)
        text = "`hello world`"
        assert strip_trivial_backticks(text) == text


class TestMinimizeWhitespace:
    def test_multiple_spaces(self):
        text = "hello    world"
        result = minimize_whitespace(text)
        assert "    " not in result

    def test_deep_indentation(self):
        text = "        deep indent"
        result = minimize_whitespace(text)
        assert len(result) - len(result.lstrip()) <= 4


class TestCompressTableToKv:
    def test_simple_table(self):
        text = "| Key | Value |\n|-----|-------|\n| Name | Duke |\n| Role | CEO |"
        result = compress_table_to_kv(text)
        assert "Name: Duke" in result
        assert "Role: CEO" in result

    def test_multi_column(self):
        text = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
        result = compress_table_to_kv(text)
        # Should convert to compact format
        assert len(result) < len(text)

    def test_no_table(self):
        text = "Just normal text without tables"
        assert compress_table_to_kv(text) == text

    def test_token_savings(self):
        table = (
            "| Server | IP | Status |\n"
            "|--------|-----|--------|\n"
            "| Gateway | 192.168.1.1 | Online |\n"
            "| Node1 | 192.168.1.2 | Online |\n"
            "| Node2 | 192.168.1.3 | Offline |"
        )
        before = estimate_tokens(table)
        after = estimate_tokens(compress_table_to_kv(table))
        assert after < before


class TestCompactBullets:
    def test_long_bullet_list(self):
        text = "- item1\n- item2\n- item3\n- item4"
        result = compact_bullets(text)
        # Should remove bullet prefixes for 3+ consecutive
        assert "- " not in result or result.count("- ") < text.count("- ")

    def test_short_list_kept(self):
        text = "- item1\n- item2"
        result = compact_bullets(text)
        assert "- " in result  # Only 2 items, keep bullets


class TestOptimizeTokens:
    def test_basic(self):
        text = (
            "**Server Config**\n\n"
            "| Key | Value |\n|-----|-------|\n| IP | 192.168.1.1 |\n"
            "状态：正常，端口：8080\n"
        )
        result = optimize_tokens(text)
        assert estimate_tokens(result) <= estimate_tokens(text)

    def test_aggressive(self):
        text = "**Bold** and *italic* with `backticks`\n- item1\n- item2\n- item3\n- item4"
        normal = optimize_tokens(text, aggressive=False)
        aggressive = optimize_tokens(text, aggressive=True)
        assert len(aggressive) <= len(normal)

    def test_empty(self):
        assert optimize_tokens("") == ""

    def test_unicode_preserved(self):
        text = "中文内容 English content 日本語"
        result = optimize_tokens(text)
        assert "中文" in result
        assert "English" in result


class TestEstimateSavings:
    def test_basic(self):
        original = "**Bold** text，with中文标点。\n| K | V |\n|--|--|\n| a | b |"
        optimized = optimize_tokens(original, aggressive=True)
        stats = estimate_savings(original, optimized)
        assert stats["original_tokens"] > 0
        assert stats["optimized_tokens"] > 0
        assert stats["token_reduction_pct"] >= 0

    def test_real_workspace_text(self):
        text = """## SSH 密钥 (在 ~/.ssh/)
| 密钥 | 可访问 |
|------|--------|
| `id_ed25519_mykey` | remote server, local Linux, worker-1, worker-2 |
| `server_key.pem` | remote server, local Linux, worker-1, worker-2 |
| `admin_key.pem` | worker-1.lan, worker-2.lan (user: admin) |

**任意密钥**可登任意节点。
"""
        optimized = optimize_tokens(text, aggressive=True)
        stats = estimate_savings(text, optimized)
        assert stats["token_reduction_pct"] > 0
        assert "密钥" in optimized
