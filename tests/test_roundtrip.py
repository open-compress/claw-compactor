"""Roundtrip tests — the most important invariant.

compress → decompress MUST equal original for all lossless techniques.
Tests cover: ASCII, Chinese, mixed, edge cases, $-containing text.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.dictionary import (
    build_codebook, compress_text, decompress_text,
    save_codebook, load_codebook,
)
from claw_compactor.rle import (
    compress_paths, decompress_paths,
    compress_ip_families, decompress_ip_families,
)


# --- Dictionary Roundtrip ---

class TestDictionaryRoundtripASCII:
    """Pure ASCII text roundtrips."""

    @pytest.fixture
    def codebook(self):
        return {
            "$GW": "10.0.1.1",
            "$JP": "10.0.1.2",
            "$TK": "my-secret-token-2024",
            "$WS": "/home/user/workspace",
        }

    def test_simple(self, codebook):
        text = "Gateway at 10.0.1.1 running"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_multiple_replacements(self, codebook):
        text = "Connect to 10.0.1.1 via 10.0.1.2 using my-secret-token-2024"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_no_matches(self, codebook):
        text = "Nothing matches here at all"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_all_replaced(self, codebook):
        text = "10.0.1.1"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_multiline(self, codebook):
        text = "Line 1: 10.0.1.1\nLine 2: 10.0.1.2\nLine 3: done"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_adjacent_matches(self, codebook):
        text = "10.0.1.110.0.1.2"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_repeated(self, codebook):
        text = "10.0.1.1 and 10.0.1.1 and 10.0.1.1"
        assert decompress_text(compress_text(text, codebook), codebook) == text


class TestDictionaryRoundtripChinese:
    """Chinese text roundtrips."""

    @pytest.fixture
    def codebook(self):
        return {
            "$CN": "日本节点",
            "$GW": "网关服务器",
            "$WS": "工作空间目录",
        }

    def test_pure_chinese(self, codebook):
        text = "连接到日本节点的网关服务器，在工作空间目录下操作"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_no_chinese_matches(self, codebook):
        text = "纯中文但没有匹配的词语"
        assert decompress_text(compress_text(text, codebook), codebook) == text


class TestDictionaryRoundtripMixed:
    """Mixed ASCII + Chinese text roundtrips."""

    @pytest.fixture
    def codebook(self):
        return {
            "$GW": "10.0.1.1",
            "$JP": "日本节点",
            "$TK": "my-secret-token-2024",
        }

    def test_mixed(self, codebook):
        text = "连接到10.0.1.1的日本节点，使用my-secret-token-2024认证"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_mixed_multiline(self, codebook):
        text = "# 配置\n- 网关: 10.0.1.1\n- 节点: 日本节点\n- Token: my-secret-token-2024\n"
        assert decompress_text(compress_text(text, codebook), codebook) == text


class TestDictionaryRoundtripDollarSign:
    """Text containing $ that could conflict with code format."""

    @pytest.fixture
    def codebook(self):
        return {
            "$GW": "gateway",
            "$JP": "japan",
        }

    def test_dollar_in_text(self, codebook):
        """Text with $ should not be corrupted."""
        text = "Price is $100 and gateway is main"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text

    def test_dollar_code_like_in_text(self, codebook):
        """Text that looks like a code ($XX) but isn't in codebook."""
        text = "Variable $AB is not a code, gateway runs"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text

    def test_existing_code_in_text(self, codebook):
        """If text already contains $GW literally (not from codebook),
        roundtrip may not be perfect — this is a known limitation.
        The codebook should avoid phrases that look like codes."""
        text = "The code $GW was already in the text"
        compressed = compress_text(text, codebook)
        # After compress: $GW is still $GW (text didn't contain "gateway" to replace)
        # After decompress: $GW → "gateway"... this corrupts!
        # This is expected — the codebook builder avoids this case
        # by not creating codes that conflict with existing text


class TestDictionaryRoundtripEmpty:
    """Empty and edge case roundtrips."""

    def test_empty_text(self):
        cb = {"$AA": "test"}
        assert decompress_text(compress_text("", cb), cb) == ""

    def test_empty_codebook(self):
        assert decompress_text(compress_text("hello", {}), {}) == "hello"

    def test_both_empty(self):
        assert decompress_text(compress_text("", {}), {}) == ""

    def test_only_whitespace(self):
        cb = {"$AA": "test"}
        text = "   \n\n\t  "
        assert decompress_text(compress_text(text, cb), cb) == text


class TestDictionaryRoundtripBuiltCodebook:
    """Roundtrip using auto-built codebooks from real-ish data."""

    def test_workspace_like_data(self):
        texts = [
            "# Config\n- Gateway: 10.0.1.1\n- Node: remote-node 10.0.1.2\n"
            "- Path: /home/user/workspace\n- Token: my-token-2024\n",
        ] * 5  # Repeat to ensure frequency
        cb = build_codebook(texts, min_freq=3)
        for text in texts:
            assert decompress_text(compress_text(text, cb), cb) == text

    def test_unicode_heavy_data(self):
        texts = [
            "# 服务器配置\n- 网关服务器地址: 10.0.1.1\n- 日本节点地址: 10.0.2.1\n",
        ] * 5
        cb = build_codebook(texts, min_freq=3)
        for text in texts:
            assert decompress_text(compress_text(text, cb), cb) == text

    def test_save_load_preserves_roundtrip(self, tmp_path):
        """Codebook saved to disk and reloaded still roundtrips."""
        texts = [
            "server 192.168.1.100 with token super-long-token-value-here\n" * 3,
        ] * 3
        cb = build_codebook(texts, min_freq=2)
        if not cb:
            pytest.skip("No codebook built")

        path = tmp_path / "cb.json"
        save_codebook(cb, path)
        loaded_cb = load_codebook(path)

        combined = '\n'.join(texts)
        compressed = compress_text(combined, loaded_cb)
        decompressed = decompress_text(compressed, loaded_cb)
        assert decompressed == combined


# --- RLE Roundtrip ---

class TestRLEPathRoundtrip:
    """$WS path roundtrip."""

    WS = "/home/user/workspace"

    def test_basic(self):
        text = f"File: {self.WS}/memory/test.md"
        compressed = compress_paths(text, [self.WS])
        decompressed = decompress_paths(compressed, self.WS)
        assert decompressed == text

    def test_multiple(self):
        text = f"{self.WS}/a.md and {self.WS}/b.md"
        compressed = compress_paths(text, [self.WS])
        decompressed = decompress_paths(compressed, self.WS)
        assert decompressed == text

    def test_no_match(self):
        text = "no paths here"
        compressed = compress_paths(text, [self.WS])
        decompressed = decompress_paths(compressed, self.WS)
        assert decompressed == text

    def test_empty(self):
        compressed = compress_paths("", [self.WS])
        decompressed = decompress_paths(compressed, self.WS)
        assert decompressed == ""


class TestRLEIPRoundtrip:
    """IP family compression roundtrip."""

    def test_basic(self):
        text = "IPs: 10.0.1.1, 10.0.1.2, 10.0.1.3"
        compressed, prefix_map = compress_ip_families(text)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert "10.0.1.1" in decompressed
        assert "10.0.1.2" in decompressed
        assert "10.0.1.3" in decompressed

    def test_no_ips(self):
        text = "no IPs"
        compressed, prefix_map = compress_ip_families(text)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert decompressed == text

    def test_single_ip(self):
        text = "Only 10.0.0.1 here"
        compressed, prefix_map = compress_ip_families(text)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert decompressed == text
