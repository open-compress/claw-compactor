"""Tests for RLE (Run-Length Encoding) structured data compression."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.rle import (
    compress_paths, decompress_paths,
    compress_ip_families, decompress_ip_families,
    compress_enumerations, compress,
    decompress,
)


WS_PATH = "/home/user/workspace"


class TestCompressPaths:
    def test_basic(self):
        text = f"File at {WS_PATH}/memory/test.md"
        result = compress_paths(text, [WS_PATH])
        assert "$WS/memory/test.md" in result
        assert WS_PATH not in result

    def test_multiple_occurrences(self):
        text = f"{WS_PATH}/a.md and {WS_PATH}/b.md"
        result = compress_paths(text, [WS_PATH])
        assert result.count("$WS") == 2

    def test_no_match(self):
        text = "No paths here"
        assert compress_paths(text, [WS_PATH]) == text

    def test_roundtrip(self):
        text = f"Path: {WS_PATH}/skills/claw-compactor"
        compressed = compress_paths(text, [WS_PATH])
        decompressed = decompress_paths(compressed, WS_PATH)
        assert decompressed == text


class TestCompressIPFamilies:
    def test_basic(self):
        text = "Server at 10.0.1.1 and node at 10.0.1.2"
        result, prefix_map = compress_ip_families(text)
        assert len(prefix_map) > 0
        # Should be shorter
        assert len(result) < len(text)

    def test_roundtrip(self):
        text = "IPs: 10.0.1.1, 10.0.1.2, 10.0.1.3"
        compressed, prefix_map = compress_ip_families(text)
        decompressed = decompress_ip_families(compressed, prefix_map)
        # All original IPs should be present
        assert "10.0.1.1" in decompressed
        assert "10.0.1.2" in decompressed
        assert "10.0.1.3" in decompressed

    def test_different_subnets(self):
        text = "10.0.0.1 and 10.0.0.2 and 192.168.1.1"
        result, prefix_map = compress_ip_families(text)
        # 10.0.0.x family should be compressed
        assert "$IP" in result or len(prefix_map) > 0

    def test_single_ip_no_compress(self):
        text = "Only one IP: 10.0.0.1"
        result, prefix_map = compress_ip_families(text)
        assert result == text
        assert len(prefix_map) == 0


class TestCompressEnumerations:
    def test_ticker_list(self):
        text = "Trading pairs: BTC, ETH, SOL, BNB, DOGE"
        result = compress_enumerations(text)
        assert "[BTC,ETH,SOL,BNB,DOGE]" in result

    def test_short_list_no_change(self):
        text = "BTC, ETH, SOL"
        result = compress_enumerations(text)
        assert result == text  # Only 3 items, no change

    def test_mixed_case_no_change(self):
        text = "apple, banana, cherry, date"
        result = compress_enumerations(text)
        assert result == text  # Not all-caps codes


class TestCompress:
    def test_combined(self):
        text = (
            f"Connected to {WS_PATH}/config\n"
            "IPs: 10.0.1.1, 10.0.1.2\n"
            "Tokens: BTC, ETH, SOL, BNB, DOGE\n"
        )
        result = compress(text, [WS_PATH])
        assert "$WS" in result
        assert len(result) < len(text)

    def test_empty(self):
        assert compress("") == ""

    def test_unicode(self):
        text = f"中文路径 {WS_PATH}/中文文件.md"
        result = compress(text, [WS_PATH])
        assert "中文" in result
        assert "$WS" in result

    def test_decompress_paths(self):
        text = f"Path: {WS_PATH}/test"
        compressed = compress(text, [WS_PATH])
        decompressed = decompress(compressed, WS_PATH)
        assert WS_PATH in decompressed
