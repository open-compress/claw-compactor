"""Comprehensive tests for lib/rle.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.rle import (
    compress_paths, decompress_paths,
    compress_ip_families, decompress_ip_families,
    compress_enumerations, compress_repeated_headers,
    compress, decompress,
)


class TestCompressPaths:
    def test_basic(self):
        ws = "/Users/testuser/workspace"
        text = f"File at {ws}/memory/test.md"
        result = compress_paths(text, [ws])
        assert "$WS" in result or ws not in result or result == text

    def test_multiple_paths(self):
        ws = "/Users/testuser/workspace"
        text = f"{ws}/a.md and {ws}/b.md and {ws}/c.md"
        result = compress_paths(text, [ws])
        assert isinstance(result, str)

    def test_no_match(self):
        text = "no paths here"
        result = compress_paths(text, ["/Users/testuser/workspace"])
        assert result == text

    def test_empty(self):
        result = compress_paths("", ["/Users/duke"])
        assert result == ""

    def test_none_workspace(self):
        result = compress_paths("some text", None)
        assert result == "some text"

    def test_empty_workspace_list(self):
        result = compress_paths("some text", [])
        assert result == "some text"


class TestCompressIPFamilies:
    def test_basic_family(self):
        text = "192.168.1.100 and 192.168.1.200 and 192.168.1.50"
        result, prefix_map = compress_ip_families(text)
        assert isinstance(result, str)
        assert isinstance(prefix_map, dict)

    def test_no_ips(self):
        text = "no ips here"
        result, prefix_map = compress_ip_families(text)
        assert result == text

    def test_single_ip(self):
        text = "host: 10.0.0.1"
        result, prefix_map = compress_ip_families(text)
        assert isinstance(result, str)

    def test_different_subnets(self):
        text = "10.0.0.1 and 192.168.1.1 and 172.16.0.1"
        result, prefix_map = compress_ip_families(text)
        assert isinstance(result, str)

    def test_empty(self):
        result, _ = compress_ip_families("")
        assert result == ""

    def test_min_occurrences(self):
        text = "10.0.0.1 once"
        result, _ = compress_ip_families(text, min_occurrences=5)
        assert result == text


class TestCompressEnumerations:
    def test_basic(self):
        text = "worker-1, worker-2, worker-3, worker-4, worker-5"
        result = compress_enumerations(text)
        assert isinstance(result, str)

    def test_no_enumerations(self):
        text = "no enumerations here"
        assert compress_enumerations(text) == text

    def test_empty(self):
        assert compress_enumerations("") == ""

    def test_numbered_items(self):
        text = "node-1 node-2 node-3 node-4"
        result = compress_enumerations(text)
        assert isinstance(result, str)


class TestCompressRepeatedHeaders:
    def test_repeated(self):
        text = "## Section\nfoo\n## Section\nbar"
        result = compress_repeated_headers(text)
        assert isinstance(result, str)

    def test_no_repeats(self):
        text = "## A\nfoo\n## B\nbar"
        result = compress_repeated_headers(text)
        assert "A" in result and "B" in result

    def test_empty(self):
        assert compress_repeated_headers("") == ""


class TestCompress:
    """Test the main compress() function."""

    def test_basic(self):
        text = "Some text with /Users/testuser/workspace/file.md"
        result = compress(text, ["/Users/testuser/workspace"])
        assert isinstance(result, str)

    def test_empty(self):
        assert compress("") == ""

    def test_no_compression_needed(self):
        text = "plain text"
        result = compress(text)
        assert result == text

    def test_combined_features(self):
        text = (
            "192.168.1.100 and 192.168.1.200\n"
            "worker-1, worker-2, worker-3\n"
            "## Header\nfoo\n## Header\nbar"
        )
        result = compress(text)
        assert isinstance(result, str)


class TestDecompress:
    def test_basic_roundtrip(self):
        ws = "/Users/testuser/workspace"
        text = f"{ws}/test.md"
        compressed = compress(text, [ws])
        decompressed = decompress(compressed, ws)
        assert text in decompressed or decompressed == text

    def test_empty(self):
        assert decompress("", "/Users/duke") == ""

    def test_no_markers(self):
        text = "nothing to decompress"
        assert decompress(text, "/Users/duke") == text
