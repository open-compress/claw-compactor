"""Comprehensive tests for lib/dictionary.py."""
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.dictionary import (
    _generate_codes, _tokenize_ngrams, _extract_ip_prefixes, _extract_path_prefixes,
    build_codebook, compress_text, decompress_text,
    save_codebook, load_codebook, compression_stats,
)
from claw_compactor.tokens import estimate_tokens


class TestGenerateCodes:
    def test_generates_n(self):
        codes = _generate_codes(10)
        assert len(codes) == 10

    def test_unique(self):
        codes = _generate_codes(100)
        assert len(set(codes)) == 100

    def test_all_start_with_dollar(self):
        codes = _generate_codes(50)
        for c in codes:
            assert c.startswith("$")

    def test_zero(self):
        codes = _generate_codes(0)
        assert isinstance(codes, list)

    def test_one(self):
        codes = _generate_codes(1)
        assert len(codes) == 1

    def test_large(self):
        codes = _generate_codes(500)
        assert len(codes) == 500
        assert len(set(codes)) == 500


class TestTokenizeNgrams:
    def test_basic(self):
        result = _tokenize_ngrams("hello world foo bar")
        assert isinstance(result, dict)  # Counter is a dict subclass

    def test_empty(self):
        result = _tokenize_ngrams("")
        assert len(result) == 0

    def test_single_word(self):
        result = _tokenize_ngrams("hello")
        assert isinstance(result, dict)

    def test_custom_max_n(self):
        result = _tokenize_ngrams("a b c d e f", max_n=2)
        assert isinstance(result, dict)

    def test_repeated_text(self):
        result = _tokenize_ngrams("hello world hello world hello world")
        # "hello world" should have high frequency
        assert any(v >= 3 for v in result.values())


class TestExtractIPPrefixes:
    def test_basic(self):
        texts = ["192.168.1.100 and 192.168.1.200"]
        result = _extract_ip_prefixes(texts)
        assert isinstance(result, dict)

    def test_no_ips(self):
        result = _extract_ip_prefixes(["no ips here"])
        assert isinstance(result, dict)

    def test_empty(self):
        result = _extract_ip_prefixes([])
        assert result == {}

    def test_multiple_subnets(self):
        texts = ["10.0.0.1 10.0.0.2 192.168.1.1 192.168.1.2"]
        result = _extract_ip_prefixes(texts)
        assert isinstance(result, dict)


class TestExtractPathPrefixes:
    def test_basic(self):
        texts = ["/Users/testuser/workspace/a.md /Users/testuser/workspace/b.md"]
        result = _extract_path_prefixes(texts)
        assert isinstance(result, dict)

    def test_no_paths(self):
        result = _extract_path_prefixes(["no paths"])
        assert isinstance(result, dict)

    def test_empty(self):
        result = _extract_path_prefixes([])
        assert result == {}


class TestBuildCodebook:
    def test_basic(self):
        texts = ["hello world " * 10, "hello world " * 5]
        cb = build_codebook(texts)
        assert isinstance(cb, dict)

    def test_empty_texts(self):
        cb = build_codebook([])
        assert isinstance(cb, dict)

    def test_min_freq(self):
        texts = ["abc abc abc def def def"]
        cb = build_codebook(texts, min_freq=2)
        assert isinstance(cb, dict)

    def test_max_entries(self):
        texts = [" ".join(f"word{i}" for i in range(100)) * 5]
        cb = build_codebook(texts, max_entries=5)
        assert len(cb) <= 5

    def test_single_text(self):
        cb = build_codebook(["repetitive text repetitive text repetitive text"])
        assert isinstance(cb, dict)


class TestCompressDecompressText:
    def test_basic_roundtrip(self):
        codebook = {"hello world": "$AA"}
        text = "hello world"
        compressed = compress_text(text, codebook)
        assert decompress_text(compressed, codebook) == text

    def test_multiple_replacements_roundtrip(self):
        codebook = {"foo": "$AA", "bar": "$AB"}
        text = "foo and bar"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text

    def test_no_match(self):
        codebook = {"xyz": "$AA"}
        text = "hello world"
        assert compress_text(text, codebook) == text
        assert decompress_text(text, codebook) == text

    def test_empty_text(self):
        codebook = {"hello": "$AA"}
        assert compress_text("", codebook) == ""
        assert decompress_text("", codebook) == ""

    def test_empty_codebook(self):
        assert compress_text("hello", {}) == "hello"
        assert decompress_text("hello", {}) == "hello"

    def test_overlapping_phrases_roundtrip(self):
        codebook = {"hello": "$AA", "hello world": "$AB"}
        text = "hello world"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text

    def test_adjacent_codes_roundtrip(self):
        codebook = {"aa": "$AA", "bb": "$AB"}
        text = "aabb"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text


class TestSaveLoadCodebook:
    def test_roundtrip(self, tmp_path):
        cb = {"hello": "$AA", "world": "$AB", "中文": "$AC"}
        path = tmp_path / "cb.json"
        save_codebook(cb, path)
        loaded = load_codebook(path)
        assert loaded == cb

    def test_empty_codebook(self, tmp_path):
        path = tmp_path / "cb.json"
        save_codebook({}, path)
        loaded = load_codebook(path)
        assert loaded == {}

    def test_overwrite(self, tmp_path):
        path = tmp_path / "cb.json"
        save_codebook({"a": "$AA"}, path)
        save_codebook({"b": "$BB"}, path)
        loaded = load_codebook(path)
        assert loaded == {"b": "$BB"}


class TestCompressionStats:
    def test_basic(self):
        codebook = {"hello world": "$AA"}
        texts = {"file.md": "hello world hello world"}
        stats = compression_stats(texts, codebook)
        assert isinstance(stats, dict)

    def test_empty(self):
        stats = compression_stats({}, {})
        assert isinstance(stats, dict)

    def test_no_compression(self):
        codebook = {"xyz": "$AA"}
        texts = {"f.md": "abc def ghi"}
        stats = compression_stats(texts, codebook)
        assert isinstance(stats, dict)


class TestEndToEnd:
    """End-to-end dictionary compression flow."""

    def test_build_compress_decompress(self):
        texts = [
            "The server at 192.168.1.100 runs Python. " * 5,
            "Python is used for compression. " * 3,
        ]
        codebook = build_codebook(texts, min_freq=2)
        original = "The server at 192.168.1.100 runs Python."
        compressed = compress_text(original, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == original

    def test_compression_saves_tokens(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        codebook = build_codebook([text], min_freq=2)
        compressed = compress_text(text, codebook)
        orig_tokens = estimate_tokens(text)
        comp_tokens = estimate_tokens(compressed)
        assert comp_tokens <= orig_tokens

    def test_file_persistence(self, tmp_path):
        texts = ["repeat phrase " * 20]
        codebook = build_codebook(texts, min_freq=2)
        path = tmp_path / "cb.json"
        save_codebook(codebook, path)
        loaded = load_codebook(path)

        original = "repeat phrase test"
        assert decompress_text(compress_text(original, loaded), loaded) == original
