"""Tests for dictionary-based compression."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.dictionary import (
    build_codebook, compress_text, decompress_text,
    save_codebook, load_codebook, compression_stats,
    _generate_codes, _tokenize_ngrams, _extract_ip_prefixes,
    _extract_path_prefixes,
)


# --- Fixtures ---

@pytest.fixture
def sample_texts():
    """Realistic memory-style texts with repetition."""
    return [
        "# Server Config\n\n- Gateway IP: 10.0.1.1\n- Node: remote-node at 10.0.1.2\n"
        "- User: example_user\n- Path: /home/user/workspace\n"
        "- NetworkID: abc123network456\n- Token: my-secret-token-2024\n",

        "# Daily Notes\n\n- Connected to 10.0.1.1 gateway\n"
        "- Deployed on remote-node (10.0.1.2)\n"
        "- User example_user ran backup\n"
        "- Workspace: /home/user/workspace\n"
        "- Using token my-secret-token-2024\n",

        "# More Notes\n\n- Gateway 10.0.1.1 is stable\n"
        "- remote-node running fine at 10.0.1.2\n"
        "- example_user home: /home/user/workspace\n"
        "- Auth: my-secret-token-2024\n",
    ]


@pytest.fixture
def codebook():
    return {
        "$GW": "10.0.1.1",
        "$JP": "10.0.1.2",
        "$U": "example_user",
        "$WS": "/home/user/workspace",
        "$TK": "my-secret-token-2024",
    }


# --- Code generation ---

class TestGenerateCodes:
    def test_count(self):
        codes = _generate_codes(10)
        assert len(codes) == 10

    def test_uniqueness(self):
        codes = _generate_codes(676)
        assert len(set(codes)) == 676

    def test_format(self):
        codes = _generate_codes(5)
        for c in codes:
            assert c.startswith('$')
            assert len(c) == 3
            assert c[1:].isalpha() and c[1:].isupper()

    def test_overflow_to_3char(self):
        codes = _generate_codes(677)
        assert len(codes) == 677
        assert len(codes[-1]) == 4  # $AAA


# --- N-gram extraction ---

class TestTokenizeNgrams:
    def test_basic(self):
        ngrams = _tokenize_ngrams("the quick brown fox jumps over the lazy dog")
        assert any("quick brown" in k for k in ngrams)

    def test_min_length_filter(self):
        ngrams = _tokenize_ngrams("a b c d e f")
        # All ngrams shorter than MIN_PHRASE_LEN should be filtered
        for gram in ngrams:
            assert len(gram) >= 6

    def test_empty(self):
        assert len(_tokenize_ngrams("")) == 0


# --- IP prefix extraction ---

class TestExtractIpPrefixes:
    def test_finds_common_prefix(self):
        texts = [
            "Server at 192.168.1.10 and 192.168.1.20",
            "Also 192.168.1.30 and 10.0.0.1",
            "More at 192.168.1.40",
        ]
        prefixes = _extract_ip_prefixes(texts)
        assert any("192.168.1." in p for p in prefixes)

    def test_no_ips(self):
        assert _extract_ip_prefixes(["no ips here"]) == {}


# --- Path prefix extraction ---

class TestExtractPathPrefixes:
    def test_finds_common_path(self):
        texts = [
            "File at /Users/duke/.openclaw/workspace/MEMORY.md",
            "Also /Users/duke/.openclaw/workspace/TOOLS.md",
            "And /Users/duke/.openclaw/workspace/memory/2026-01.md",
        ]
        prefixes = _extract_path_prefixes(texts)
        assert any("/Users/duke/.openclaw" in p for p in prefixes)

    def test_no_paths(self):
        assert _extract_path_prefixes(["no paths"]) == {}


# --- Codebook building ---

class TestBuildCodebook:
    def test_builds_from_texts(self, sample_texts):
        cb = build_codebook(sample_texts, min_freq=2)
        assert len(cb) > 0
        # All values should be real phrases
        for code, phrase in cb.items():
            assert code.startswith('$')
            assert len(phrase) >= 6

    def test_empty_texts(self):
        cb = build_codebook([], min_freq=2)
        assert cb == {}

    def test_no_frequent_phrases(self):
        cb = build_codebook(["unique text one", "different text two"], min_freq=5)
        assert cb == {}

    def test_max_entries(self, sample_texts):
        cb = build_codebook(sample_texts, max_entries=3, min_freq=2)
        assert len(cb) <= 3

    def test_no_overlapping_codes(self, sample_texts):
        cb = build_codebook(sample_texts, min_freq=2)
        codes = list(cb.keys())
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                assert c1 != c2


# --- Compression / Decompression ---

class TestCompressDecompress:
    def test_basic_compress(self, codebook):
        text = "Gateway at 10.0.1.1 running"
        r = compress_text(text, codebook)
        assert "$GW" in r
        assert "10.0.1.1" not in r

    def test_basic_decompress(self, codebook):
        compressed = "Gateway at $GW running"
        r = decompress_text(compressed, codebook)
        assert "10.0.1.1" in r
        assert "$GW" not in r

    def test_roundtrip(self, codebook):
        """Compression then decompression must be lossless."""
        text = (
            "Connect to 10.0.1.1 as example_user\n"
            "Node 10.0.1.2 is remote-node\n"
            "Workspace: /home/user/workspace\n"
            "Token: my-secret-token-2024\n"
        )
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text

    def test_roundtrip_no_matches(self, codebook):
        text = "Nothing to compress here."
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_roundtrip_all_matches(self, codebook):
        text = "10.0.1.1 10.0.1.2 example_user"
        assert decompress_text(compress_text(text, codebook), codebook) == text

    def test_empty_codebook(self):
        assert compress_text("hello", {}) == "hello"
        assert decompress_text("hello", {}) == "hello"

    def test_empty_text(self, codebook):
        assert compress_text("", codebook) == ""
        assert decompress_text("", codebook) == ""

    def test_multiple_occurrences(self, codebook):
        text = "IP 10.0.1.1 and again 10.0.1.1"
        compressed = compress_text(text, codebook)
        assert compressed.count("$GW") == 2
        assert decompress_text(compressed, codebook) == text

    def test_adjacent_codes(self, codebook):
        text = "10.0.1.1example_user"
        compressed = compress_text(text, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == text


class TestRoundtripWithBuiltCodebook:
    """Test that build → compress → decompress is lossless."""

    def test_full_pipeline(self, sample_texts):
        cb = build_codebook(sample_texts, min_freq=2)
        for text in sample_texts:
            compressed = compress_text(text, cb)
            decompressed = decompress_text(compressed, cb)
            assert decompressed == text, f"Roundtrip failed for text starting with: {text[:50]}"

    def test_large_corpus(self):
        """Test with a larger synthetic corpus."""
        base = "Server 192.168.1.{i} is running service-{i} on /opt/app/service-{i}/config"
        texts = [base.format(i=i) for i in range(50)]
        cb = build_codebook(texts, min_freq=3)
        combined = '\n'.join(texts)
        compressed = compress_text(combined, cb)
        decompressed = decompress_text(compressed, cb)
        assert decompressed == combined


# --- Save / Load ---

class TestSaveLoadCodebook:
    def test_save_load_roundtrip(self, tmp_path, codebook):
        path = tmp_path / "cb.json"
        save_codebook(codebook, path)
        loaded = load_codebook(path)
        assert loaded == codebook

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_codebook(tmp_path / "nope.json")

    def test_load_invalid(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not_entries": true}')
        with pytest.raises(ValueError):
            load_codebook(path)

    def test_save_creates_dirs(self, tmp_path, codebook):
        path = tmp_path / "deep" / "nested" / "cb.json"
        save_codebook(codebook, path)
        assert path.exists()


# --- Stats ---

class TestCompressionStats:
    def test_basic(self, codebook):
        original = "10.0.1.1 is the gateway for example_user"
        compressed = compress_text(original, codebook)
        stats = compression_stats(original, compressed, codebook)
        assert stats["gross_reduction_pct"] > 0
        assert stats["codebook_entries"] == len(codebook)
        assert stats["codes_used"] > 0

    def test_empty(self):
        stats = compression_stats("", "", {})
        assert stats["gross_reduction_pct"] == 0.0


# --- Integration with dictionary_compress.py CLI ---

class TestCLICommands:
    def test_build_and_stats(self, tmp_path):
        from dictionary_compress import cmd_build, cmd_stats
        # Create workspace
        (tmp_path / "MEMORY.md").write_text(
            "# Mem\n\n- Server: 192.168.1.100\n- Server: 192.168.1.100\n"
            "- Path: /long/repeated/path/here\n- Path: /long/repeated/path/here\n"
            "- Token: my-long-token-value-2024\n- Token: my-long-token-value-2024\n"
            "- Again server 192.168.1.100 and /long/repeated/path/here\n"
        )
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "day1.md").write_text(
            "# Day\n- Used 192.168.1.100\n- At /long/repeated/path/here\n"
            "- Auth: my-long-token-value-2024\n"
        )

        cb_path = tmp_path / "memory" / ".codebook.json"
        result = cmd_build(tmp_path, cb_path, min_freq=2)
        assert result["codebook_entries"] > 0
        assert cb_path.exists()

        stats = cmd_stats(tmp_path, cb_path)
        assert stats["codebook_entries"] > 0

    def test_compress_decompress_roundtrip(self, tmp_path):
        from dictionary_compress import cmd_build, cmd_compress, cmd_decompress

        original_text = (
            "# Notes\n- IP: 10.20.30.40 is primary\n- IP: 10.20.30.40 backup too\n"
            "- Workspace /home/user/.openclaw/workspace is main\n"
            "- Also /home/user/.openclaw/workspace for backup\n"
            "- Token super-secret-token-2024 used\n"
            "- Again super-secret-token-2024 here\n"
            "- And 10.20.30.40 again\n"
        )
        (tmp_path / "MEMORY.md").write_text(original_text)
        mem = tmp_path / "memory"
        mem.mkdir()

        cb_path = mem / ".codebook.json"
        cmd_build(tmp_path, cb_path, min_freq=2)

        # Compress
        cmd_compress(tmp_path, cb_path, dry_run=False)
        compressed = (tmp_path / "MEMORY.md").read_text()
        assert len(compressed) <= len(original_text)

        # Decompress
        cmd_decompress(tmp_path, cb_path, dry_run=False)
        restored = (tmp_path / "MEMORY.md").read_text()
        assert restored == original_text
