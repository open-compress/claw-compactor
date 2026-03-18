"""Comprehensive roundtrip tests — dictionary, RLE, and combined must be perfectly reversible."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.dictionary import build_codebook, compress_text, decompress_text
from claw_compactor.rle import compress, decompress, compress_paths, decompress_paths, compress_ip_families, decompress_ip_families


class TestDictionaryRoundtrip:
    """Dictionary compression must be perfectly reversible."""

    @pytest.fixture
    def codebook(self):
        texts = [
            "The server at 192.168.1.100 runs Python 3.11. "
            "The server at 192.168.1.100 handles requests. "
            "Python 3.11 is the main runtime. "
            "The workspace at /Users/duke/workspace contains files. "
            "/Users/duke/workspace has memory files.",
        ]
        return build_codebook(texts)

    def test_basic_roundtrip(self, codebook):
        original = "The server at 192.168.1.100 runs Python 3.11."
        compressed = compress_text(original, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == original

    def test_empty_roundtrip(self, codebook):
        assert decompress_text(compress_text("", codebook), codebook) == ""

    def test_no_matches_roundtrip(self, codebook):
        original = "zzzz xxxx yyyy qqqq"
        assert decompress_text(compress_text(original, codebook), codebook) == original

    def test_unicode_roundtrip(self, codebook):
        original = "中文 192.168.1.100 日本語 Python 3.11"
        compressed = compress_text(original, codebook)
        assert decompress_text(compressed, codebook) == original

    def test_multiline_roundtrip(self, codebook):
        original = "Line 1: server at 192.168.1.100\nLine 2: Python 3.11\nLine 3: done"
        compressed = compress_text(original, codebook)
        assert decompress_text(compressed, codebook) == original

    def test_special_chars_roundtrip(self, codebook):
        original = "regex: [a-z]+ and $HOME and \\n and 192.168.1.100"
        compressed = compress_text(original, codebook)
        assert decompress_text(compressed, codebook) == original

    def test_large_text_roundtrip(self, codebook):
        original = ("The server at 192.168.1.100 runs Python 3.11. " * 100).strip()
        compressed = compress_text(original, codebook)
        assert decompress_text(compressed, codebook) == original

    def test_codebook_codes_dont_collide(self, codebook):
        """Codes in output shouldn't accidentally match other entries."""
        original = "simple text with $AA literal dollar signs"
        compressed = compress_text(original, codebook)
        decompressed = decompress_text(compressed, codebook)
        assert decompressed == original

    def test_all_entries_roundtrip(self, codebook):
        """Each codebook entry individually roundtrips."""
        for phrase, code in codebook.items():
            compressed = compress_text(phrase, codebook)
            decompressed = decompress_text(compressed, codebook)
            assert decompressed == phrase, f"Failed for: {phrase}"

    def test_repeated_compression_stable(self, codebook):
        """Compressing already-compressed text then decompressing should work."""
        original = "The server at 192.168.1.100 runs Python 3.11."
        c1 = compress_text(original, codebook)
        # Decompress should always get back to original
        assert decompress_text(c1, codebook) == original


class TestRLERoundtrip:
    """RLE/path compression must be reversible."""

    def test_path_roundtrip(self):
        ws = "/Users/duke/workspace"
        original = f"File at {ws}/memory/test.md and {ws}/TOOLS.md"
        compressed = compress_paths(original, [ws])
        decompressed = decompress_paths(compressed, ws)
        assert decompressed == original

    def test_path_no_match(self):
        ws = "/Users/duke/workspace"
        original = "No paths here at all"
        compressed = compress_paths(original, [ws])
        decompressed = decompress_paths(compressed, ws)
        assert decompressed == original

    def test_path_empty(self):
        ws = "/Users/duke/workspace"
        assert decompress_paths(compress_paths("", [ws]), ws) == ""

    def test_ip_roundtrip(self):
        original = "Server 192.168.1.100 and 192.168.1.200 and 192.168.1.50"
        compressed, prefix_map = compress_ip_families(original)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert decompressed == original

    def test_ip_no_families(self):
        original = "No IPs here"
        compressed, prefix_map = compress_ip_families(original)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert decompressed == original

    def test_ip_single_occurrence(self):
        original = "One IP: 10.0.0.1"
        compressed, prefix_map = compress_ip_families(original)
        decompressed = decompress_ip_families(compressed, prefix_map)
        assert decompressed == original

    def test_ip_empty(self):
        compressed, prefix_map = compress_ip_families("")
        assert decompress_ip_families(compressed, prefix_map) == ""

    def test_mixed_content_roundtrip(self):
        ws = "/Users/duke/workspace"
        original = (
            f"Server at 192.168.1.100 runs on {ws}/scripts/main.py.\n"
            f"Backup at 192.168.1.200 uses {ws}/memory/backup.md.\n"
            f"Another host 192.168.1.50 with {ws}/TOOLS.md"
        )
        # Path compression
        p_compressed = compress_paths(original, [ws])
        p_back = decompress_paths(p_compressed, ws)
        assert p_back == original

        # IP compression
        i_compressed, ip_map = compress_ip_families(original)
        i_back = decompress_ip_families(i_compressed, ip_map)
        assert i_back == original


class TestCombinedRoundtrip:
    """Combined dictionary + RLE roundtrip."""

    def test_dict_then_rle_roundtrip(self):
        texts = [
            "Python 3.11 is great. Python 3.11 runs fast. Python 3.11 everywhere.",
        ]
        codebook = build_codebook(texts)
        original = "Python 3.11 is used on all servers."

        # Dict compress → RLE compress → RLE decompress → Dict decompress
        dict_compressed = compress_text(original, codebook)
        rle_compressed = compress_paths(dict_compressed, ["/Users/duke"])
        rle_back = decompress_paths(rle_compressed, "/Users/duke")
        dict_back = decompress_text(rle_back, codebook)
        assert dict_back == original

    def test_rle_then_dict_roundtrip(self):
        ws = "/Users/duke/workspace"
        texts = [f"{ws}/memory/test.md appears often. {ws}/memory/test.md again."]
        codebook = build_codebook(texts)
        original = f"Check {ws}/memory/test.md for details"

        rle_compressed = compress_paths(original, [ws])
        dict_compressed = compress_text(rle_compressed, codebook)
        dict_back = decompress_text(dict_compressed, codebook)
        rle_back = decompress_paths(dict_back, ws)
        assert rle_back == original


class TestEdgeCaseRoundtrips:
    def test_only_whitespace(self):
        codebook = build_codebook(["a b c d e f"])
        original = "   \n\n\t  "
        assert decompress_text(compress_text(original, codebook), codebook) == original

    def test_only_newlines(self):
        codebook = build_codebook(["test test test"])
        original = "\n\n\n"
        assert decompress_text(compress_text(original, codebook), codebook) == original

    def test_very_long_text(self):
        base = "word " * 50
        codebook = build_codebook([base] * 3)
        original = base * 10
        assert decompress_text(compress_text(original, codebook), codebook) == original

    def test_binary_like_content(self):
        codebook = build_codebook(["test test"])
        original = "hex: 0xDEADBEEF 0xCAFEBABE"
        assert decompress_text(compress_text(original, codebook), codebook) == original
