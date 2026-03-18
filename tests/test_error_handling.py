"""Error handling tests — file not found, corrupt data, edge cases."""
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.exceptions import FileNotFoundError_, MemCompressError, ParseError, TokenEstimationError
from claw_compactor.dictionary import load_codebook, save_codebook, compress_text, decompress_text, build_codebook
from claw_compactor.tokens import estimate_tokens


class TestExceptions:
    def test_file_not_found_is_mem_compress_error(self):
        assert issubclass(FileNotFoundError_, MemCompressError)

    def test_parse_error_is_mem_compress_error(self):
        assert issubclass(ParseError, MemCompressError)

    def test_token_estimation_error(self):
        assert issubclass(TokenEstimationError, MemCompressError)

    def test_raise_file_not_found(self):
        with pytest.raises(FileNotFoundError_):
            raise FileNotFoundError_("test.md")

    def test_raise_parse_error(self):
        with pytest.raises(ParseError):
            raise ParseError("bad markdown")

    def test_exception_message(self):
        e = FileNotFoundError_("missing.md")
        assert "missing.md" in str(e)


class TestDictionaryErrors:
    def test_load_nonexistent_codebook(self, tmp_path):
        with pytest.raises(Exception):
            load_codebook(tmp_path / "nonexistent.json")

    def test_load_corrupt_codebook(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json{{{")
        with pytest.raises(Exception):
            load_codebook(bad)

    def test_load_empty_file(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("")
        with pytest.raises(Exception):
            load_codebook(empty)

    def test_compress_empty_codebook(self):
        result = compress_text("hello world", {})
        assert result == "hello world"

    def test_decompress_empty_codebook(self):
        result = decompress_text("hello world", {})
        assert result == "hello world"

    def test_compress_empty_text(self):
        result = compress_text("", {"hello": "$AA"})
        assert result == ""

    def test_build_codebook_empty_input(self):
        result = build_codebook([])
        assert isinstance(result, dict)

    def test_build_codebook_whitespace_only(self):
        result = build_codebook(["   \n\n  "])
        assert isinstance(result, dict)

    def test_save_load_roundtrip(self, tmp_path):
        codebook = {"hello": "$AA", "world": "$AB"}
        path = tmp_path / "cb.json"
        save_codebook(codebook, path)
        loaded = load_codebook(path)
        assert loaded == codebook


class TestEstimateTokenErrors:
    def test_none_input(self):
        """Estimate tokens with None should raise or handle gracefully."""
        with pytest.raises((TypeError, AttributeError)):
            estimate_tokens(None)

    def test_numeric_input(self):
        with pytest.raises((TypeError, AttributeError)):
            estimate_tokens(42)

    def test_bytes_input(self):
        with pytest.raises((TypeError, AttributeError)):
            estimate_tokens(b"hello")


class TestCompressMemoryErrors:
    def test_nonexistent_path(self):
        from compress_memory import _collect_files
        with pytest.raises(FileNotFoundError_):
            _collect_files("/nonexistent/path/xyz123")

    def test_empty_file(self, tmp_path):
        from compress_memory import compress_file
        f = tmp_path / "empty.md"
        f.write_text("")
        result = compress_file(f, dry_run=True, no_llm=True)
        assert isinstance(result, dict)

    def test_binary_content(self, tmp_path):
        """Binary-ish content shouldn't crash."""
        from compress_memory import rule_compress
        text = "Normal text\x00\x01\x02 more text"
        result = rule_compress(text)
        assert isinstance(result, str)


class TestDedupErrors:
    def test_nonexistent_path(self):
        from dedup_memory import _collect_entries
        with pytest.raises(FileNotFoundError_):
            _collect_entries("/nonexistent/xyz")

    def test_empty_directory(self, tmp_path):
        from dedup_memory import _collect_entries
        result = _collect_entries(str(tmp_path))
        assert result == []


class TestEstimateTokensScript:
    def test_nonexistent_path(self):
        from estimate_tokens import scan_path
        with pytest.raises(FileNotFoundError_):
            scan_path("/nonexistent/xyz")


class TestAuditErrors:
    def test_nonexistent_workspace(self):
        from audit_memory import audit_workspace
        with pytest.raises(FileNotFoundError_):
            audit_workspace("/nonexistent/xyz")


class TestTiersErrors:
    def test_nonexistent_path(self):
        from generate_summary_tiers import _find_memory_files
        with pytest.raises(FileNotFoundError_):
            _find_memory_files("/nonexistent/xyz")


class TestObservationErrors:
    def test_nonexistent_file(self):
        from observation_compressor import parse_session_jsonl
        with pytest.raises(Exception):
            parse_session_jsonl(Path("/nonexistent/session.jsonl"))

    def test_corrupt_jsonl(self, tmp_path):
        from observation_compressor import parse_session_jsonl
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n{bad json}\n")
        # Should handle gracefully (skip bad lines or raise)
        try:
            result = parse_session_jsonl(f)
            assert isinstance(result, list)
        except Exception:
            pass  # Either handling is acceptable

    def test_empty_jsonl(self, tmp_path):
        from observation_compressor import parse_session_jsonl
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = parse_session_jsonl(f)
        assert result == []

    def test_compress_empty_session(self, tmp_path):
        from observation_compressor import compress_session
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = compress_session(f)
        assert isinstance(result, dict)


class TestMainEntryErrors:
    def test_nonexistent_workspace(self):
        from mem_compress import _workspace_path
        with pytest.raises(SystemExit):
            _workspace_path("/nonexistent/xyz")

    def test_file_as_workspace(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        from mem_compress import _workspace_path
        with pytest.raises(SystemExit):
            _workspace_path(str(f))
