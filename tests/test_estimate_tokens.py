"""Tests for estimate_tokens.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from estimate_tokens import scan_path, format_human, _score_potential
from claw_compactor.tokens import estimate_tokens, using_tiktoken


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_simple_english(self):
        tokens = estimate_tokens("Hello world, this is a test.")
        assert 0 < tokens < 50

    def test_unicode_chinese(self):
        assert estimate_tokens("你好世界") > 0

    def test_large_input(self):
        assert estimate_tokens("word " * 25000) > 1000


class TestScorePotential:
    def test_high_tokens(self):
        assert _score_potential(3000, 2500) == "high"

    def test_high_reduction(self):
        assert _score_potential(1000, 800) == "high"

    def test_medium(self):
        assert _score_potential(600, 560) == "medium"

    def test_low(self):
        assert _score_potential(100, 98) == "low"

    def test_zero(self):
        assert _score_potential(0, 0) == "low"


class TestScanPath:
    def test_workspace(self, tmp_workspace):
        results = scan_path(str(tmp_workspace))
        assert len(results) > 0
        assert all("tokens" in r for r in results)

    def test_single_file(self, tmp_workspace):
        assert len(scan_path(str(tmp_workspace / "MEMORY.md"))) == 1

    def test_empty_file(self, empty_file):
        results = scan_path(str(empty_file))
        assert len(results) == 1
        assert results[0]["tokens"] == 0

    def test_nonexistent(self):
        with pytest.raises(Exception):
            scan_path("/nonexistent/xyz")

    def test_large_file(self, large_file):
        results = scan_path(str(large_file))
        assert results[0]["tokens"] > 1000

    def test_unicode(self, unicode_file):
        assert scan_path(str(unicode_file))[0]["tokens"] > 0

    def test_broken_markdown(self, broken_markdown):
        assert len(scan_path(str(broken_markdown))) == 1

    def test_threshold(self, tmp_workspace):
        assert len(scan_path(str(tmp_workspace), threshold=999999)) == 0

    def test_headers_only(self, headers_only):
        assert len(scan_path(str(headers_only))) == 1

    def test_single_line(self, single_line):
        assert scan_path(str(single_line))[0]["tokens"] > 0

    def test_sorted_desc(self, tmp_workspace):
        tokens = [r["tokens"] for r in scan_path(str(tmp_workspace))]
        assert tokens == sorted(tokens, reverse=True)


class TestFormatHuman:
    def test_empty(self):
        assert "No files" in format_human([])

    def test_normal(self, tmp_workspace):
        output = format_human(scan_path(str(tmp_workspace)))
        assert "Token Estimation Report" in output

    def test_json_roundtrip(self, tmp_workspace):
        results = scan_path(str(tmp_workspace))
        j = json.dumps({"files": results})
        assert "files" in json.loads(j)
