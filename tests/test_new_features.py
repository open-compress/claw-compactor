"""Tests for new features: llm_compress_file, extract_key_facts, generate_auto_summary."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compress_memory import llm_compress_file
from generate_summary_tiers import extract_key_facts, generate_auto_summary


class TestLlmCompressFile:
    def test_creates_prompt_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\n" + "\n".join(f"- Entry {i}" for i in range(50)))
        r = llm_compress_file(f)
        prompt_path = Path(r["prompt_file"])
        assert prompt_path.exists()
        assert r["original_tokens"] > 0
        assert r["prompt_tokens"] > r["rule_compressed_tokens"]
        # Cleanup
        prompt_path.unlink()

    def test_stats(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Simple\n\nContent\n")
        r = llm_compress_file(f, target_pct=30)
        assert r["target_pct"] == 30
        assert "instruction" in r
        Path(r["prompt_file"]).unlink()


class TestExtractKeyFacts:
    def test_key_value(self):
        facts = extract_key_facts("- Name: Alex\n- IP: 192.168.1.1\n")
        assert any("Alex" in f for f in facts)
        assert any("192.168.1.1" in f for f in facts)

    def test_important_markers(self):
        facts = extract_key_facts("⚠️ Critical: do not delete\nNormal line\n")
        assert any("Critical" in f for f in facts)

    def test_empty(self):
        assert extract_key_facts("") == []

    def test_headers_skipped(self):
        facts = extract_key_facts("# Header\n## Sub\nContent: value\n")
        assert not any("Header" in f for f in facts)

    def test_dedup(self):
        facts = extract_key_facts("- A: 1\n- A: 1\n")
        assert len(facts) == 1


class TestGenerateAutoSummary:
    def test_basic(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Notes\n\n- Server: 192.168.1.1\n- Port: 8080\n")
        summary = generate_auto_summary([f], budget=200)
        assert "# Auto Summary" in summary
        assert "192.168.1.1" in summary

    def test_budget_limit(self, tmp_path):
        f = tmp_path / "big.md"
        lines = "\n".join(f"- Key{i}: Value{i} with some extra text" for i in range(200))
        f.write_text(f"# Big\n\n{lines}\n")
        summary = generate_auto_summary([f], budget=100)
        # Should be within budget
        from claw_compactor.tokens import estimate_tokens
        assert estimate_tokens(summary) <= 120  # small margin

    def test_empty_files(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        summary = generate_auto_summary([f])
        assert "# Auto Summary" in summary
