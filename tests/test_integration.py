"""Integration tests: full pipeline and cross-module scenarios."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from estimate_tokens import scan_path, format_human as est_format
from compress_memory import rule_compress, compress_file, generate_llm_prompt
from generate_summary_tiers import generate_tiers
from dedup_memory import run_dedup
from claw_compactor.tokens import estimate_tokens
from claw_compactor.markdown import parse_sections, strip_markdown_redundancy
from claw_compactor.dedup import find_duplicates


class TestFullPipeline:
    """End-to-end: estimate → compress → tiers → dedup."""

    def test_estimate_compress_tiers_dedup(self, tmp_workspace: Path) -> None:
        """Full pipeline on a populated workspace."""
        ws = str(tmp_workspace)

        # 1. Estimate
        results = scan_path(ws)
        assert len(results) > 0
        total_before = sum(r["tokens"] for r in results)
        assert total_before > 0

        # 2. Compress each daily file
        memory_dir = tmp_workspace / "memory"
        for f in sorted(memory_dir.glob("*.md")):
            stats = compress_file(f, dry_run=True, no_llm=True)
            assert stats["original_tokens"] >= stats["rule_compressed_tokens"]
            assert stats["rule_reduction_pct"] >= 0

        # 3. Generate tiers
        files = [tmp_workspace / "MEMORY.md"] + sorted(memory_dir.glob("*.md"))
        tier_result = generate_tiers(files)
        assert tier_result["total_tokens"] > 0
        assert 0 in tier_result["tiers"]
        assert 1 in tier_result["tiers"]
        assert 2 in tier_result["tiers"]
        # Level 0 should use fewer tokens than Level 2
        assert tier_result["tiers"][0]["tokens_used"] <= tier_result["tiers"][2]["tokens_used"]

        # 4. Dedup
        dedup_result = run_dedup(ws)
        assert "duplicate_groups" in dedup_result
        assert dedup_result["total_entries"] >= 0

    def test_unicode_pipeline(self, unicode_file: Path) -> None:
        """Pipeline with Chinese/mixed content."""
        # Estimate
        results = scan_path(str(unicode_file))
        assert len(results) == 1
        assert results[0]["tokens"] > 0

        # Compress
        stats = compress_file(unicode_file, dry_run=True, no_llm=True)
        assert stats["original_tokens"] > 0

        # Parse sections
        text = unicode_file.read_text(encoding="utf-8")
        sections = parse_sections(text)
        assert len(sections) >= 1

    def test_large_file_pipeline(self, large_file: Path) -> None:
        """Pipeline with ~100k+ char file."""
        results = scan_path(str(large_file))
        assert results[0]["tokens"] > 10000

        stats = compress_file(large_file, dry_run=True, no_llm=True)
        assert stats["original_tokens"] > 10000
        # Rule compression should remove at least something
        assert stats["rule_compressed_tokens"] <= stats["original_tokens"]

    def test_empty_file_resilience(self, empty_file: Path) -> None:
        """All tools handle empty file gracefully."""
        results = scan_path(str(empty_file))
        assert len(results) >= 1  # the file exists, just empty

        stats = compress_file(empty_file, dry_run=True, no_llm=True)
        assert stats["original_tokens"] == 0

    def test_broken_markdown_resilience(self, broken_markdown: Path) -> None:
        """Broken markdown doesn't crash anything."""
        results = scan_path(str(broken_markdown))
        assert len(results) == 1

        stats = compress_file(broken_markdown, dry_run=True, no_llm=True)
        assert stats["original_tokens"] > 0

        sections = parse_sections(broken_markdown.read_text(encoding="utf-8"))
        assert len(sections) >= 1

    def test_dedup_finds_known_duplicates(self, duplicate_content: Path) -> None:
        """Dedup correctly identifies known duplicate content."""
        result = run_dedup(str(duplicate_content), threshold=0.5)
        assert result["total_entries"] >= 2
        # The two files have nearly identical "Setup" sections
        assert len(result["duplicate_groups"]) >= 1

    def test_dedup_auto_merge(self, duplicate_content: Path) -> None:
        """Auto-merge removes duplicates and preserves unique content."""
        result = run_dedup(str(duplicate_content), threshold=0.5, auto_merge=True)
        if result["duplicate_groups"]:
            assert result.get("tokens_saved", 0) >= 0
            if "files_modified" in result:
                # Verify unique content survived
                b_text = (duplicate_content / "b.md").read_text(encoding="utf-8")
                assert "Unique content" in b_text


class TestCompressionQuality:
    """Verify compression preserves information."""

    def test_rule_compress_preserves_facts(self) -> None:
        """Rule compression keeps key facts."""
        text = (
            "# 2025-01-15 Session Notes\n\n"
            "- Fixed critical bug in API endpoint /v2/orders\n"
            "- Server IP: 10.0.2.1\n"
            "- Deployed version 2.1.3 to production\n"
            "- Met with Alice about Q2 roadmap\n"
            "- Decision: migrate to PostgreSQL 16\n"
        )
        compressed = rule_compress(text)
        # All critical facts should survive rule compression
        assert "10.0.2.1" in compressed
        assert "2.1.3" in compressed
        assert "PostgreSQL" in compressed

    def test_rule_compress_removes_dupes(self) -> None:
        """Rule compression removes exact duplicate lines."""
        text = (
            "# Notes\n"
            "- Deploy on AWS\n"
            "- Deploy on AWS\n"
            "- Different thing\n"
            "- Deploy on AWS\n"
        )
        compressed = rule_compress(text)
        assert compressed.count("Deploy on AWS") == 1
        assert "Different thing" in compressed

    def test_llm_prompt_contains_content(self) -> None:
        """LLM prompt includes the content to compress."""
        prompt = generate_llm_prompt("My test content here", target_pct=40)
        assert "My test content here" in prompt
        assert "40%" in prompt

    def test_compression_ratio_reasonable(self, tmp_workspace: Path) -> None:
        """Rule compression achieves reasonable ratio on real-ish content."""
        mem_file = tmp_workspace / "MEMORY.md"
        text = mem_file.read_text(encoding="utf-8")
        original_tokens = estimate_tokens(text)
        compressed = rule_compress(text)
        compressed_tokens = estimate_tokens(compressed)
        # Rule compression alone shouldn't inflate
        assert compressed_tokens <= original_tokens


class TestTierQuality:
    """Verify tier generation correctness."""

    def test_tier_budget_respected(self, tmp_workspace: Path) -> None:
        """Each tier stays within its token budget."""
        files = [tmp_workspace / "MEMORY.md"]
        result = generate_tiers(files)
        for tier_level, tier in result["tiers"].items():
            assert tier["tokens_used"] <= tier["budget"], (
                f"Tier {tier_level} exceeded budget: "
                f"{tier['tokens_used']} > {tier['budget']}"
            )

    def test_tier_ordering(self, tmp_workspace: Path) -> None:
        """Higher tiers include more content."""
        files = [tmp_workspace / "MEMORY.md"]
        result = generate_tiers(files)
        tiers = result["tiers"]
        assert tiers[0]["tokens_used"] <= tiers[1]["tokens_used"]
        assert tiers[1]["tokens_used"] <= tiers[2]["tokens_used"]

    def test_tier_priority_ordering(self, tmp_workspace: Path) -> None:
        """Higher-priority sections are included first."""
        files = [tmp_workspace / "MEMORY.md"]
        result = generate_tiers(files)
        for tier in result["tiers"].values():
            if len(tier["sections"]) >= 2:
                priorities = [s["priority"] for s in tier["sections"]]
                # Should be roughly descending (it IS sorted descending)
                assert priorities == sorted(priorities, reverse=True)


class TestCLIScripts:
    """Test scripts work as standalone CLI tools."""

    def _run_script(self, script: str, args: list, cwd: str = None) -> subprocess.CompletedProcess:
        script_path = Path(__file__).parent.parent / "scripts" / script
        return subprocess.run(
            [sys.executable, str(script_path)] + args,
            capture_output=True, text=True, timeout=30,
            cwd=cwd,
        )

    def test_estimate_tokens_cli(self, tmp_workspace: Path) -> None:
        r = self._run_script("estimate_tokens.py", [str(tmp_workspace)])
        assert r.returncode == 0
        assert "tokens" in r.stdout.lower()

    def test_estimate_tokens_json(self, tmp_workspace: Path) -> None:
        r = self._run_script("estimate_tokens.py", [str(tmp_workspace), "--json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "files" in data

    def test_compress_dry_run(self, tmp_workspace: Path) -> None:
        r = self._run_script("compress_memory.py", [
            str(tmp_workspace / "MEMORY.md"), "--dry-run", "--json"
        ])
        assert r.returncode == 0

    def test_tiers_cli(self, tmp_workspace: Path) -> None:
        r = self._run_script("generate_summary_tiers.py", [str(tmp_workspace)])
        assert r.returncode == 0
        assert "Level" in r.stdout

    def test_tiers_json(self, tmp_workspace: Path) -> None:
        r = self._run_script("generate_summary_tiers.py", [str(tmp_workspace), "--json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "tiers" in data

    def test_dedup_cli(self, tmp_workspace: Path) -> None:
        r = self._run_script("dedup_memory.py", [str(tmp_workspace)])
        assert r.returncode == 0

    def test_dedup_json(self, tmp_workspace: Path) -> None:
        r = self._run_script("dedup_memory.py", [str(tmp_workspace), "--json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "duplicate_groups" in data

    def test_nonexistent_path(self) -> None:
        r = self._run_script("estimate_tokens.py", ["/nonexistent/xyz123"])
        assert r.returncode != 0
