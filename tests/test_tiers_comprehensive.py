"""Comprehensive tests for generate_summary_tiers.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_summary_tiers import (
    _classify_section, _find_memory_files, generate_tiers, format_human,
    extract_key_facts, generate_auto_summary, format_tier_template, TIERS,
)


class TestClassifySection:
    def test_decision(self):
        assert _classify_section("Important Decisions") >= 8

    def test_action(self):
        assert _classify_section("Action Items") >= 7

    def test_config(self):
        assert _classify_section("Configuration Setup") >= 6

    def test_notes(self):
        assert _classify_section("Random Notes") >= 3

    def test_history(self):
        assert _classify_section("History Log") >= 1

    def test_unknown(self):
        result = _classify_section("zzzzzzzzzzz")
        assert isinstance(result, int)

    def test_empty(self):
        result = _classify_section("")
        assert isinstance(result, int)

    def test_case_insensitive(self):
        a = _classify_section("DECISION")
        b = _classify_section("decision")
        assert a == b


class TestFindMemoryFiles:
    def test_finds_files(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("memory")
        (tmp_path / "TOOLS.md").write_text("tools")
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "daily.md").write_text("daily")
        files = _find_memory_files(str(tmp_path))
        assert len(files) >= 2

    def test_empty_workspace(self, tmp_path):
        files = _find_memory_files(str(tmp_path))
        assert isinstance(files, list)

    def test_nonexistent(self):
        from claw_compactor.exceptions import FileNotFoundError_
        with pytest.raises(FileNotFoundError_):
            _find_memory_files("/nonexistent/xyz")


class TestGenerateTiers:
    def test_basic(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Memory\n## Decisions\n- Use Python\n## Notes\n- Hello\n")
        files = [tmp_path / "MEMORY.md"]
        result = generate_tiers(files)
        assert isinstance(result, dict)

    def test_returns_tier_info(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# M\n## A\nfoo\n## B\nbar\n")
        files = [tmp_path / "MEMORY.md"]
        result = generate_tiers(files)
        assert "tiers" in result or "sections" in result

    def test_empty_files(self, tmp_path):
        (tmp_path / "empty.md").write_text("")
        result = generate_tiers([tmp_path / "empty.md"])
        assert isinstance(result, dict)

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\nContent A\n")
        (tmp_path / "b.md").write_text("# B\nContent B\n")
        result = generate_tiers([tmp_path / "a.md", tmp_path / "b.md"])
        assert isinstance(result, dict)


class TestFormatHuman:
    def test_basic(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Memory\n## Decisions\n- X\n")
        files = [tmp_path / "MEMORY.md"]
        result = generate_tiers(files)
        output = format_human(result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_mentions_levels(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# M\n## A\nfoo\n")
        result = generate_tiers([tmp_path / "MEMORY.md"])
        output = format_human(result)
        assert "Level" in output or "level" in output


class TestExtractKeyFacts:
    def test_basic(self):
        text = "Server IP: 192.168.1.100\nPort: 8080\nSome filler text\nDecision: Use Python"
        facts = extract_key_facts(text)
        assert isinstance(facts, list)

    def test_empty(self):
        facts = extract_key_facts("")
        assert facts == []

    def test_bullets(self):
        text = "- Important: use SSH keys\n- Note: backup at 3am\n- Random stuff"
        facts = extract_key_facts(text)
        assert isinstance(facts, list)

    def test_preserves_key_info(self):
        text = "Decision: migrate to Python 3.11\nAction: update CI pipeline"
        facts = extract_key_facts(text)
        assert len(facts) > 0


class TestGenerateAutoSummary:
    def test_basic(self, tmp_path):
        (tmp_path / "test.md").write_text("# Config\nServer: 192.168.1.100\nPort: 8080\nUser: admin\n")
        summary = generate_auto_summary([tmp_path / "test.md"])
        assert isinstance(summary, str)

    def test_custom_budget(self, tmp_path):
        (tmp_path / "test.md").write_text("# Test\nContent\n")
        summary = generate_auto_summary([tmp_path / "test.md"], budget=50)
        assert isinstance(summary, str)

    def test_empty_file(self, tmp_path):
        (tmp_path / "empty.md").write_text("")
        summary = generate_auto_summary([tmp_path / "empty.md"])
        assert isinstance(summary, str)


class TestTiersConfig:
    def test_tier_0_exists(self):
        assert 0 in TIERS
        assert TIERS[0]["budget"] <= 300

    def test_tier_1_exists(self):
        assert 1 in TIERS
        assert TIERS[1]["budget"] <= 1500

    def test_tier_2_exists(self):
        assert 2 in TIERS
        assert TIERS[2]["budget"] <= 5000

    def test_budgets_ascending(self):
        assert TIERS[0]["budget"] < TIERS[1]["budget"] < TIERS[2]["budget"]


class TestFormatTierTemplate:
    def test_basic(self, tmp_path):
        (tmp_path / "test.md").write_text("# Test\n## Section\nContent\n")
        result = generate_tiers([tmp_path / "test.md"])
        template = format_tier_template(result, 0)
        assert isinstance(template, str)

    def test_all_tiers(self, tmp_path):
        (tmp_path / "test.md").write_text("# Test\n## A\nfoo\n## B\nbar\n")
        result = generate_tiers([tmp_path / "test.md"])
        for level in [0, 1, 2]:
            template = format_tier_template(result, level)
            assert isinstance(template, str)
