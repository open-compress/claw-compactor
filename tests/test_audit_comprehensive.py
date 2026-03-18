"""Comprehensive tests for audit_memory.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audit_memory import (
    audit_file, audit_workspace, format_report,
    _file_age_days, _has_tables, _has_emoji, _count_empty_sections,
)


class TestHelpers:
    def test_file_age_days(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("x")
        age = _file_age_days(f)
        assert 0 <= age < 1  # Just created

    def test_has_tables_true(self):
        assert _has_tables("| A | B |\n|---|---|\n| 1 | 2 |")

    def test_has_tables_false(self):
        assert not _has_tables("no tables here")

    def test_has_tables_empty(self):
        assert not _has_tables("")

    def test_has_tables_pipe_no_separator(self):
        assert not _has_tables("a | b\nc | d")

    def test_has_emoji_true(self):
        assert _has_emoji("hello 🎉 world")

    def test_has_emoji_false(self):
        assert not _has_emoji("plain text")

    def test_has_emoji_empty(self):
        assert not _has_emoji("")

    def test_count_empty_sections(self):
        text = "# A\nfoo\n# B\n\n# C\nbar"
        count = _count_empty_sections(text)
        assert count >= 1

    def test_count_empty_sections_none(self):
        text = "# A\nfoo\n# B\nbar"
        count = _count_empty_sections(text)
        assert count == 0

    def test_count_empty_sections_empty_text(self):
        assert _count_empty_sections("") == 0


class TestAuditFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "MEMORY.md"
        f.write_text("# Memory\n\n## Decisions\n- Use Python\n")
        result = audit_file(f)
        assert "tokens" in result
        assert "path" in result

    def test_with_tables(self, tmp_path):
        f = tmp_path / "TOOLS.md"
        f.write_text("| A | B |\n|---|---|\n| 1 | 2 |")
        result = audit_file(f)
        suggestions = " ".join(result.get("suggestions", []))
        assert "table" in suggestions.lower() or "compress" in suggestions.lower()

    def test_with_emoji(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("🎉 Party time! 🎊 Fun! 🎈 Balloons!")
        result = audit_file(f)
        # May suggest emoji stripping or just note it
        assert isinstance(result, dict)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        result = audit_file(f)
        assert result["tokens"] == 0

    def test_stale_detection(self, tmp_path):
        f = tmp_path / "old.md"
        f.write_text("old content")
        result = audit_file(f, stale_days=0)
        assert isinstance(result.get("is_stale"), bool)


class TestAuditWorkspace:
    def test_basic(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Memory\nContent\n")
        (tmp_path / "TOOLS.md").write_text("# Tools\nMore\n")
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "2026-01-01.md").write_text("Daily note\n")
        result = audit_workspace(str(tmp_path))
        assert isinstance(result, dict)
        assert "total_tokens" in result or "files" in result

    def test_empty_workspace(self, tmp_path):
        result = audit_workspace(str(tmp_path))
        assert isinstance(result, dict)

    def test_nonexistent(self):
        from claw_compactor.exceptions import FileNotFoundError_
        with pytest.raises(FileNotFoundError_):
            audit_workspace("/nonexistent/xyz")


class TestFormatReport:
    def test_basic(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Memory\nContent\n")
        result = audit_workspace(str(tmp_path))
        report = format_report(result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_empty_report(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("")
        result = audit_workspace(str(tmp_path))
        report = format_report(result)
        assert isinstance(report, str)
