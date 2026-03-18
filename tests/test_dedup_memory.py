"""Tests for dedup_memory.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dedup_memory import run_dedup, format_human, _collect_entries
from claw_compactor.dedup import find_duplicates, merge_duplicates, jaccard, _shingles


class TestShingles:
    def test_basic(self):
        assert len(_shingles("the quick brown fox jumps over")) > 0

    def test_short(self):
        assert len(_shingles("hi")) == 1

    def test_empty(self):
        assert len(_shingles("")) == 1


class TestJaccard:
    def test_identical(self):
        assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint(self):
        assert jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        assert 0 < jaccard({1, 2, 3}, {2, 3, 4}) < 1

    def test_empty(self):
        assert jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        assert jaccard({1}, set()) == 0.0


class TestFindDuplicates:
    def test_exact(self):
        entries = ["Install Python 3.10 and configure environment", "Install Python 3.10 and configure environment", "Something different about database"]
        groups = find_duplicates(entries, threshold=0.5)
        assert len(groups) >= 1

    def test_near(self):
        entries = ["Install Python 3.10 and set up virtual env for dev", "Install Python 3.10 and set up virtual env for prod", "Configure nginx reverse proxy"]
        assert len(find_duplicates(entries, threshold=0.5)) >= 1

    def test_no_duplicates(self):
        assert len(find_duplicates(["Alpha beta gamma", "One two three four", "日本語テスト"], threshold=0.8)) == 0

    def test_single(self):
        assert find_duplicates(["one"]) == []

    def test_empty(self):
        assert find_duplicates([]) == []

    def test_large(self):
        entries = ["Entry {} with filler content for testing".format(i) for i in range(500)]
        entries += [entries[0], entries[1]]
        assert len(find_duplicates(entries, threshold=0.8)) >= 2

    def test_unicode(self):
        entries = ["配置数据库连接字符串和环境变量", "配置数据库连接字符串和环境变量", "不同的内容关于部署"]
        assert len(find_duplicates(entries, threshold=0.5)) >= 1


class TestMergeDuplicates:
    def test_no_groups(self):
        assert merge_duplicates(["a", "b"], []) == ["a", "b"]


class TestCollectEntries:
    def test_workspace(self, tmp_workspace):
        entries = _collect_entries(str(tmp_workspace))
        assert len(entries) > 0

    def test_nonexistent(self):
        with pytest.raises(Exception):
            _collect_entries("/nonexistent/xyz")

    def test_empty(self, empty_file):
        assert len(_collect_entries(str(empty_file))) == 0

    def test_unicode(self, unicode_file):
        assert len(_collect_entries(str(unicode_file))) > 0


class TestRunDedup:
    def test_basic(self, tmp_workspace):
        result = run_dedup(str(tmp_workspace))
        assert "total_entries" in result
        assert "duplicate_groups" in result

    def test_with_dupes(self, duplicate_content):
        result = run_dedup(str(duplicate_content), threshold=0.5)
        assert result["total_entries"] > 0

    def test_auto_merge(self, duplicate_content):
        result = run_dedup(str(duplicate_content), threshold=0.5, auto_merge=True)
        assert "tokens_before" in result

    def test_empty(self, tmp_path):
        (tmp_path / "e.md").write_text("", encoding="utf-8")
        assert run_dedup(str(tmp_path))["total_entries"] == 0

    def test_nonexistent(self):
        with pytest.raises(Exception):
            run_dedup("/nonexistent/xyz")

    def test_json(self, tmp_workspace):
        assert json.loads(json.dumps(run_dedup(str(tmp_workspace))))


class TestFormatHuman:
    def test_no_dupes(self):
        assert "No duplicates" in format_human({"total_entries": 5, "duplicate_groups": [], "entries_removed": 0})

    def test_with_dupes(self, duplicate_content):
        output = format_human(run_dedup(str(duplicate_content), threshold=0.3))
        assert "Deduplication Report" in output
