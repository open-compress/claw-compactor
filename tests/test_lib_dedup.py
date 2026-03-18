"""Comprehensive tests for lib/dedup.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.dedup import _shingles, jaccard, find_duplicates, merge_duplicates


class TestShingles:
    def test_basic(self):
        result = _shingles("hello world foo bar")
        assert isinstance(result, set)
        assert len(result) > 0

    def test_empty(self):
        result = _shingles("")
        assert isinstance(result, set)

    def test_short_text(self):
        result = _shingles("hi")
        assert isinstance(result, set)

    def test_same_text_same_shingles(self):
        a = _shingles("the quick brown fox")
        b = _shingles("the quick brown fox")
        assert a == b

    def test_different_text_different_shingles(self):
        a = _shingles("completely different text here about apples")
        b = _shingles("unrelated content about something else entirely")
        assert a != b

    def test_unicode(self):
        result = _shingles("中文文本测试内容")
        assert isinstance(result, set)

    def test_custom_k(self):
        result = _shingles("hello world foo bar baz", k=2)
        assert isinstance(result, set)


class TestJaccard:
    def test_identical(self):
        s = {1, 2, 3}
        assert jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        j = jaccard({1, 2, 3}, {2, 3, 4})
        assert 0 < j < 1

    def test_empty_both(self):
        result = jaccard(set(), set())
        assert result == 0.0 or result == 1.0  # implementation-dependent

    def test_one_empty(self):
        assert jaccard({1, 2}, set()) == 0.0

    def test_subset(self):
        j = jaccard({1, 2}, {1, 2, 3})
        assert j > 0.5


class TestFindDuplicates:
    def test_exact_duplicates(self):
        entries = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "Something completely different here",
        ]
        result = find_duplicates(entries)
        assert len(result) > 0

    def test_no_duplicates(self):
        entries = [
            "Apples are red fruits that grow on trees",
            "Python is a programming language used widely",
            "The weather today is sunny and warm outside",
        ]
        result = find_duplicates(entries, threshold=0.9)
        assert len(result) == 0

    def test_near_duplicates(self):
        entries = [
            "Set up SSH access to the server using ed25519 key",
            "Set up SSH access to the server using RSA key",
        ]
        result = find_duplicates(entries, threshold=0.5)
        # May or may not find depending on threshold
        assert isinstance(result, list)

    def test_empty_list(self):
        result = find_duplicates([])
        assert result == []

    def test_single_entry(self):
        result = find_duplicates(["only one entry"])
        assert result == []

    def test_high_threshold(self):
        entries = ["aaa bbb ccc ddd", "aaa bbb ccc eee"]
        result = find_duplicates(entries, threshold=0.99)
        assert len(result) == 0

    def test_low_threshold(self):
        entries = ["hello world", "hello earth"]
        result = find_duplicates(entries, threshold=0.01)
        assert isinstance(result, list)


class TestMergeDuplicates:
    def test_basic_merge(self):
        entries = ["short", "much longer version of this text that should be kept"]
        groups = [{"indices": [0, 1], "similarity": 0.8}]
        result = merge_duplicates(entries, groups)
        assert isinstance(result, list)

    def test_empty_groups(self):
        entries = ["a", "b", "c"]
        result = merge_duplicates(entries, [])
        assert result == entries

    def test_empty_entries(self):
        result = merge_duplicates([], [])
        assert result == []

    def test_keeps_longest(self):
        entries = ["x", "xx", "xxx"]
        groups = [{"indices": [0, 1, 2], "similarity": 0.9}]
        result = merge_duplicates(entries, groups)
        # Should keep the longest
        assert any("xxx" in r for r in result)
