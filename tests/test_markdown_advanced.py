"""Tests for advanced markdown compression functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.markdown import (
    normalize_chinese_punctuation,
    strip_emoji,
    remove_empty_sections,
    compress_markdown_table,
    merge_similar_bullets,
    merge_short_bullets,
)


class TestNormalizeChinesePunctuation:
    def test_basic(self):
        assert normalize_chinese_punctuation("你好，世界！") == "你好,世界!"

    def test_all_punctuation(self):
        r = normalize_chinese_punctuation('说："你好"；（这里）')
        assert '；' not in r
        assert '\u201c' not in r and '\u201d' not in r
        assert '（' not in r and '）' not in r

    def test_no_change_ascii(self):
        text = "Hello, world!"
        assert normalize_chinese_punctuation(text) == text

    def test_mixed(self):
        r = normalize_chinese_punctuation("IP：192.168.1.1，端口：8080")
        assert r == "IP:192.168.1.1,端口:8080"

    def test_empty(self):
        assert normalize_chinese_punctuation("") == ""

    def test_brackets(self):
        assert normalize_chinese_punctuation("【重要】") == "[重要]"


class TestStripEmoji:
    def test_removes_emoji(self):
        assert "Title" in strip_emoji("⚠️ Title")
        assert "⚠" not in strip_emoji("⚠️ Title")

    def test_preserves_text(self):
        assert strip_emoji("no emoji here") == "no emoji here"

    def test_multiple_emoji(self):
        r = strip_emoji("🏆 Winner 🎉 Party 🚀 Launch")
        assert "Winner" in r and "Party" in r and "Launch" in r
        assert "🏆" not in r

    def test_empty(self):
        assert strip_emoji("") == ""

    def test_only_emoji(self):
        r = strip_emoji("🎉🎊🎈")
        assert r.strip() == ""

    def test_chinese_with_emoji(self):
        r = strip_emoji("📊 项目状态")
        assert "项目状态" in r

    def test_no_double_spaces(self):
        r = strip_emoji("A 🎉 B")
        assert "  " not in r


class TestRemoveEmptySections:
    def test_removes_empty(self):
        text = "# Title\n\nContent\n\n## Empty\n\n## HasContent\n\nStuff"
        r = remove_empty_sections(text)
        assert "Empty" not in r
        assert "HasContent" in r

    def test_preserves_parent_with_children(self):
        text = "# Parent\n\n## Child\n\nContent"
        r = remove_empty_sections(text)
        assert "# Parent" in r
        assert "## Child" in r

    def test_no_sections(self):
        text = "Just plain text"
        assert remove_empty_sections(text) == text

    def test_all_empty(self):
        text = "## A\n\n## B\n\n## C\n"
        r = remove_empty_sections(text)
        assert "A" not in r and "B" not in r and "C" not in r

    def test_empty_input(self):
        assert remove_empty_sections("") == ""

    def test_nested_empty(self):
        text = "# Top\n\n## Mid\n\n### Deep\n\nContent"
        r = remove_empty_sections(text)
        assert "# Top" in r  # has child
        assert "## Mid" in r  # has child
        assert "### Deep" in r  # has content


class TestCompressMarkdownTable:
    def test_two_column(self):
        table = "| Key | Value |\n|-----|-------|\n| A | 1 |\n| B | 2 |"
        r = compress_markdown_table(table)
        assert "- A: 1" in r
        assert "- B: 2" in r
        assert "|" not in r

    def test_three_column(self):
        table = "| Name | Type | IP |\n|------|------|----|\n| srv1 | VPS | 1.2.3.4 |"
        r = compress_markdown_table(table)
        assert "srv1" in r
        assert "Type=VPS" in r
        assert "IP=1.2.3.4" in r

    def test_wide_table_preserved(self):
        header = "| A | B | C | D | E |"
        sep = "|---|---|---|---|---|"
        row = "| 1 | 2 | 3 | 4 | 5 |"
        table = f"{header}\n{sep}\n{row}"
        r = compress_markdown_table(table)
        assert "|" in r  # preserved

    def test_no_table(self):
        text = "# Title\n\nJust text"
        assert compress_markdown_table(text) == text

    def test_mixed_content(self):
        text = "Before\n\n| K | V |\n|---|---|\n| x | y |\n\nAfter"
        r = compress_markdown_table(text)
        assert "Before" in r
        assert "After" in r
        assert "- x: y" in r

    def test_empty_table(self):
        table = "| K | V |\n|---|---|"
        r = compress_markdown_table(table)
        # No rows, should be kept as-is or be harmless
        assert isinstance(r, str)

    def test_real_world_table(self):
        table = (
            "| 节点 | 类型 | IP | 状态 |\n"
            "|------|------|-----|------|\n"
            "| my-server | macOS | localhost | ✅ |\n"
            "| remote-node | VPS | 10.0.1.2 | ✅ |"
        )
        r = compress_markdown_table(table)
        assert "my-server" in r
        assert "|" not in r  # 4 cols = compressed


class TestMergeSimilarBullets:
    def test_merges_near_duplicates(self):
        text = "- Deploy v2.1 to staging\n- Deploy v2.1 to staging server"
        r = merge_similar_bullets(text)
        assert r.count("Deploy") == 1
        assert "server" in r  # kept longer

    def test_preserves_different_items(self):
        text = "- Apple\n- Banana\n- Cherry"
        r = merge_similar_bullets(text)
        assert "Apple" in r and "Banana" in r and "Cherry" in r

    def test_empty(self):
        assert merge_similar_bullets("") == ""

    def test_no_bullets(self):
        text = "Just text\nMore text"
        assert merge_similar_bullets(text) == text

    def test_different_indent_levels(self):
        text = "- Parent\n  - Child A\n  - Child A extended"
        r = merge_similar_bullets(text)
        assert "Parent" in r

    def test_threshold(self):
        text = "- item one\n- item two"
        # These are 0.75 similar, default threshold is 0.80
        r = merge_similar_bullets(text)
        assert "item one" in r and "item two" in r

    def test_three_similar(self):
        text = "- Setup the server config\n- Setup the server configuration\n- Setup server config file"
        r = merge_similar_bullets(text)
        lines = [l for l in r.split('\n') if l.strip().startswith('-')]
        assert len(lines) < 3


class TestMergeShortBullets:
    def test_merges_short(self):
        text = "- A\n- B\n- C\n- D"
        r = merge_short_bullets(text)
        assert "A, B, C, D" in r
        assert r.count('\n- ') == 0 or r.count('\n-') < 4

    def test_preserves_long_bullets(self):
        text = "- This is a long bullet point\n- Another long bullet point"
        r = merge_short_bullets(text)
        assert "This is a long" in r
        assert r.count('\n') >= 1

    def test_mixed(self):
        text = "- A\n- B\n- C\n- This is long enough to not merge"
        r = merge_short_bullets(text)
        assert "A, B, C" in r
        assert "This is long" in r

    def test_fewer_than_three(self):
        text = "- A\n- B"
        r = merge_short_bullets(text)
        # Only 2 short bullets, not enough to merge
        assert "- A" in r and "- B" in r

    def test_empty(self):
        assert merge_short_bullets("") == ""

    def test_no_bullets(self):
        text = "plain text"
        assert merge_short_bullets(text) == text

    def test_max_merge(self):
        text = "\n".join(f"- {chr(65+i)}" for i in range(10))
        r = merge_short_bullets(text, max_merge=3)
        # Should create multiple merged lines
        lines = [l for l in r.split('\n') if l.strip()]
        assert len(lines) < 10


class TestIntegrationNewRules:
    """Test that new rules work together in rule_compress pipeline."""

    def test_real_memory_style(self):
        """Simulate a real memory file with tables, emoji, Chinese punct."""
        from compress_memory import rule_compress

        text = """# 📊 项目状态

## ⚠️ 铁律

1. **主 session 不跑 exec** — 全部派 sub-agent
2. **交易所 API 全走日本节点** — CCXT routed through remote-node

## 节点

| 节点 | 类型 | IP |
|------|------|-----|
| my-server | macOS | localhost |
| remote-node | VPS | 10.0.1.2 |

## 空标题

## 数据

- 下载数据
- 下载数据到服务器
- A
- B
- C
- D
"""
        r = rule_compress(text)
        # Emoji stripped
        assert "📊" not in r
        # Table compressed
        assert "| 节点 |" not in r  # no table headers
        assert "my-server" in r
        # Empty section removed
        assert "空标题" not in r  # removed (truly empty, no child)
        # Short bullets merged (A, B, C, D are short single-char items)
        assert "A, B, C, D" in r

    def test_chinese_punctuation_in_pipeline(self):
        from compress_memory import rule_compress

        text = "# 笔记\n\n- 节点：remote-node，状态：在线"
        r = rule_compress(text)
        assert ":" in r  # Chinese colons converted
        assert "," in r  # Chinese commas converted
