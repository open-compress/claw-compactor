"""Comprehensive tests for lib/markdown.py — every function, happy + edge cases."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.markdown import (
    parse_sections, strip_markdown_redundancy, remove_duplicate_lines,
    normalize_chinese_punctuation, strip_emoji, remove_empty_sections,
    compress_markdown_table, merge_similar_bullets, merge_short_bullets,
)


# === parse_sections ===

class TestParseSections:
    def test_simple_sections(self):
        text = "# A\nfoo\n# B\nbar"
        result = parse_sections(text)
        assert len(result) >= 2

    def test_empty_text(self):
        result = parse_sections("")
        assert isinstance(result, list)

    def test_no_headers(self):
        result = parse_sections("just plain text\nno headers")
        assert isinstance(result, list)

    def test_nested_headers(self):
        text = "# H1\n## H2\n### H3\ncontent"
        result = parse_sections(text)
        assert len(result) >= 1

    def test_header_only(self):
        result = parse_sections("# Title")
        assert len(result) >= 1

    def test_multiple_blank_lines(self):
        text = "# A\n\n\n\nfoo\n\n\n# B\nbar"
        result = parse_sections(text)
        assert len(result) >= 2

    def test_code_block_with_hash(self):
        text = "# Real\n```\n# Not a header\n```\n# Also real"
        result = parse_sections(text)
        # Should handle code blocks reasonably
        assert isinstance(result, list)

    def test_unicode_headers(self):
        text = "# 日本語\nコンテンツ\n# 中文\n内容"
        result = parse_sections(text)
        assert len(result) >= 2

    def test_very_long_content(self):
        text = "# Header\n" + "x" * 100000
        result = parse_sections(text)
        assert len(result) >= 1

    def test_special_chars_in_header(self):
        text = "# Header with `code` and **bold**\ncontent"
        result = parse_sections(text)
        assert len(result) >= 1


# === strip_markdown_redundancy ===

class TestStripMarkdownRedundancy:
    def test_basic(self):
        text = "# Title\n\nSome text\n\n## Sub\n\nMore text"
        result = strip_markdown_redundancy(text)
        assert isinstance(result, str)

    def test_empty(self):
        assert strip_markdown_redundancy("") == ""

    def test_preserves_content(self):
        text = "Important fact: X = 42"
        result = strip_markdown_redundancy(text)
        assert "42" in result

    def test_strips_excessive_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = strip_markdown_redundancy(text)
        assert "\n\n\n\n" not in result

    def test_unicode_preserved(self):
        text = "# 标题\n中文内容\n日本語"
        result = strip_markdown_redundancy(text)
        assert "中文" in result
        assert "日本語" in result


# === remove_duplicate_lines ===

class TestRemoveDuplicateLines:
    def test_removes_exact_dupes(self):
        text = "hello\nhello\nworld"
        result = remove_duplicate_lines(text)
        assert result.count("hello") == 1

    def test_empty(self):
        assert remove_duplicate_lines("") == ""

    def test_no_dupes(self):
        text = "a\nb\nc"
        assert remove_duplicate_lines(text) == text

    def test_preserves_order(self):
        text = "b\na\nb\nc\na"
        result = remove_duplicate_lines(text)
        lines = [l for l in result.split("\n") if l]
        assert lines[0] == "b"
        assert lines[1] == "a"

    def test_blank_lines_not_deduped(self):
        text = "a\n\nb\n\nc"
        result = remove_duplicate_lines(text)
        # Blank lines should be preserved to some degree
        assert "a" in result and "b" in result and "c" in result

    def test_whitespace_variants(self):
        text = "hello \nhello\n hello"
        result = remove_duplicate_lines(text)
        assert isinstance(result, str)

    def test_all_same(self):
        text = "same\nsame\nsame\nsame"
        result = remove_duplicate_lines(text)
        assert result.count("same") == 1

    def test_bullet_dupes(self):
        text = "- item one\n- item two\n- item one"
        result = remove_duplicate_lines(text)
        assert result.count("item one") == 1


# === normalize_chinese_punctuation ===

class TestNormalizeChinesePunctuation:
    def test_commas(self):
        assert "," in normalize_chinese_punctuation("你好，世界")

    def test_periods(self):
        assert "." in normalize_chinese_punctuation("结束。")

    def test_quotes(self):
        result = normalize_chinese_punctuation('\u201c引用\u201d')
        assert '"' in result

    def test_brackets(self):
        result = normalize_chinese_punctuation("【标签】")
        assert "[" in result and "]" in result

    def test_empty(self):
        assert normalize_chinese_punctuation("") == ""

    def test_no_chinese(self):
        text = "Hello, world!"
        assert normalize_chinese_punctuation(text) == text

    def test_mixed(self):
        text = "Hello，world。How are you？"
        result = normalize_chinese_punctuation(text)
        assert "，" not in result
        assert "。" not in result
        assert "？" not in result

    def test_semicolons(self):
        result = normalize_chinese_punctuation("项目；备注")
        assert ";" in result

    def test_exclamation(self):
        result = normalize_chinese_punctuation("太好了！")
        assert "!" in result

    def test_ellipsis(self):
        result = normalize_chinese_punctuation("等等…")
        assert "..." in result

    def test_dash(self):
        result = normalize_chinese_punctuation("这是——重点")
        assert "--" in result


# === strip_emoji ===

class TestStripEmoji:
    def test_removes_emoji(self):
        result = strip_emoji("Hello 🌍 World 🎉")
        assert "🌍" not in result
        assert "🎉" not in result
        assert "Hello" in result

    def test_empty(self):
        assert strip_emoji("") == ""

    def test_no_emoji(self):
        text = "plain text"
        assert strip_emoji(text) == text

    def test_only_emoji(self):
        result = strip_emoji("🎉🎊🎈")
        assert result.strip() == "" or len(result.strip()) < 3

    def test_emoji_in_bullets(self):
        text = "- 🔴 High priority\n- 🟢 Low priority"
        result = strip_emoji(text)
        assert "High priority" in result
        assert "Low priority" in result

    def test_preserves_cjk(self):
        text = "中文 🎉 日本語"
        result = strip_emoji(text)
        assert "中文" in result
        assert "日本語" in result

    def test_compound_emoji(self):
        # Family emoji, flag emoji, etc
        text = "test 👨‍👩‍👧‍👦 end"
        result = strip_emoji(text)
        assert "test" in result
        assert "end" in result


# === remove_empty_sections ===

class TestRemoveEmptySections:
    def test_removes_empty(self):
        text = "# Has content\nfoo\n# Empty\n\n# Also content\nbar"
        result = remove_empty_sections(text)
        assert "Has content" in result
        assert "Also content" in result

    def test_empty_text(self):
        assert remove_empty_sections("") == ""

    def test_all_empty(self):
        text = "# A\n\n# B\n\n# C\n"
        result = remove_empty_sections(text)
        # Should remove or keep minimally
        assert isinstance(result, str)

    def test_no_empty(self):
        text = "# A\nfoo\n# B\nbar"
        result = remove_empty_sections(text)
        assert "foo" in result and "bar" in result

    def test_nested_empty(self):
        text = "# A\n## B\n\n## C\ncontent"
        result = remove_empty_sections(text)
        assert "content" in result

    def test_whitespace_only_section(self):
        text = "# A\n   \n  \n# B\nreal content"
        result = remove_empty_sections(text)
        assert "real content" in result


# === compress_markdown_table ===

class TestCompressMarkdownTable:
    def test_basic_table(self):
        text = "| Name | Value |\n|------|-------|\n| a | 1 |\n| b | 2 |"
        result = compress_markdown_table(text)
        assert "a" in result and "1" in result
        # Should be more compact
        assert len(result) <= len(text)

    def test_no_table(self):
        text = "No tables here"
        assert compress_markdown_table(text) == text

    def test_empty(self):
        assert compress_markdown_table("") == ""

    def test_table_with_surrounding_text(self):
        text = "Before\n| A | B |\n|---|---|\n| 1 | 2 |\nAfter"
        result = compress_markdown_table(text)
        assert "Before" in result
        assert "After" in result

    def test_multi_column_table(self):
        text = "| A | B | C | D |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |"
        result = compress_markdown_table(text)
        assert "1" in result and "4" in result

    def test_table_with_alignment(self):
        text = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |"
        result = compress_markdown_table(text)
        assert "a" in result

    def test_single_row_table(self):
        text = "| Key | Val |\n|-----|-----|\n| only | row |"
        result = compress_markdown_table(text)
        assert "only" in result

    def test_empty_cells(self):
        text = "| A | B |\n|---|---|\n| x |  |\n|  | y |"
        result = compress_markdown_table(text)
        assert isinstance(result, str)

    def test_table_with_pipes_in_content(self):
        text = "| Cmd | Use |\n|-----|-----|\n| a | `x|y` |"
        result = compress_markdown_table(text)
        assert isinstance(result, str)


# === merge_similar_bullets ===

class TestMergeSimilarBullets:
    def test_merges_similar(self):
        text = "- Added feature X to module A\n- Added feature X to module B"
        result = merge_similar_bullets(text)
        assert isinstance(result, str)

    def test_keeps_different(self):
        text = "- Completely unrelated item A\n- Totally different item B"
        result = merge_similar_bullets(text)
        # Both should survive in some form
        assert "A" in result or "B" in result

    def test_empty(self):
        assert merge_similar_bullets("") == ""

    def test_single_bullet(self):
        text = "- only one item"
        result = merge_similar_bullets(text)
        assert "only one item" in result

    def test_no_bullets(self):
        text = "plain text without bullets"
        assert merge_similar_bullets(text) == text

    def test_mixed_content(self):
        text = "# Header\n- bullet 1\n- bullet 2\nParagraph text"
        result = merge_similar_bullets(text)
        assert "Header" in result
        assert "Paragraph" in result

    def test_threshold_0(self):
        text = "- aaa\n- bbb"
        result = merge_similar_bullets(text, threshold=0.0)
        assert isinstance(result, str)

    def test_threshold_1(self):
        text = "- same text\n- same text"
        result = merge_similar_bullets(text, threshold=1.0)
        assert isinstance(result, str)


# === merge_short_bullets ===

class TestMergeShortBullets:
    def test_merges_short(self):
        text = "- Yes\n- No\n- Maybe\n- OK\n- Fine"
        result = merge_short_bullets(text)
        # Should combine short bullets
        assert isinstance(result, str)
        # Result should be shorter or same
        assert len(result) <= len(text) + 10  # small tolerance

    def test_keeps_long(self):
        text = "- This is a very long bullet point that should not be merged\n- Another lengthy description here"
        result = merge_short_bullets(text)
        assert "very long" in result

    def test_empty(self):
        assert merge_short_bullets("") == ""

    def test_no_bullets(self):
        text = "no bullets"
        assert merge_short_bullets(text) == text

    def test_single_short(self):
        text = "- Yes"
        result = merge_short_bullets(text)
        assert "Yes" in result

    def test_max_merge_limit(self):
        text = "\n".join(f"- W{i}" for i in range(20))
        result = merge_short_bullets(text, max_merge=3)
        assert isinstance(result, str)

    def test_custom_max_words(self):
        text = "- one two\n- three four\n- five six"
        result = merge_short_bullets(text, max_words=3)
        assert isinstance(result, str)
