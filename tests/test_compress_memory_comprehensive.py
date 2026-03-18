"""Comprehensive tests for compress_memory.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compress_memory import (
    rule_compress, compress_file, generate_llm_prompt,
    _collect_files, _file_age_days,
)
from claw_compactor.tokens import estimate_tokens


class TestRuleCompress:
    def test_basic(self):
        text = "# Title\n\nSome text\n\n## Section\n\nMore text"
        result = rule_compress(text)
        assert isinstance(result, str)

    def test_empty(self):
        assert rule_compress("") == ""

    def test_duplicate_lines(self):
        text = "- item one\n- item one\n- item two"
        result = rule_compress(text)
        assert result.count("item one") == 1

    def test_emoji_stripping(self):
        text = "🎉 Celebration! 🎊 Party!"
        result = rule_compress(text)
        assert "🎉" not in result

    def test_emoji_stripping_disabled(self):
        text = "🎉 Celebration!"
        result = rule_compress(text, enable_emoji_strip=False)
        assert "Celebration" in result

    def test_chinese_punctuation(self):
        text = "测试，内容。重要！"
        result = rule_compress(text)
        assert "，" not in result

    def test_empty_section_removal(self):
        text = "# Has content\nfoo\n# Empty\n\n# Also content\nbar"
        result = rule_compress(text)
        assert "foo" in result and "bar" in result

    def test_table_compression(self):
        text = "| Key | Value |\n|-----|-------|\n| a | 1 |\n| b | 2 |"
        result = rule_compress(text)
        assert "a" in result and "1" in result

    def test_never_loses_data(self):
        """All key data should survive compression."""
        text = "# Config\nServer: 192.168.1.100\nPort: 8080\nUser: admin\nDate: 2026-01-01"
        result = rule_compress(text)
        assert "192.168.1.100" in result
        assert "8080" in result
        assert "admin" in result
        assert "2026-01-01" in result

    def test_idempotent(self):
        text = "# Title\n- item 1\n- item 2\n- item 3"
        first = rule_compress(text)
        second = rule_compress(first)
        assert first == second

    def test_large_text(self):
        text = "# Header\n" + "\n".join(f"- Item {i}: value {i}" for i in range(1000))
        result = rule_compress(text)
        assert len(result) <= len(text)

    def test_mixed_language(self):
        text = "# 标题 Title\n- English item\n- 中文项目\n- 日本語アイテム"
        result = rule_compress(text)
        assert "English" in result
        assert "中文" in result

    def test_bullet_merging(self):
        text = "- Updated module A config\n- Updated module B config\n- Updated module C config"
        result = rule_compress(text)
        assert isinstance(result, str)

    def test_short_bullet_merging(self):
        text = "- Yes\n- No\n- OK\n- Done\n- Fine"
        result = rule_compress(text)
        assert isinstance(result, str)

    def test_savings_positive(self):
        """Rule compression should not increase token count."""
        text = (
            "# Configuration Notes\n\n"
            "| Setting | Value | Description |\n"
            "|---------|-------|-------------|\n"
            "| timeout | 30 | Request timeout |\n"
            "| retries | 3 | Max retries |\n\n"
            "## Empty Section\n\n"
            "## Notes\n"
            "- Hello world 🎉\n"
            "- Hello world 🎉\n"
            "- Testing，内容。\n"
        )
        original_tokens = estimate_tokens(text)
        compressed = rule_compress(text)
        compressed_tokens = estimate_tokens(compressed)
        assert compressed_tokens <= original_tokens


class TestCompressFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Test\n\nContent here\n\n## Section\n\nMore content\n")
        result = compress_file(f, dry_run=True, no_llm=True)
        assert "original_tokens" in result
        assert "rule_compressed_tokens" in result

    def test_dry_run_no_change(self, tmp_path):
        f = tmp_path / "test.md"
        original = "# Test\nHello world\nHello world\n"
        f.write_text(original)
        compress_file(f, dry_run=True, no_llm=True)
        assert f.read_text() == original

    def test_writes_when_not_dry_run(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Test\n\nHello world 🎉\nHello world 🎉\n\n## Empty\n\n")
        compress_file(f, dry_run=False, no_llm=True)
        result = f.read_text()
        assert result.count("Hello world") >= 1

    def test_output_file(self, tmp_path):
        f = tmp_path / "input.md"
        f.write_text("# Test\nContent\n")
        out = str(tmp_path / "output.md")
        compress_file(f, output=out, no_llm=True)
        assert Path(out).exists()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        result = compress_file(f, dry_run=True, no_llm=True)
        assert isinstance(result, dict)


class TestGenerateLlmPrompt:
    def test_basic(self):
        prompt = generate_llm_prompt("Some content to compress", target_pct=50)
        assert isinstance(prompt, str)
        assert "50" in prompt
        assert "compress" in prompt.lower()

    def test_includes_content(self):
        prompt = generate_llm_prompt("UNIQUE_STRING_12345")
        assert "UNIQUE_STRING_12345" in prompt

    def test_custom_target(self):
        prompt = generate_llm_prompt("text", target_pct=30)
        assert "30" in prompt


class TestCollectFiles:
    def test_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("hello")
        (tmp_path / "b.md").write_text("world")
        files = _collect_files(str(tmp_path))
        assert len(files) >= 2

    def test_single_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("content")
        files = _collect_files(str(f))
        assert len(files) == 1

    def test_nonexistent(self):
        from claw_compactor.exceptions import FileNotFoundError_
        with pytest.raises(FileNotFoundError_):
            _collect_files("/nonexistent/xyz")

    def test_older_than(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("content")
        # File was just created, so older_than=1 should exclude it
        files = _collect_files(str(tmp_path), older_than=1)
        assert len(files) == 0

    def test_skips_non_md(self, tmp_path):
        (tmp_path / "test.md").write_text("md")
        (tmp_path / "test.txt").write_text("txt")
        (tmp_path / "test.py").write_text("py")
        files = _collect_files(str(tmp_path))
        assert all(str(f).endswith(".md") for f in files)
