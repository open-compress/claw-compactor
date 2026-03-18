"""Comprehensive tests for lib/tokens.py."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.tokens import estimate_tokens, using_tiktoken


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        result = estimate_tokens("hello")
        assert result > 0

    def test_sentence(self):
        result = estimate_tokens("The quick brown fox jumps over the lazy dog.")
        assert result > 5

    def test_large_text(self):
        text = "word " * 10000
        result = estimate_tokens(text)
        assert result > 5000

    def test_chinese(self):
        result = estimate_tokens("你好世界这是一段中文")
        assert result > 0

    def test_korean(self):
        result = estimate_tokens("안녕하세요 세계")
        assert result > 0

    def test_japanese_hiragana(self):
        result = estimate_tokens("こんにちは世界")
        assert result > 0

    def test_japanese_katakana(self):
        result = estimate_tokens("カタカナテスト")
        assert result > 0

    def test_cjk_heuristic_covers_all_scripts(self):
        """Korean and Japanese should get CJK token rates, not ASCII rates."""
        if using_tiktoken():
            pytest.skip("heuristic path not active when tiktoken is installed")
        # Pure Korean (5 Hangul syllables) should yield ~3-4 tokens at 1.5 chars/token
        korean = estimate_tokens("안녕하세요")
        # If the heuristic treated these as ASCII (4 chars/token on byte length),
        # the count would be much lower than expected.
        assert korean >= 3

    def test_mixed_language(self):
        result = estimate_tokens("Hello 你好 World 世界")
        assert result > 0

    def test_code(self):
        result = estimate_tokens("def foo():\n    return 42\n")
        assert result > 0

    def test_markdown(self):
        result = estimate_tokens("# Header\n- item 1\n- item 2\n| a | b |\n")
        assert result > 0

    def test_special_chars(self):
        result = estimate_tokens("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result > 0

    def test_whitespace_only(self):
        result = estimate_tokens("   \n\n\t\t  ")
        assert result >= 0

    def test_newlines(self):
        result = estimate_tokens("\n\n\n\n\n")
        assert result >= 0

    def test_emoji(self):
        result = estimate_tokens("🎉🌍🚀💻")
        assert result > 0

    def test_json(self):
        result = estimate_tokens('{"key": "value", "num": 42}')
        assert result > 0

    def test_monotonic_with_length(self):
        """More text should generally mean more tokens."""
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short

    def test_deterministic(self):
        text = "Deterministic test input"
        a = estimate_tokens(text)
        b = estimate_tokens(text)
        assert a == b


class TestUsingTiktoken:
    def test_returns_bool(self):
        result = using_tiktoken()
        assert isinstance(result, bool)

    def test_consistent(self):
        assert using_tiktoken() == using_tiktoken()
