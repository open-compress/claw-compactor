"""Tests for lib.tokens module."""

from claw_compactor.tokens import estimate_tokens, using_tiktoken


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short(self) -> None:
        assert estimate_tokens("abc") >= 1

    def test_known_length(self) -> None:
        result = estimate_tokens("a" * 400)
        assert 50 <= result <= 200

    def test_large(self) -> None:
        assert estimate_tokens("x" * 400_000) >= 50_000

    def test_unicode_chinese(self) -> None:
        assert estimate_tokens("你好世界测试" * 100) > 0

    def test_mixed(self) -> None:
        assert estimate_tokens("Hello 你好 `code` **bold**") > 0

    def test_using_tiktoken_bool(self) -> None:
        assert isinstance(using_tiktoken(), bool)
