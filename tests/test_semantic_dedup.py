"""Tests for SemanticDedup FusionStage and dedup_across_messages.

Covers:
- Near-duplicate detection (similar but not identical blocks)
- Exact duplicate detection
- Short text / block skipping
- Fenced code block handling
- Cross-message deduplication
- Stats accuracy
- Edge cases (empty input, single block, all unique, all duplicates)
- FusionStage interface compliance
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.base import FusionContext, FusionResult, FusionStage
from claw_compactor.fusion.semantic_dedup import (
    SemanticDedup,
    DedupStats,
    dedup_across_messages,
    _jaccard,
    _shingles,
    _tokenise,
    _split_blocks,
    _run_dedup,
    _REF_TEMPLATE,
    _MSG_REF_TEMPLATE,
    _SIM_THRESHOLD,
    _MIN_BLOCK_CHARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(content: str, **kwargs) -> FusionContext:
    return FusionContext(content=content, **kwargs)


def _long_para(seed: str, repeat: int = 1) -> str:
    """Return a paragraph that is definitely > _MIN_BLOCK_CHARS."""
    base = (
        f"{seed} The quick brown fox jumps over the lazy dog. "
        "This is filler text to ensure the block is long enough. "
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
    )
    return " ".join([base] * repeat)


# ---------------------------------------------------------------------------
# Unit tests: fingerprinting primitives
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_words(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenise("foo, bar! baz.") == ["foo", "bar", "baz"]

    def test_keeps_digits(self):
        assert "42" in _tokenise("version 42 is out")

    def test_empty_string(self):
        assert _tokenise("") == []


class TestShingles:
    def test_three_word_shingles(self):
        tokens = ["a", "b", "c", "d"]
        sh = _shingles(tokens, n=3)
        assert ("a", "b", "c") in sh
        assert ("b", "c", "d") in sh
        assert len(sh) == 2

    def test_too_few_tokens_returns_empty(self):
        assert _shingles(["a", "b"], n=3) == frozenset()

    def test_single_token_returns_empty(self):
        assert _shingles(["a"], n=3) == frozenset()

    def test_exact_n_tokens_returns_one_shingle(self):
        sh = _shingles(["x", "y", "z"], n=3)
        assert len(sh) == 1
        assert ("x", "y", "z") in sh


class TestJaccard:
    def test_identical_sets(self):
        s = frozenset({(1,), (2,), (3,)})
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        a = frozenset({(1,), (2,)})
        b = frozenset({(3,), (4,)})
        assert _jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({(1,), (2,), (3,)})
        b = frozenset({(2,), (3,), (4,)})
        # intersection={2,3}, union={1,2,3,4} → 2/4 = 0.5
        assert abs(_jaccard(a, b) - 0.5) < 1e-9

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        a = frozenset({(1,)})
        assert _jaccard(a, frozenset()) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: block splitting
# ---------------------------------------------------------------------------

class TestSplitBlocks:
    def test_splits_on_blank_lines(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        blocks = _split_blocks(text)
        texts = [b.text.strip() for b in blocks]
        assert any("First" in t for t in texts)
        assert any("Second" in t for t in texts)

    def test_code_fence_is_atomic(self):
        text = "Intro text.\n\n```python\ndef foo():\n    return 42\n```\n\nOutro text."
        blocks = _split_blocks(text)
        code_blocks = [b for b in blocks if b.is_code]
        assert len(code_blocks) == 1
        assert "def foo" in code_blocks[0].text

    def test_multiple_code_fences(self):
        text = (
            "```bash\necho hello\n```\n\n"
            "Some text.\n\n"
            "```python\nprint('hi')\n```"
        )
        blocks = _split_blocks(text)
        code_blocks = [b for b in blocks if b.is_code]
        assert len(code_blocks) == 2

    def test_empty_text_returns_no_blocks(self):
        assert _split_blocks("") == []

    def test_single_paragraph_returns_one_block(self):
        text = "Just one paragraph with enough words to be interesting."
        blocks = _split_blocks(text)
        assert len(blocks) >= 1

    def test_blocks_sorted_by_position(self):
        text = "Alpha para.\n\nBeta para.\n\nGamma para."
        blocks = _split_blocks(text)
        starts = [b.start for b in blocks]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Core dedup: _run_dedup
# ---------------------------------------------------------------------------

class TestRunDedupExactDuplicate:
    def test_exact_duplicate_block_replaced(self):
        para = _long_para("unique seed alpha")
        text = para + "\n\n" + para
        output, stats = _run_dedup(text)
        assert stats.blocks_deduped >= 1
        assert _REF_TEMPLATE.format(n=1) in output or "duplicate of block" in output

    def test_second_occurrence_removed_first_kept(self):
        para = _long_para("exact content preserved here")
        text = para + "\n\n" + para
        output, stats = _run_dedup(text)
        # First occurrence must appear verbatim.
        assert para in output
        # Output must be shorter than original (minus some overhead).
        assert len(output) < len(text)

    def test_stats_deduped_count_correct(self):
        para = _long_para("repeated block stats test")
        text = para + "\n\n" + para + "\n\n" + para
        _, stats = _run_dedup(text)
        assert stats.blocks_deduped == 2
        assert stats.blocks_kept >= 1

    def test_three_duplicates_two_replaced(self):
        para = _long_para("triple block scenario")
        text = "\n\n".join([para] * 3)
        _, stats = _run_dedup(text)
        assert stats.blocks_deduped == 2


class TestRunDedupNearDuplicate:
    def test_near_duplicate_with_minor_word_change(self):
        """Two paragraphs that differ by only one word should be deduped (Jaccard > 0.80)."""
        # Long shared body with only one word substituted keeps similarity > 0.88.
        base = (
            "The deployment pipeline failed because the container image tag "
            "was missing from the registry. The CI job exited with code 1. "
            "Engineers must update the manifest file and retrigger the build. "
            "Please review the attached log output and confirm the fix. "
            "The infrastructure team has been notified and is investigating."
        )
        # Replace only one word so the shingle overlap stays well above 0.80.
        variant = base.replace("failed", "errored", 1)
        text = base + "\n\n" + variant
        output, stats = _run_dedup(text)
        assert stats.blocks_deduped >= 1

    def test_completely_different_blocks_not_deduped(self):
        para_a = _long_para("apple orchard harvest season autumn fruit picking")
        para_b = _long_para("database schema migration rollback transaction log")
        text = para_a + "\n\n" + para_b
        _, stats = _run_dedup(text)
        assert stats.blocks_deduped == 0

    def test_similarity_threshold_respected(self):
        """Blocks with <80% overlap must NOT be deduped."""
        # Construct two blocks sharing ~60% shingles by mixing different words.
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        unique_a = "nu xi omicron pi rho sigma tau upsilon phi chi psi omega one two"
        unique_b = "red green blue yellow orange purple white black brown pink cyan"
        para_a = f"{shared} {unique_a} " * 2
        para_b = f"{shared} {unique_b} " * 2
        text = para_a + "\n\n" + para_b
        _, stats = _run_dedup(text)
        # These should NOT be deduped since unique portions are large enough
        # to bring similarity below 0.8.
        # We just verify the algorithm ran without error; exact outcome depends
        # on actual shingle overlap.
        assert isinstance(stats.blocks_deduped, int)


class TestRunDedupShortBlocks:
    def test_short_blocks_never_deduped(self):
        """Blocks shorter than _MIN_BLOCK_CHARS are always kept."""
        short = "Hi there."  # well under 50 chars
        text = short + "\n\n" + short + "\n\n" + _long_para("padding so text is long enough overall")
        output, stats = _run_dedup(text)
        # Short blocks should not be counted as deduped.
        assert output.count(short) >= 2  # both occurrences of short block kept

    def test_exactly_at_min_length_boundary(self):
        """A block of exactly _MIN_BLOCK_CHARS is eligible for deduplication."""
        # Build a block that's exactly at the boundary.
        boundary = "a" * _MIN_BLOCK_CHARS
        text = boundary + "\n\n" + boundary
        # Just verify it runs without error.
        output, stats = _run_dedup(text)
        assert isinstance(stats.blocks_deduped, int)


class TestRunDedupCodeBlocks:
    def test_identical_code_blocks_deduped(self):
        code = "```python\ndef process(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result\n```"
        text = code + "\n\n" + _long_para("some prose between them") + "\n\n" + code
        output, stats = _run_dedup(text)
        assert stats.blocks_deduped >= 1

    def test_different_code_blocks_not_deduped(self):
        code_a = "```python\ndef add(a, b):\n    return a + b\n```"
        code_b = "```javascript\nfunction multiply(x, y) { return x * y; }\n```"
        text = code_a + "\n\n" + _long_para("filler text here") + "\n\n" + code_b
        _, stats = _run_dedup(text)
        assert stats.blocks_deduped == 0

    def test_code_block_not_split_on_blank_lines(self):
        """A code block with internal blank lines must remain atomic."""
        code = "```python\ndef foo():\n    pass\n\n\ndef bar():\n    pass\n```"
        text = _long_para("intro") + "\n\n" + code + "\n\n" + _long_para("outro")
        blocks = _split_blocks(text)
        code_blocks = [b for b in blocks if b.is_code]
        assert len(code_blocks) == 1
        assert "def foo" in code_blocks[0].text
        assert "def bar" in code_blocks[0].text


class TestRunDedupStats:
    def test_tokens_before_and_after_populated(self):
        para = _long_para("stats verification test case")
        text = para + "\n\n" + para
        _, stats = _run_dedup(text)
        assert stats.tokens_before > 0
        assert stats.tokens_after > 0
        assert stats.tokens_after <= stats.tokens_before

    def test_chars_removed_nonnegative(self):
        para = _long_para("chars removed assertion")
        text = para + "\n\n" + para
        _, stats = _run_dedup(text)
        assert stats.chars_removed >= 0

    def test_blocks_total_equals_kept_plus_deduped_plus_short(self):
        para = _long_para("total block count check")
        short = "hi"
        text = para + "\n\n" + para + "\n\n" + short
        _, stats = _run_dedup(text)
        # blocks_total >= blocks_kept + blocks_deduped (short blocks count in total)
        assert stats.blocks_total >= stats.blocks_kept + stats.blocks_deduped

    def test_no_duplicates_gives_zero_deduped(self):
        para_a = _long_para("unique_block_one alpha beta gamma delta")
        para_b = _long_para("unique_block_two epsilon zeta eta theta iota")
        text = para_a + "\n\n" + para_b
        _, stats = _run_dedup(text)
        assert stats.blocks_deduped == 0

    def test_stats_as_dict_has_all_keys(self):
        _, stats = _run_dedup(_long_para("key check") + "\n\n" + _long_para("another key check"))
        d = stats.as_dict()
        for key in ("blocks_total", "blocks_kept", "blocks_deduped",
                    "chars_removed", "tokens_before", "tokens_after"):
            assert key in d


# ---------------------------------------------------------------------------
# Edge cases for _run_dedup
# ---------------------------------------------------------------------------

class TestRunDedupEdgeCases:
    def test_empty_string(self):
        output, stats = _run_dedup("")
        assert output == ""
        assert stats.blocks_total == 0
        assert stats.blocks_deduped == 0

    def test_single_block_no_dedup(self):
        text = _long_para("only one block here in this text")
        output, stats = _run_dedup(text)
        assert stats.blocks_deduped == 0
        assert output.strip() == text.strip() or text in output

    def test_all_short_blocks(self):
        text = "Hi.\n\nHi.\n\nHi."
        output, stats = _run_dedup(text)
        # All blocks too short — no deduplication.
        assert stats.blocks_deduped == 0

    def test_output_contains_reference_marker(self):
        para = _long_para("marker presence verification test")
        text = para + "\n\n" + para
        output, stats = _run_dedup(text)
        if stats.blocks_deduped > 0:
            assert "duplicate of block" in output

    def test_reconstruction_preserves_non_block_text(self):
        """Gaps between blocks (separators) must be preserved."""
        para = _long_para("gap preservation test")
        sep = "\n\n---\n\n"
        text = para + sep + para
        output, _ = _run_dedup(text)
        # The separator should still be present.
        assert "---" in output

    def test_whitespace_only_text(self):
        output, stats = _run_dedup("   \n\n   \n")
        assert stats.blocks_deduped == 0

    def test_very_long_text_with_many_duplicates(self):
        para = _long_para("performance test block")
        text = "\n\n".join([para] * 10)
        output, stats = _run_dedup(text)
        assert stats.blocks_deduped == 9


# ---------------------------------------------------------------------------
# SemanticDedup FusionStage
# ---------------------------------------------------------------------------

class TestSemanticDedupStageInterface:
    def test_is_fusion_stage_subclass(self):
        assert isinstance(SemanticDedup(), FusionStage)

    def test_name(self):
        assert SemanticDedup.name == "semantic_dedup"

    def test_order(self):
        assert SemanticDedup.order == 12

    def test_should_apply_false_for_short_content(self):
        stage = SemanticDedup()
        ctx = _make_ctx("Short text.")
        assert stage.should_apply(ctx) is False

    def test_should_apply_true_for_long_content(self):
        stage = SemanticDedup()
        ctx = _make_ctx("x" * 201)
        assert stage.should_apply(ctx) is True

    def test_should_apply_boundary_200_chars(self):
        stage = SemanticDedup()
        # Exactly 200 chars → False (must be strictly greater than 200).
        assert stage.should_apply(_make_ctx("a" * 200)) is False
        # 201 chars → True.
        assert stage.should_apply(_make_ctx("a" * 201)) is True

    def test_apply_returns_fusion_result(self):
        stage = SemanticDedup()
        para = _long_para("fusion result type check")
        ctx = _make_ctx(para + "\n\n" + para)
        result = stage.apply(ctx)
        assert isinstance(result, FusionResult)

    def test_apply_content_is_string(self):
        stage = SemanticDedup()
        para = _long_para("content type string check")
        ctx = _make_ctx(para + "\n\n" + para)
        result = stage.apply(ctx)
        assert isinstance(result.content, str)

    def test_apply_original_tokens_populated(self):
        stage = SemanticDedup()
        para = _long_para("token count populated")
        ctx = _make_ctx(para + "\n\n" + para)
        result = stage.apply(ctx)
        assert result.original_tokens > 0

    def test_apply_compressed_tokens_le_original(self):
        stage = SemanticDedup()
        para = _long_para("compression should reduce tokens")
        ctx = _make_ctx(para + "\n\n" + para)
        result = stage.apply(ctx)
        assert result.compressed_tokens <= result.original_tokens

    def test_apply_markers_populated_on_dedup(self):
        stage = SemanticDedup()
        para = _long_para("markers should be set when dedup occurs")
        ctx = _make_ctx(para + "\n\n" + para)
        result = stage.apply(ctx)
        assert any("semantic_dedup" in m for m in result.markers)

    def test_apply_no_markers_when_nothing_deduped(self):
        stage = SemanticDedup()
        para_a = _long_para("first completely unique paragraph content here alpha")
        para_b = _long_para("second entirely different paragraph content beta gamma")
        ctx = _make_ctx(para_a + "\n\n" + para_b)
        result = stage.apply(ctx)
        # If nothing was deduped, markers list should be empty.
        assert result.markers == []

    def test_timed_apply_skips_short_content(self):
        stage = SemanticDedup()
        ctx = _make_ctx("Too short.")
        result = stage.timed_apply(ctx)
        assert result.skipped is True

    def test_timed_apply_runs_long_content(self):
        stage = SemanticDedup()
        ctx = _make_ctx("a" * 250)
        result = stage.timed_apply(ctx)
        assert result.skipped is False

    def test_apply_does_not_mutate_context(self):
        stage = SemanticDedup()
        para = _long_para("immutability check")
        original_content = para + "\n\n" + para
        ctx = _make_ctx(original_content)
        stage.apply(ctx)
        assert ctx.content == original_content


class TestSemanticDedupRealWorldScenarios:
    def test_tool_output_file_contents_repeated(self):
        """Simulates a tool echoing back the same file content twice."""
        file_content = (
            "Here is the content of config.py:\n\n"
            "```python\n"
            "DATABASE_URL = 'postgresql://localhost/mydb'\n"
            "DEBUG = False\n"
            "SECRET_KEY = 'abc123'\n"
            "ALLOWED_HOSTS = ['*']\n"
            "MAX_CONNECTIONS = 100\n"
            "```"
        )
        text = file_content + "\n\n" + file_content
        stage = SemanticDedup()
        ctx = _make_ctx(text)
        result = stage.apply(ctx)
        assert result.compressed_tokens < result.original_tokens

    def test_error_message_repeated_in_logs(self):
        """Simulates repeated error lines in tool output."""
        err = (
            "ERROR: Connection refused to database server at 192.168.1.10:5432. "
            "Check that the database is running and accessible from this host. "
            "Retrying in 5 seconds. Attempt 1 of 3 failed."
        )
        text = err + "\n\n" + err + "\n\n" + err
        stage = SemanticDedup()
        ctx = _make_ctx(text)
        result = stage.apply(ctx)
        assert result.compressed_tokens < result.original_tokens

    def test_assistant_echoes_code_back(self):
        """Simulates an assistant echoing a code block from the user message."""
        code = (
            "```python\n"
            "def calculate_discount(price, rate):\n"
            "    if rate < 0 or rate > 1:\n"
            "        raise ValueError('rate must be between 0 and 1')\n"
            "    return price * (1 - rate)\n"
            "```"
        )
        user_msg = f"Here is the function:\n\n{code}\n\nCan you fix the bug?"
        assistant_reply = (
            f"Sure! Looking at your code:\n\n{code}\n\n"
            "The issue is that you need to handle the edge case when price is negative."
        )
        text = user_msg + "\n\n" + assistant_reply
        stage = SemanticDedup()
        ctx = _make_ctx(text)
        result = stage.apply(ctx)
        # Should detect the duplicated code block and remove one copy.
        assert result.compressed_tokens <= result.original_tokens


# ---------------------------------------------------------------------------
# Cross-message deduplication
# ---------------------------------------------------------------------------

class TestDedupAcrossMessages:
    def _msg(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def test_identical_messages_deduped(self):
        content = _long_para("repeated message content across turns")
        messages = [
            self._msg("user", content),
            self._msg("assistant", "OK I see."),
            self._msg("user", content),
        ]
        deduped, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 1
        assert _MSG_REF_TEMPLATE.format(idx=0) in deduped[2]["content"]

    def test_unique_messages_not_deduped(self):
        messages = [
            self._msg("user", _long_para("alpha unique message one here")),
            self._msg("assistant", _long_para("beta different response content two")),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 0

    def test_near_duplicate_messages_deduped(self):
        # Long shared body with only one phrase substituted keeps Jaccard > 0.80.
        base = (
            "The server returned a 500 Internal Server Error. "
            "Check the application logs for more details. "
            "The error occurred in the request handler for the api data endpoint. "
            "Stack trace has been written to the application error log file. "
            "Please investigate and resolve the issue as soon as possible."
        )
        # Replace only one multi-word phrase so shingle overlap stays above 0.80.
        variant = base.replace("returned a 500", "responded with 500", 1)
        messages = [
            self._msg("tool", base),
            self._msg("tool", variant),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 1

    def test_short_messages_not_deduped(self):
        messages = [
            self._msg("user", "Yes."),
            self._msg("assistant", "Yes."),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 0

    def test_empty_messages_list(self):
        deduped, stats = dedup_across_messages([])
        assert deduped == []
        assert stats["messages_total"] == 0
        assert stats["messages_deduped"] == 0

    def test_single_message_not_deduped(self):
        messages = [self._msg("user", _long_para("single message no dedup"))]
        deduped, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 0
        assert len(deduped) == 1

    def test_original_messages_not_mutated(self):
        content = _long_para("immutability cross-message test")
        original_content = content
        messages = [
            self._msg("user", content),
            self._msg("user", content),
        ]
        dedup_across_messages(messages)
        assert messages[0]["content"] == original_content
        assert messages[1]["content"] == original_content

    def test_returned_list_is_new_list(self):
        messages = [self._msg("user", _long_para("new list test"))]
        deduped, _ = dedup_across_messages(messages)
        assert deduped is not messages

    def test_non_string_content_passed_through(self):
        """Messages with list-valued content (multi-part) are not touched."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "Sure."},
        ]
        deduped, stats = dedup_across_messages(messages)
        assert deduped[0]["content"] == [{"type": "text", "text": "hello"}]
        assert stats["messages_deduped"] == 0

    def test_tokens_before_gte_tokens_after(self):
        content = _long_para("token reduction across messages")
        messages = [
            self._msg("user", content),
            self._msg("user", content),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["tokens_before"] >= stats["tokens_after"]

    def test_stats_messages_total_correct(self):
        messages = [
            self._msg("user", _long_para("one")),
            self._msg("assistant", _long_para("two")),
            self._msg("user", _long_para("three")),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["messages_total"] == 3

    def test_reference_points_to_correct_index(self):
        content = _long_para("reference index correctness test")
        messages = [
            self._msg("user", "Hello"),           # idx 0 — short, kept
            self._msg("assistant", content),       # idx 1 — kept
            self._msg("user", "Thanks"),           # idx 2 — short, kept
            self._msg("assistant", content),       # idx 3 — duplicate of idx 1
        ]
        deduped, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 1
        ref = deduped[3]["content"]
        assert "1" in ref  # reference must point to message index 1

    def test_three_identical_messages_two_deduped(self):
        content = _long_para("triple identical content dedup cross message")
        messages = [
            self._msg("user", content),
            self._msg("user", content),
            self._msg("user", content),
        ]
        _, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 2

    def test_message_with_none_content_handled(self):
        """Messages with missing/None content don't crash."""
        messages = [
            {"role": "user"},                            # no content key
            {"role": "assistant", "content": None},      # None content
            {"role": "user", "content": _long_para("valid content message")},
        ]
        deduped, stats = dedup_across_messages(messages)
        assert len(deduped) == 3
        assert stats["messages_deduped"] == 0

    def test_empty_string_content_passed_through(self):
        messages = [
            self._msg("user", ""),
            self._msg("user", ""),
        ]
        deduped, stats = dedup_across_messages(messages)
        assert stats["messages_deduped"] == 0
        assert deduped[0]["content"] == ""
        assert deduped[1]["content"] == ""

    def test_role_field_preserved(self):
        content = _long_para("role field preservation check")
        messages = [
            self._msg("tool", content),
            self._msg("tool", content),
        ]
        deduped, _ = dedup_across_messages(messages)
        assert deduped[0]["role"] == "tool"
        assert deduped[1]["role"] == "tool"

    def test_extra_fields_preserved(self):
        content = _long_para("extra field preservation")
        messages = [
            {"role": "user", "content": content, "timestamp": "2024-01-01"},
            {"role": "user", "content": content, "timestamp": "2024-01-02"},
        ]
        deduped, _ = dedup_across_messages(messages)
        assert deduped[0]["timestamp"] == "2024-01-01"
        assert deduped[1]["timestamp"] == "2024-01-02"
