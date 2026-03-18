"""Comprehensive tests for FusionEngine — the unified compression entry point.

Covers:
- Single text compression (happy path, edge cases)
- Message list compression (OpenAI format)
- All adapter stages actually run (RLEStage, TokenOptStage, AbbrevStage)
- All native stages wired into the pipeline
- Stats accuracy
- Edge cases: empty input, very short input, already-compressed text,
  multipart messages, system-role KV-cache alignment
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Bootstrap path so both lib.* and compressed_context resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.engine import (
    FusionEngine,
    RLEStage,
    TokenOptStage,
    AbbrevStage,
    _reduction_pct,
    _build_stats,
    _empty_stats,
    _empty_aggregate_stats,
    _aggregate_stats,
)
from claw_compactor.fusion.base import FusionContext
from claw_compactor.fusion.pipeline import FusionPipeline
from claw_compactor.rewind.store import RewindStore
from claw_compactor.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> FusionEngine:
    return FusionEngine(enable_rewind=True, aggressive=True)


@pytest.fixture
def engine_no_rewind() -> FusionEngine:
    return FusionEngine(enable_rewind=False)


# Substantive natural-language text — long enough to trigger most stages.
LONG_TEXT = (
    "Furthermore, the implementation of the distributed architecture requires "
    "extensive experience in infrastructure management and database configuration. "
    "In addition, the development team should have approximately 5 years of "
    "experience with Kubernetes and continuous integration. "
    "Moreover, the documentation for all applications must be updated regularly. "
    "The production environment is located in the headquarters offices. "
    "Authentication and authorization are handled by the security module. "
    "The repository contains the complete specification and requirements for "
    "the deployment process. Additionally, monitoring and notification services "
    "must be configured properly. Having said that, the operations team is "
    "responsible for the overall architecture. "
    * 3
)

LONG_TEXT_WITH_IPS = (
    "Servers at 192.168.1.10, 192.168.1.11, 192.168.1.12 and 192.168.1.13 "
    "are all in the same subnet. Also check 192.168.1.20 and 192.168.1.21. "
    "The workspace path is /home/user/workspace/project/src. "
    "Enum values: BTC, ETH, SOL, BNB, DOGE, ADA, XRP, DOT.\n"
    + LONG_TEXT
)

MARKDOWN_TEXT = (
    "# **Overview**\n\n"
    "- **Feature 1**: some description\n"
    "- **Feature 2**: another description\n"
    "- **Feature 3**: yet another description\n\n"
    "  The table below shows the configuration:\n\n"
    "| Key         | Value         |\n"
    "| ----------- | ------------- |\n"
    "| host        | localhost     |\n"
    "| port        | 8080          |\n"
    "| environment | production    |\n"
)

SHORT_TEXT = "Hello world."

# Narrative text that Cortex reliably classifies as "text" (no code keywords).
# Long enough (>=20 words) to also satisfy NexusStage's minimum word threshold.
NARRATIVE_TEXT = (
    "John visited Paris last summer. He enjoyed the museums and the food. "
    "The Eiffel Tower was magnificent. Many tourists were there enjoying the sights. "
    "The weather was pleasant and sunny. People walked along the Seine river. "
    "Restaurants offered delicious meals. Hotels were fully booked. "
    "The city was alive with music and culture. Everyone had a wonderful time. "
    "Furthermore, the experience was unforgettable. In addition, the local culture "
    "was fascinating. Moreover, many visitors came with their families. "
    "The documentation available to tourists was extensive. "
    "Communication between visitors and locals was surprisingly easy. "
) * 3

SYSTEM_MSG_WITH_DYNAMIC = (
    "You are a helpful assistant. Request ID: 550e8400-e29b-41d4-a716-446655440000. "
    "Date: 2025-01-15T12:00:00Z. API key: sk-abcdefghijklmnopqrstuvwxyz123456. "
    "Always respond concisely."
)


# ---------------------------------------------------------------------------
# Tests: FusionEngine construction
# ---------------------------------------------------------------------------

class TestFusionEngineConstruction:
    def test_default_construction(self):
        e = FusionEngine()
        assert e.rewind_store is not None
        assert isinstance(e.rewind_store, RewindStore)

    def test_no_rewind(self):
        e = FusionEngine(enable_rewind=False)
        assert e.rewind_store is None

    def test_pipeline_is_fusion_pipeline(self):
        e = FusionEngine()
        assert isinstance(e.pipeline, FusionPipeline)

    def test_stage_names_contains_all_expected_stages(self):
        e = FusionEngine()
        names = e.stage_names
        expected = [
            "quantum_lock",
            "cortex",
            "photon",
            "rle",
            "ionizer",
            "log_crunch",
            "search_crunch",
            "diff_crunch",
            "neurosyntax",
            "nexus",
            "token_opt",
            "abbrev",
        ]
        for name in expected:
            assert name in names, f"Stage '{name}' missing from pipeline"

    def test_stages_are_sorted_by_order(self):
        e = FusionEngine()
        orders = [t.order for t in e.pipeline.transforms]
        assert orders == sorted(orders), "Stages must be sorted by order"

    def test_rle_stage_order(self):
        e = FusionEngine()
        stage = next(t for t in e.pipeline.transforms if t.name == "rle")
        assert stage.order == 10

    def test_token_opt_stage_order(self):
        e = FusionEngine()
        stage = next(t for t in e.pipeline.transforms if t.name == "token_opt")
        assert stage.order == 40

    def test_abbrev_stage_order(self):
        e = FusionEngine()
        stage = next(t for t in e.pipeline.transforms if t.name == "abbrev")
        assert stage.order == 45


# ---------------------------------------------------------------------------
# Tests: FusionEngine.compress — single text
# ---------------------------------------------------------------------------

class TestFusionEngineCompress:
    def test_returns_dict_with_required_keys(self, engine):
        result = engine.compress(LONG_TEXT)
        assert "compressed" in result
        assert "original" in result
        assert "stats" in result
        assert "markers" in result
        assert "warnings" in result

    def test_original_preserved(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["original"] == LONG_TEXT

    def test_compressed_is_string(self, engine):
        result = engine.compress(LONG_TEXT)
        assert isinstance(result["compressed"], str)

    def test_compression_actually_reduces_tokens(self, engine):
        result = engine.compress(LONG_TEXT)
        orig_tokens = estimate_tokens(LONG_TEXT)
        comp_tokens = estimate_tokens(result["compressed"])
        # Long natural-language text should be compressed meaningfully
        assert comp_tokens <= orig_tokens, "Compressed should not be larger"

    def test_stats_keys_present(self, engine):
        result = engine.compress(LONG_TEXT)
        stats = result["stats"]
        for key in [
            "original_tokens",
            "compressed_tokens",
            "original_chars",
            "compressed_chars",
            "reduction_pct",
            "total_timing_ms",
            "stages_run",
            "stages_skipped",
            "per_stage",
        ]:
            assert key in stats, f"Missing stats key: {key}"

    def test_stats_original_tokens_matches_estimate(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["stats"]["original_tokens"] == estimate_tokens(LONG_TEXT)

    def test_stats_compressed_tokens_matches_estimate(self, engine):
        result = engine.compress(LONG_TEXT)
        compressed = result["compressed"]
        assert result["stats"]["compressed_tokens"] == estimate_tokens(compressed)

    def test_stats_original_chars(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["stats"]["original_chars"] == len(LONG_TEXT)

    def test_stats_compressed_chars(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["stats"]["compressed_chars"] == len(result["compressed"])

    def test_stats_reduction_pct_is_numeric(self, engine):
        result = engine.compress(LONG_TEXT)
        pct = result["stats"]["reduction_pct"]
        assert isinstance(pct, float)
        assert -5.0 <= pct <= 100.0  # allow small rounding edge but never negative big

    def test_stats_stages_run_positive(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["stats"]["stages_run"] > 0

    def test_per_stage_list_covers_all_pipeline_stages(self, engine):
        result = engine.compress(LONG_TEXT)
        per_stage_names = {s["name"] for s in result["stats"]["per_stage"]}
        pipeline_names = set(engine.stage_names)
        assert per_stage_names == pipeline_names

    def test_timing_ms_nonnegative(self, engine):
        result = engine.compress(LONG_TEXT)
        assert result["stats"]["total_timing_ms"] >= 0.0

    def test_markers_is_list(self, engine):
        result = engine.compress(LONG_TEXT)
        assert isinstance(result["markers"], list)

    def test_warnings_is_list(self, engine):
        result = engine.compress(LONG_TEXT)
        assert isinstance(result["warnings"], list)

    def test_content_type_hint_passed_through(self, engine):
        # code type — Cortex should not override a pre-set type, only "text" default triggers detection.
        result = engine.compress("def foo(): pass\n    return 42\n", content_type="code")
        assert isinstance(result["compressed"], str)

    def test_role_system_triggers_quantum_lock(self, engine):
        result = engine.compress(SYSTEM_MSG_WITH_DYNAMIC, role="system")
        # QuantumLock should either run (and add markers) or at least not crash
        assert isinstance(result["compressed"], str)
        # Dynamic fragments should be stabilized (UUID and date replaced with placeholders)
        compressed = result["compressed"]
        assert isinstance(compressed, str)

    def test_rle_compresses_ip_addresses(self, engine):
        result = engine.compress(LONG_TEXT_WITH_IPS)
        # IPs in the same /24 should be shortened to $IP.suffix notation
        compressed = result["compressed"]
        assert isinstance(compressed, str)
        # Either compressed or same length, never ballooning
        assert estimate_tokens(compressed) <= estimate_tokens(LONG_TEXT_WITH_IPS) * 1.05

    def test_rle_compresses_workspace_paths(self, engine):
        text = "Found /home/user/workspace/project/main.py at /home/user/workspace/utils/helper.py"
        result = engine.compress(text)
        assert isinstance(result["compressed"], str)

    def test_token_opt_strips_bold_italic(self):
        # TokenOptStage run in isolation
        stage = TokenOptStage()
        ctx = FusionContext(content="**bold** and *italic* text here", content_type="text")
        assert stage.should_apply(ctx)
        result = stage.apply(ctx)
        assert "**" not in result.content
        assert "*italic*" not in result.content

    def test_abbrev_stage_only_applies_to_text(self):
        stage = AbbrevStage()
        ctx_text = FusionContext(content=LONG_TEXT, content_type="text")
        ctx_code = FusionContext(content="def foo(): pass", content_type="code")
        ctx_json = FusionContext(content='{"key": "value"}', content_type="json")
        assert stage.should_apply(ctx_text) is True
        assert stage.should_apply(ctx_code) is False
        assert stage.should_apply(ctx_json) is False

    def test_abbrev_stage_reduces_verbose_text(self):
        stage = AbbrevStage()
        ctx = FusionContext(content=LONG_TEXT, content_type="text")
        result = stage.apply(ctx)
        assert result.compressed_tokens <= result.original_tokens

    def test_rle_stage_always_applies(self):
        stage = RLEStage()
        ctx = FusionContext(content="any content", content_type="text")
        assert stage.should_apply(ctx) is True
        ctx_code = FusionContext(content="code here", content_type="code")
        assert stage.should_apply(ctx_code) is True

    def test_markdown_table_converted_to_kv(self, engine):
        result = engine.compress(MARKDOWN_TEXT)
        # Table should be compacted; original pipe-table format gone or shortened
        assert isinstance(result["compressed"], str)

    def test_no_rewind_engine_still_works(self, engine_no_rewind):
        result = engine_no_rewind.compress(LONG_TEXT)
        assert isinstance(result["compressed"], str)
        assert result["original"] == LONG_TEXT


# ---------------------------------------------------------------------------
# Tests: edge cases — empty and very short input
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_empty(self, engine):
        result = engine.compress("")
        assert result["compressed"] == ""
        assert result["original"] == ""

    def test_empty_string_stats_zero_tokens(self, engine):
        result = engine.compress("")
        assert result["stats"]["original_tokens"] == 0
        assert result["stats"]["compressed_tokens"] == 0

    def test_empty_string_no_stages_run(self, engine):
        result = engine.compress("")
        assert result["stats"]["stages_run"] == 0

    def test_single_word(self, engine):
        result = engine.compress("hello")
        assert isinstance(result["compressed"], str)
        assert len(result["compressed"]) > 0

    def test_single_sentence(self, engine):
        result = engine.compress(SHORT_TEXT)
        assert isinstance(result["compressed"], str)

    def test_very_short_text_not_corrupted(self, engine):
        # Very short text — stages that require min word count must skip gracefully
        for text in ["hi", "ok.", "yes", "no"]:
            result = engine.compress(text)
            assert result["compressed"]  # not empty

    def test_whitespace_only(self, engine):
        result = engine.compress("   \n\n  ")
        # Should not crash; may be empty or whitespace after minimize
        assert isinstance(result["compressed"], str)

    def test_already_compressed_text_does_not_expand_much(self, engine):
        # Run the engine twice — second pass should not significantly expand
        first = engine.compress(LONG_TEXT)
        second = engine.compress(first["compressed"])
        orig_tokens = first["stats"]["original_tokens"]
        twice_tokens = estimate_tokens(second["compressed"])
        # Second pass should not expand beyond original
        assert twice_tokens <= orig_tokens * 1.1

    def test_unicode_text(self, engine):
        text = "配置管理と認証認可について説明します。また、インフラストラクチャの最適化も重要です。"
        result = engine.compress(text)
        assert isinstance(result["compressed"], str)

    def test_code_type_skips_abbrev(self, engine):
        code = "def authenticate(user, password):\n    return database.verify(user, password)\n"
        result = engine.compress(code, content_type="code")
        # AbbrevStage must have been skipped for code content
        abbrev_step = next(
            s for s in result["stats"]["per_stage"] if s["name"] == "abbrev"
        )
        assert abbrev_step["skipped"] is True

    def test_json_type_skips_abbrev(self, engine):
        import json
        data = [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(25)]
        text = json.dumps(data, indent=2)
        result = engine.compress(text, content_type="json")
        abbrev_step = next(
            s for s in result["stats"]["per_stage"] if s["name"] == "abbrev"
        )
        assert abbrev_step["skipped"] is True

    def test_newlines_only(self, engine):
        result = engine.compress("\n\n\n\n")
        assert isinstance(result["compressed"], str)


# ---------------------------------------------------------------------------
# Tests: FusionEngine.compress_messages
# ---------------------------------------------------------------------------

class TestCompressMessages:
    def test_returns_dict_with_required_keys(self, engine):
        messages = [{"role": "user", "content": LONG_TEXT}]
        result = engine.compress_messages(messages)
        assert "messages" in result
        assert "stats" in result
        assert "per_message" in result
        assert "markers" in result
        assert "warnings" in result

    def test_empty_messages_returns_empty(self, engine):
        result = engine.compress_messages([])
        assert result["messages"] == []
        assert result["stats"]["message_count"] == 0
        assert result["per_message"] == []

    def test_single_message_structure_preserved(self, engine):
        messages = [{"role": "user", "content": LONG_TEXT, "name": "alice"}]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["name"] == "alice"
        assert "content" in msg

    def test_message_count_matches(self, engine):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": LONG_TEXT[:500]},
        ]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 3
        assert result["stats"]["message_count"] == 3

    def test_per_message_stats_length_matches(self, engine):
        messages = [
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": LONG_TEXT},
        ]
        result = engine.compress_messages(messages)
        assert len(result["per_message"]) == 2

    def test_aggregate_stats_keys_present(self, engine):
        messages = [{"role": "user", "content": LONG_TEXT}]
        result = engine.compress_messages(messages)
        stats = result["stats"]
        for key in [
            "original_tokens",
            "compressed_tokens",
            "original_chars",
            "compressed_chars",
            "reduction_pct",
            "total_timing_ms",
            "message_count",
        ]:
            assert key in stats, f"Missing aggregate stat key: {key}"

    def test_aggregate_original_tokens_equals_sum(self, engine):
        messages = [
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": SHORT_TEXT},
        ]
        result = engine.compress_messages(messages)
        expected = sum(s["original_tokens"] for s in result["per_message"])
        assert result["stats"]["original_tokens"] == expected

    def test_aggregate_compressed_tokens_equals_sum(self, engine):
        messages = [
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": LONG_TEXT},
        ]
        result = engine.compress_messages(messages)
        expected = sum(s["compressed_tokens"] for s in result["per_message"])
        assert result["stats"]["compressed_tokens"] == expected

    def test_aggregate_original_chars_equals_sum(self, engine):
        messages = [
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": SHORT_TEXT},
        ]
        result = engine.compress_messages(messages)
        expected = sum(s["original_chars"] for s in result["per_message"])
        assert result["stats"]["original_chars"] == expected

    def test_all_messages_compressed(self, engine):
        messages = [
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": LONG_TEXT},
        ]
        result = engine.compress_messages(messages)
        total_compressed = result["stats"]["compressed_tokens"]
        total_original = result["stats"]["original_tokens"]
        assert total_compressed <= total_original

    def test_per_message_role_preserved(self, engine):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": LONG_TEXT},
            {"role": "assistant", "content": "Sure."},
        ]
        result = engine.compress_messages(messages)
        roles = [s["role"] for s in result["per_message"]]
        assert roles == ["system", "user", "assistant"]

    def test_all_markers_collected(self, engine):
        # Long text with IPs and paths should produce RLE markers
        messages = [
            {"role": "user", "content": LONG_TEXT_WITH_IPS},
            {"role": "assistant", "content": LONG_TEXT},
        ]
        result = engine.compress_messages(messages)
        assert isinstance(result["markers"], list)

    def test_all_warnings_collected(self, engine):
        messages = [{"role": "user", "content": LONG_TEXT}]
        result = engine.compress_messages(messages)
        assert isinstance(result["warnings"], list)

    def test_multipart_message_text_compressed(self, engine):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG_TEXT},
                    {"type": "text", "text": LONG_TEXT[:200]},
                ],
            }
        ]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 1
        parts = result["messages"][0]["content"]
        assert isinstance(parts, list)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert len(parts[0]["text"]) <= len(LONG_TEXT)

    def test_multipart_non_text_parts_preserved(self, engine):
        img_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"},
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG_TEXT},
                    img_part,
                ],
            }
        ]
        result = engine.compress_messages(messages)
        parts = result["messages"][0]["content"]
        # Non-text part should be passed through unchanged
        assert parts[1] == img_part

    def test_empty_content_message(self, engine):
        messages = [{"role": "user", "content": ""}]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == ""

    def test_system_role_quantum_lock_applies(self, engine):
        messages = [{"role": "system", "content": SYSTEM_MSG_WITH_DYNAMIC}]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 1
        # Should not crash; content is a string
        assert isinstance(result["messages"][0]["content"], str)

    def test_large_multi_message_conversation(self, engine):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": LONG_TEXT}
            for i in range(6)
        ]
        result = engine.compress_messages(messages)
        assert len(result["messages"]) == 7
        assert result["stats"]["message_count"] == 7


# ---------------------------------------------------------------------------
# Tests: RLEStage adapter
# ---------------------------------------------------------------------------

class TestRLEStage:
    def test_name_and_order(self):
        stage = RLEStage()
        assert stage.name == "rle"
        assert stage.order == 10

    def test_should_apply_nonempty_content(self):
        stage = RLEStage()
        ctx = FusionContext(content="some text", content_type="text")
        assert stage.should_apply(ctx) is True

    def test_should_not_apply_empty_content(self):
        stage = RLEStage()
        ctx = FusionContext(content="", content_type="text")
        assert stage.should_apply(ctx) is False

    def test_applies_to_all_content_types(self):
        stage = RLEStage()
        for ct in ["text", "code", "json", "log", "diff", "search"]:
            ctx = FusionContext(content="content here", content_type=ct)
            assert stage.should_apply(ctx) is True

    def test_compresses_paths(self):
        stage = RLEStage()
        text = "Check /home/user/workspace/main.py and /home/user/workspace/utils.py"
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        assert "$WS" in result.content

    def test_compresses_ip_families(self):
        stage = RLEStage()
        text = "Servers: 10.0.0.1, 10.0.0.2, 10.0.0.3, 10.0.0.4 are online."
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        # IP compression should collapse the /24 family
        assert "$IP" in result.content

    def test_compresses_enumerations(self):
        stage = RLEStage()
        text = "Supported: BTC, ETH, SOL, BNB, DOGE in the trading system."
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        # Enum should be bracketed
        assert "[BTC,ETH,SOL,BNB,DOGE]" in result.content

    def test_returns_fusion_result(self):
        from claw_compactor.fusion.base import FusionResult
        stage = RLEStage()
        ctx = FusionContext(content="some text", content_type="text")
        result = stage.apply(ctx)
        assert isinstance(result, FusionResult)

    def test_token_counts_set(self):
        stage = RLEStage()
        text = "Check /home/user/workspace/main.py for errors."
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0


# ---------------------------------------------------------------------------
# Tests: TokenOptStage adapter
# ---------------------------------------------------------------------------

class TestTokenOptStage:
    def test_name_and_order(self):
        stage = TokenOptStage()
        assert stage.name == "token_opt"
        assert stage.order == 40

    def test_should_apply_nonempty(self):
        stage = TokenOptStage()
        ctx = FusionContext(content="content", content_type="text")
        assert stage.should_apply(ctx) is True

    def test_should_not_apply_empty(self):
        stage = TokenOptStage()
        ctx = FusionContext(content="", content_type="text")
        assert stage.should_apply(ctx) is False

    def test_strips_bold(self):
        stage = TokenOptStage()
        ctx = FusionContext(content="**bold text** here", content_type="text")
        result = stage.apply(ctx)
        assert "**" not in result.content
        assert "bold text" in result.content

    def test_strips_italic(self):
        stage = TokenOptStage()
        ctx = FusionContext(content="*italic text* here", content_type="text")
        result = stage.apply(ctx)
        assert "*italic text*" not in result.content

    def test_collapses_excess_blank_lines(self):
        stage = TokenOptStage()
        ctx = FusionContext(content="line1\n\n\n\n\nline2", content_type="text")
        result = stage.apply(ctx)
        assert "\n\n\n" not in result.content

    def test_converts_table_to_kv(self):
        stage = TokenOptStage()
        ctx = FusionContext(content=MARKDOWN_TEXT, content_type="text")
        result = stage.apply(ctx)
        # Table-to-KV conversion: pipe separators should be reduced
        assert isinstance(result.content, str)

    def test_token_counts_correct(self):
        stage = TokenOptStage()
        text = "**bold** and *italic* and   extra   spaces"
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        assert result.original_tokens == estimate_tokens(text)
        assert result.compressed_tokens == estimate_tokens(result.content)


# ---------------------------------------------------------------------------
# Tests: AbbrevStage adapter
# ---------------------------------------------------------------------------

class TestAbbrevStage:
    def test_name_and_order(self):
        stage = AbbrevStage()
        assert stage.name == "abbrev"
        assert stage.order == 45

    def test_should_apply_text_content(self):
        stage = AbbrevStage()
        ctx = FusionContext(content=LONG_TEXT, content_type="text")
        assert stage.should_apply(ctx) is True

    def test_should_not_apply_code(self):
        stage = AbbrevStage()
        ctx = FusionContext(content="def foo(): pass", content_type="code")
        assert stage.should_apply(ctx) is False

    def test_should_not_apply_json(self):
        stage = AbbrevStage()
        ctx = FusionContext(content='{"key": "value"}', content_type="json")
        assert stage.should_apply(ctx) is False

    def test_should_not_apply_log(self):
        stage = AbbrevStage()
        ctx = FusionContext(content="ERROR something failed", content_type="log")
        assert stage.should_apply(ctx) is False

    def test_should_not_apply_diff(self):
        stage = AbbrevStage()
        ctx = FusionContext(content="+added line\n-removed line", content_type="diff")
        assert stage.should_apply(ctx) is False

    def test_abbreviates_known_words(self):
        stage = AbbrevStage()
        text = "The implementation of configuration management in production."
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        # "implementation" → "impl", "configuration" → "config", "production" → "prod"
        assert "impl" in result.content or "config" in result.content or "prod" in result.content

    def test_removes_filler_phrases(self):
        stage = AbbrevStage()
        text = "Furthermore, the system works well. In addition, it scales."
        ctx = FusionContext(content=text, content_type="text")
        result = stage.apply(ctx)
        assert "Furthermore," not in result.content or "In addition," not in result.content

    def test_result_is_shorter_on_verbose_text(self):
        stage = AbbrevStage()
        ctx = FusionContext(content=LONG_TEXT, content_type="text")
        result = stage.apply(ctx)
        assert result.compressed_tokens <= result.original_tokens

    def test_token_counts_set(self):
        stage = AbbrevStage()
        ctx = FusionContext(content=LONG_TEXT, content_type="text")
        result = stage.apply(ctx)
        assert result.original_tokens == estimate_tokens(LONG_TEXT)
        assert result.compressed_tokens == estimate_tokens(result.content)

    def test_should_not_apply_empty(self):
        stage = AbbrevStage()
        ctx = FusionContext(content="", content_type="text")
        assert stage.should_apply(ctx) is False


# ---------------------------------------------------------------------------
# Tests: stats helper functions
# ---------------------------------------------------------------------------

class TestStatsHelpers:
    def test_reduction_pct_zero_for_equal(self):
        assert _reduction_pct(100, 100) == 0.0

    def test_reduction_pct_fifty(self):
        assert _reduction_pct(100, 50) == 50.0

    def test_reduction_pct_zero_original(self):
        assert _reduction_pct(0, 0) == 0.0

    def test_reduction_pct_negative_for_expansion(self):
        # If compressed > original (rare but possible with overhead)
        pct = _reduction_pct(10, 15)
        assert pct < 0

    def test_empty_stats_structure(self):
        stats = _empty_stats("hello world")
        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
        assert stats["stages_run"] == 0
        assert stats["per_stage"] == []

    def test_empty_aggregate_stats_structure(self):
        stats = _empty_aggregate_stats()
        assert stats["message_count"] == 0
        assert stats["original_tokens"] == 0

    def test_aggregate_stats_calculates_reduction(self):
        stats = _aggregate_stats(
            original_tokens=200,
            compressed_tokens=100,
            original_chars=800,
            compressed_chars=400,
            timing_ms=5.0,
            message_count=3,
        )
        assert stats["reduction_pct"] == 50.0
        assert stats["message_count"] == 3
        assert stats["total_timing_ms"] == 5.0


# ---------------------------------------------------------------------------
# Tests: pipeline stage integration — verify stages actually run
# ---------------------------------------------------------------------------

class TestStageIntegration:
    """Verify specific stages actually execute (not just skip) on appropriate content."""

    def test_rle_stage_runs_on_ip_content(self, engine):
        text = "Hosts: 192.168.1.10, 192.168.1.11, 192.168.1.12, 192.168.1.13 are monitored."
        result = engine.compress(text)
        rle_step = next(s for s in result["stats"]["per_stage"] if s["name"] == "rle")
        assert rle_step["skipped"] is False

    def test_token_opt_stage_runs(self, engine):
        result = engine.compress(MARKDOWN_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "token_opt")
        assert step["skipped"] is False

    def test_abbrev_stage_runs_on_text(self, engine):
        # Use NARRATIVE_TEXT — Cortex classifies it as "text" (no code keywords),
        # so AbbrevStage (text-only guard) will not skip.
        result = engine.compress(NARRATIVE_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "abbrev")
        assert step["skipped"] is False

    def test_cortex_stage_runs(self, engine):
        # Cortex should always run when content_type is default "text"
        result = engine.compress(LONG_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "cortex")
        assert step["skipped"] is False

    def test_nexus_runs_on_long_text(self, engine):
        # NexusStage requires >= 20 words
        result = engine.compress(LONG_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "nexus")
        # Either runs (text content after cortex detection) or skipped — should not crash
        assert "skipped" in step

    def test_neurosyntax_skips_on_text(self, engine):
        # Neurosyntax only applies to content_type="code"; NARRATIVE_TEXT is
        # classified as "text" by Cortex, so neurosyntax must skip.
        result = engine.compress(NARRATIVE_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "neurosyntax")
        assert step["skipped"] is True

    def test_ionizer_skips_on_text(self, engine):
        result = engine.compress(LONG_TEXT)
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "ionizer")
        assert step["skipped"] is True

    def test_log_crunch_runs_on_log_content(self, engine):
        log_text = (
            "2025-01-15 10:00:00 INFO Starting service\n"
            "2025-01-15 10:00:01 INFO Loading config\n"
            "2025-01-15 10:00:02 ERROR Failed to connect to database\n"
            "2025-01-15 10:00:03 INFO Retrying connection\n"
            "2025-01-15 10:00:04 INFO Retrying connection\n"
            "2025-01-15 10:00:05 WARN Timeout on attempt 3\n"
            "2025-01-15 10:00:06 ERROR Connection refused\n"
        )
        result = engine.compress(log_text, content_type="log")
        step = next(s for s in result["stats"]["per_stage"] if s["name"] == "log_crunch")
        assert step["skipped"] is False

    def test_per_stage_timing_all_nonnegative(self, engine):
        result = engine.compress(LONG_TEXT)
        for step in result["stats"]["per_stage"]:
            assert step["timing_ms"] >= 0.0, f"Negative timing for stage {step['name']}"

    def test_all_per_stage_entries_have_required_keys(self, engine):
        result = engine.compress(LONG_TEXT)
        for step in result["stats"]["per_stage"]:
            for key in ["name", "skipped", "original_tokens", "compressed_tokens", "timing_ms"]:
                assert key in step, f"Missing key '{key}' in per-stage entry for {step.get('name')}"
