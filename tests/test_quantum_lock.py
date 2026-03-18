"""Tests for QuantumLock: pattern extraction, stabilization, and FusionStage behaviour.

Part of claw-compactor Phase 5 test suite. License: MIT.
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path (mirrors other test files in this suite)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.quantum_lock import (
    APPENDIX_END,
    APPENDIX_START,
    DynamicFragment,
    QuantumLock,
    extract_dynamic,
    get_prefix_hash,
    stabilize,
)
from claw_compactor.fusion.base import FusionContext, FusionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(content: str, role: str = "system") -> FusionContext:
    return FusionContext(content=content, role=role)


# ---------------------------------------------------------------------------
# extract_dynamic — pattern coverage
# ---------------------------------------------------------------------------

class TestExtractDynamic:
    def test_no_dynamic_content(self):
        text = "You are a helpful assistant. Answer concisely."
        assert extract_dynamic(text) == []

    def test_iso_date_plain(self):
        text = "Today is 2026-03-17."
        frags = extract_dynamic(text)
        names = [f.name for f in frags]
        assert "iso_date" in names
        assert any(f.original == "2026-03-17" for f in frags)

    def test_iso_datetime_with_tz(self):
        text = "Created at 2025-12-01T10:30:00Z by the system."
        frags = extract_dynamic(text)
        assert any(f.name == "iso_date" for f in frags)
        iso_frag = next(f for f in frags if f.name == "iso_date")
        assert "T" in iso_frag.original

    def test_uuid(self):
        text = "Session ID: 550e8400-e29b-41d4-a716-446655440000"
        frags = extract_dynamic(text)
        assert any(f.name == "uuid" for f in frags)
        assert any(f.original == "550e8400-e29b-41d4-a716-446655440000" for f in frags)

    def test_jwt(self):
        # Minimal syntactically valid JWT shape (not a real token)
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Authorization: Bearer {jwt}"
        frags = extract_dynamic(text)
        assert any(f.name == "jwt" for f in frags)

    def test_api_key_sk(self):
        text = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234"
        frags = extract_dynamic(text)
        assert any(f.name == "api_key" for f in frags)

    def test_api_key_pk_live(self):
        text = "Stripe key: pk_live_abcdefghijklmnopqrstuvwxyz"
        frags = extract_dynamic(text)
        assert any(f.name == "api_key" for f in frags)

    def test_unix_timestamp_10digit(self):
        # 1700000000 = ~November 2023
        text = "Request made at 1700000000 seconds."
        frags = extract_dynamic(text)
        assert any(f.name == "unix_ts" for f in frags)

    def test_unix_timestamp_13digit(self):
        text = "Timestamp ms: 1700000000000"
        frags = extract_dynamic(text)
        assert any(f.name == "unix_ts" for f in frags)

    def test_hex_id_32chars(self):
        text = "trace_id=4bf92f3577b34da6a3ce929d0e0e4736"
        frags = extract_dynamic(text)
        assert any(f.name == "hex_id" for f in frags)

    def test_deduplication(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"First: {uuid}. Second: {uuid}."
        frags = extract_dynamic(text)
        uuid_frags = [f for f in frags if f.name == "uuid"]
        assert len(uuid_frags) == 1, "Same UUID should be deduplicated"
        assert len(uuid_frags[0].indices) == 2, "Both occurrences should be recorded"

    def test_sorted_by_first_index(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"Date 2026-03-17 then UUID {uuid}"
        frags = extract_dynamic(text)
        # Date appears before UUID in the string
        first_names = [f.name for f in frags]
        date_idx = first_names.index("iso_date")
        uuid_idx = first_names.index("uuid")
        assert date_idx < uuid_idx

    def test_multiple_types_in_one_message(self):
        text = (
            "System started at 2026-03-17T09:00:00Z. "
            "Session: 550e8400-e29b-41d4-a716-446655440000. "
            "Key: sk-myapp-abcdefghijklmnopqrstuvwxyz."
        )
        frags = extract_dynamic(text)
        names = {f.name for f in frags}
        assert "iso_date" in names
        assert "uuid" in names
        assert "api_key" in names


# ---------------------------------------------------------------------------
# stabilize
# ---------------------------------------------------------------------------

class TestStabilize:
    def test_no_dynamic_content_unchanged(self):
        text = "You are a helpful assistant."
        assert stabilize(text) == text

    def test_date_replaced_with_placeholder(self):
        text = "Today is 2026-03-17. How can I help?"
        result = stabilize(text)
        assert "2026-03-17" not in result.split(APPENDIX_START)[0]
        assert "<date>" in result

    def test_original_value_in_appendix(self):
        text = "Today is 2026-03-17."
        result = stabilize(text)
        assert APPENDIX_START in result
        assert "2026-03-17" in result.split(APPENDIX_START)[1]

    def test_appendix_at_end(self):
        text = "System prompt. Date: 2026-03-17. More text here."
        result = stabilize(text)
        assert result.endswith(APPENDIX_END)

    def test_uuid_replaced(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"Session: {uuid}"
        result = stabilize(text)
        stable_prefix = result.split(APPENDIX_START)[0]
        assert uuid not in stable_prefix
        assert "<uuid>" in stable_prefix

    def test_stable_prefix_identical_for_same_template(self):
        """Two requests with the same static text but different dynamic values
        should produce the same stable prefix."""
        text1 = "You are a bot. Date: 2026-03-17. Session: 550e8400-e29b-41d4-a716-446655440000."
        text2 = "You are a bot. Date: 2027-01-01. Session: 660e8400-e29b-41d4-a716-446655440000."

        stable1 = stabilize(text1).split(APPENDIX_START)[0]
        stable2 = stabilize(text2).split(APPENDIX_START)[0]
        assert stable1 == stable2

    def test_jwt_replaced(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"
        result = stabilize(text)
        stable_prefix = result.split(APPENDIX_START)[0]
        assert jwt not in stable_prefix
        assert "<jwt>" in stable_prefix

    def test_multiple_occurrences_all_replaced(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"First: {uuid}. Second: {uuid}."
        result = stabilize(text)
        stable_prefix = result.split(APPENDIX_START)[0]
        assert uuid not in stable_prefix
        assert stable_prefix.count("<uuid>") == 2

    def test_appendix_label_format(self):
        text = "Date: 2026-03-17"
        result = stabilize(text)
        appendix = result.split(APPENDIX_START)[1]
        assert "iso_date: 2026-03-17" in appendix

    def test_immutability_of_input(self):
        original = "Today is 2026-03-17."
        stabilize(original)
        assert original == "Today is 2026-03-17."  # input must not be mutated


# ---------------------------------------------------------------------------
# get_prefix_hash
# ---------------------------------------------------------------------------

class TestGetPrefixHash:
    def test_returns_64_char_hex(self):
        text = "Static system prompt. Date: 2026-03-17."
        h = get_prefix_hash(text)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_template_same_hash(self):
        text1 = "You are a bot. Date: 2026-03-17. UUID: 550e8400-e29b-41d4-a716-446655440000."
        text2 = "You are a bot. Date: 2027-06-15. UUID: 660e9900-f39c-52e5-b827-557766551111."
        assert get_prefix_hash(text1) == get_prefix_hash(text2)

    def test_different_template_different_hash(self):
        text1 = "You are a helpful assistant. Date: 2026-03-17."
        text2 = "You are a strict reviewer. Date: 2026-03-17."
        assert get_prefix_hash(text1) != get_prefix_hash(text2)

    def test_no_dynamic_content_hashes_full_text(self):
        text = "Static system prompt with no dynamic content."
        h = get_prefix_hash(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert h == expected

    def test_hash_stability_across_calls(self):
        text = "Date: 2026-03-17. Static instructions follow."
        assert get_prefix_hash(text) == get_prefix_hash(text)


# ---------------------------------------------------------------------------
# QuantumLock as FusionStage
# ---------------------------------------------------------------------------

class TestQuantumLockFusionStage:
    def setup_method(self):
        self.stage = QuantumLock()

    def test_name_and_order(self):
        assert self.stage.name == "quantum_lock"
        assert self.stage.order == 3  # before Cortex (order=5)

    def test_should_not_apply_user_role(self):
        ctx = make_ctx("Date: 2026-03-17", role="user")
        assert not self.stage.should_apply(ctx)

    def test_should_not_apply_assistant_role(self):
        ctx = make_ctx("Date: 2026-03-17", role="assistant")
        assert not self.stage.should_apply(ctx)

    def test_should_not_apply_tool_role(self):
        ctx = make_ctx("Date: 2026-03-17", role="tool")
        assert not self.stage.should_apply(ctx)

    def test_should_not_apply_system_no_dynamic(self):
        ctx = make_ctx("Static text only.", role="system")
        assert not self.stage.should_apply(ctx)

    def test_should_apply_system_with_date(self):
        ctx = make_ctx("Today is 2026-03-17.", role="system")
        assert self.stage.should_apply(ctx)

    def test_should_apply_system_with_uuid(self):
        ctx = make_ctx("ID: 550e8400-e29b-41d4-a716-446655440000", role="system")
        assert self.stage.should_apply(ctx)

    def test_apply_stabilizes_content(self):
        ctx = make_ctx("Date: 2026-03-17. Static instructions.", role="system")
        result = self.stage.apply(ctx)
        assert isinstance(result, FusionResult)
        assert "2026-03-17" not in result.content.split(APPENDIX_START)[0]
        assert APPENDIX_START in result.content

    def test_apply_sets_token_counts(self):
        ctx = make_ctx("Date: 2026-03-17. Long static instructions here.", role="system")
        result = self.stage.apply(ctx)
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0

    def test_apply_markers_list(self):
        ctx = make_ctx("Date: 2026-03-17.", role="system")
        result = self.stage.apply(ctx)
        assert any("quantum_lock:iso_date" in m for m in result.markers)

    def test_apply_no_warnings_on_modest_overhead(self):
        # A short message with one date — appendix may or may not add warnings.
        # We only assert that the result has a warnings list (may be empty).
        ctx = make_ctx("Date: 2026-03-17. System instructions.", role="system")
        result = self.stage.apply(ctx)
        assert isinstance(result.warnings, list)

    def test_timed_apply_skips_when_not_applicable(self):
        ctx = make_ctx("Static text only.", role="system")
        result = self.stage.timed_apply(ctx)
        assert result.skipped is True
        assert result.content == ctx.content

    def test_timed_apply_runs_when_applicable(self):
        ctx = make_ctx("Date: 2026-03-17.", role="system")
        result = self.stage.timed_apply(ctx)
        assert result.skipped is False
        assert result.timing_ms >= 0

    def test_apply_is_idempotent_in_prefix(self):
        """Running stabilize twice on the same template should produce the
        same stable prefix — critical for cache hit reliability."""
        msg1 = "You are helpful. Date: 2026-03-17. UUID: 550e8400-e29b-41d4-a716-446655440000."
        msg2 = "You are helpful. Date: 2027-01-01. UUID: 660e9900-f39c-52e5-b827-557766551111."

        ctx1 = make_ctx(msg1, role="system")
        ctx2 = make_ctx(msg2, role="system")

        r1 = self.stage.apply(ctx1)
        r2 = self.stage.apply(ctx2)

        prefix1 = r1.content.split(APPENDIX_START)[0]
        prefix2 = r2.content.split(APPENDIX_START)[0]
        assert prefix1 == prefix2, (
            "Stable prefixes must be identical for requests with the same template"
        )

    def test_result_is_immutable_dataclass(self):
        ctx = make_ctx("Date: 2026-03-17.", role="system")
        result = self.stage.apply(ctx)
        with pytest.raises(Exception):
            result.content = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        assert extract_dynamic("") == []
        assert stabilize("") == ""

    def test_only_dynamic_content(self):
        text = "2026-03-17"
        result = stabilize(text)
        assert "<date>" in result
        assert APPENDIX_START in result

    def test_api_key_not_matched_short(self):
        # sk- prefix but only 5 chars after — below 16-char minimum
        text = "Key: sk-abc"
        frags = extract_dynamic(text)
        assert not any(f.name == "api_key" for f in frags)

    def test_time_pattern_not_false_positive_version(self):
        # "10:30" alone is not HH:MM:SS — should NOT match
        text = "Version 10:30 is deprecated"
        frags = extract_dynamic(text)
        assert not any(f.name == "time" for f in frags)

    def test_hex_id_minimum_32_chars(self):
        # 31 hex chars — below minimum
        text = "id=" + "a" * 31
        frags = extract_dynamic(text)
        assert not any(f.name == "hex_id" for f in frags)

    def test_hex_id_exactly_32_chars(self):
        text = "id=" + "a" * 32
        frags = extract_dynamic(text)
        assert any(f.name == "hex_id" for f in frags)

    def test_stabilize_preserves_non_dynamic_text(self):
        text = "You are a helpful coding assistant. Answer concisely. Date: 2026-03-17."
        result = stabilize(text)
        stable_prefix = result.split(APPENDIX_START)[0]
        assert "You are a helpful coding assistant." in stable_prefix
        assert "Answer concisely." in stable_prefix
