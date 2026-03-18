"""Tests for Rewind reversible compression engine.

Covers: RewindStore, markers (embed/extract/has/strip), and retriever
(rewind_tool_def, handle_rewind).

The retriever module (scripts/lib/rewind/retriever.py) may not yet exist on
disk; those tests import it conditionally and skip gracefully if missing, while
still testing the store and marker modules which are present.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.rewind.store import RewindStore
from claw_compactor.rewind.marker import (
    embed_marker,
    extract_markers,
    has_markers,
    strip_markers,
    MarkerInfo,
)

# ---------------------------------------------------------------------------
# Try to import retriever; skip retriever tests if module is missing
# ---------------------------------------------------------------------------
try:
    from claw_compactor.rewind.retriever import rewind_tool_def, handle_rewind
    _HAS_RETRIEVER = True
except (ImportError, ModuleNotFoundError):
    _HAS_RETRIEVER = False

requires_retriever = pytest.mark.skipif(
    not _HAS_RETRIEVER,
    reason="lib/rewind/retriever.py not yet implemented",
)


# ===========================================================================
# RewindStore
# ===========================================================================

class TestRewindStoreBasics:
    def test_store_returns_24_char_hex(self):
        store = RewindStore()
        hash_id = store.store("hello world", "hw")
        assert len(hash_id) == 24
        assert all(c in "0123456789abcdef" for c in hash_id)

    def test_store_same_content_returns_same_hash(self):
        store = RewindStore()
        h1 = store.store("same content", "sc")
        h2 = store.store("same content", "sc")
        assert h1 == h2

    def test_store_different_content_different_hash(self):
        store = RewindStore()
        h1 = store.store("content A", "cA")
        h2 = store.store("content B", "cB")
        assert h1 != h2

    def test_retrieve_returns_original_text(self):
        store = RewindStore()
        original = "The quick brown fox jumps over the lazy dog."
        hash_id = store.store(original, "compressed version")
        result = store.retrieve(hash_id)
        assert result == original

    def test_retrieve_unknown_hash_returns_none(self):
        store = RewindStore()
        result = store.retrieve("a" * 24)
        assert result is None

    def test_retrieve_wrong_length_hash_returns_none(self):
        store = RewindStore()
        result = store.retrieve("tooshort")
        assert result is None

    def test_size_increments_on_store(self):
        store = RewindStore()
        assert store.size == 0
        store.store("a", "x")
        assert store.size == 1
        store.store("b", "y")
        assert store.size == 2

    def test_size_does_not_grow_for_duplicate_content(self):
        store = RewindStore()
        store.store("same", "s")
        store.store("same", "s")
        assert store.size == 1

    def test_clear_empties_store(self):
        store = RewindStore()
        store.store("hello", "h")
        store.clear()
        assert store.size == 0

    def test_store_with_token_counts(self):
        store = RewindStore()
        hash_id = store.store("text", "t", original_tokens=10, compressed_tokens=5)
        assert len(hash_id) == 24
        # retrieve still works
        assert store.retrieve(hash_id) == "text"

    def test_store_empty_string(self):
        store = RewindStore()
        hash_id = store.store("", "")
        assert len(hash_id) == 24
        assert store.retrieve(hash_id) == ""

    def test_store_unicode_content(self):
        store = RewindStore()
        original = "中文内容 emoji 🎉 日本語テスト"
        hash_id = store.store(original, "compressed")
        assert store.retrieve(hash_id) == original


class TestRewindStoreTTL:
    def test_retrieve_within_ttl_succeeds(self, monkeypatch):
        base_time = 1000.0
        call_count = [0]

        def mock_monotonic():
            # First call: store time; second call: retrieve time (within TTL)
            call_count[0] += 1
            if call_count[0] <= 1:
                return base_time
            return base_time + 50  # 50s later, TTL=600

        monkeypatch.setattr(time, "monotonic", mock_monotonic)
        store = RewindStore(ttl_seconds=600)
        hash_id = store.store("hello", "h")
        result = store.retrieve(hash_id)
        assert result == "hello"

    def test_retrieve_after_ttl_returns_none(self, monkeypatch):
        base_time = 1000.0
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return base_time
            return base_time + 700  # 700s later, TTL=600

        monkeypatch.setattr(time, "monotonic", mock_monotonic)
        store = RewindStore(ttl_seconds=600)
        hash_id = store.store("hello", "h")
        result = store.retrieve(hash_id)
        assert result is None

    def test_expired_entry_removed_from_store(self, monkeypatch):
        base_time = 1000.0
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return base_time
            return base_time + 700

        monkeypatch.setattr(time, "monotonic", mock_monotonic)
        store = RewindStore(ttl_seconds=600)
        hash_id = store.store("hello", "h")
        store.retrieve(hash_id)  # triggers expiry deletion
        assert store.size == 0

    def test_short_ttl_expires_quickly(self, monkeypatch):
        base_time = 500.0
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return base_time
            return base_time + 2  # 2s later, TTL=1

        monkeypatch.setattr(time, "monotonic", mock_monotonic)
        store = RewindStore(ttl_seconds=1)
        hash_id = store.store("ephemeral", "e")
        assert store.retrieve(hash_id) is None


class TestRewindStoreLRU:
    def test_lru_eviction_when_max_entries_exceeded(self):
        store = RewindStore(max_entries=3)
        h1 = store.store("first",  "f1")
        h2 = store.store("second", "f2")
        h3 = store.store("third",  "f3")
        assert store.size == 3
        # Adding a 4th entry should evict the oldest (h1)
        store.store("fourth", "f4")
        assert store.size == 3
        assert store.retrieve(h1) is None

    def test_lru_retains_most_recent_entries(self):
        store = RewindStore(max_entries=3)
        store.store("first",  "f1")
        h2 = store.store("second", "f2")
        h3 = store.store("third",  "f3")
        h4 = store.store("fourth", "f4")
        assert store.retrieve(h2) == "second"
        assert store.retrieve(h3) == "third"
        assert store.retrieve(h4) == "fourth"

    def test_accessing_entry_refreshes_lru_position(self):
        store = RewindStore(max_entries=3)
        h1 = store.store("first",  "f1")
        store.store("second", "f2")
        store.store("third",  "f3")
        # Access h1 to make it recently used
        store.retrieve(h1)
        # Add fourth entry — should evict second (oldest unreferenced)
        store.store("fourth", "f4")
        assert store.retrieve(h1) == "first"

    def test_max_entries_one_keeps_last_only(self):
        store = RewindStore(max_entries=1)
        store.store("first", "f")
        h2 = store.store("second", "s")
        assert store.size == 1
        assert store.retrieve(h2) == "second"


class TestRewindStoreSearch:
    def test_search_with_matching_keyword_returns_matching_lines(self):
        store = RewindStore()
        original = "line one: apple\nline two: banana\nline three: cherry"
        hash_id = store.store(original, "compressed")
        result = store.search(hash_id, ["banana"])
        assert "banana" in result
        assert "apple" not in result
        assert "cherry" not in result

    def test_search_with_no_keywords_returns_full_original(self):
        store = RewindStore()
        original = "line one\nline two\nline three"
        hash_id = store.store(original, "compressed")
        result = store.search(hash_id, [])
        assert result == original

    def test_search_case_insensitive(self):
        store = RewindStore()
        original = "Deploy on AWS\nSetup Python 3.11"
        hash_id = store.store(original, "c")
        result = store.search(hash_id, ["aws"])
        assert "AWS" in result

    def test_search_no_match_returns_full_original(self):
        store = RewindStore()
        original = "line one\nline two"
        hash_id = store.store(original, "c")
        result = store.search(hash_id, ["zzznomatch"])
        assert result == original

    def test_search_unknown_hash_returns_none(self):
        store = RewindStore()
        result = store.search("a" * 24, ["keyword"])
        assert result is None

    def test_search_multiple_keywords_returns_union(self):
        store = RewindStore()
        original = "apple pie\nbanana split\ncherry cola\ndate cake"
        hash_id = store.store(original, "c")
        result = store.search(hash_id, ["apple", "cherry"])
        assert "apple" in result
        assert "cherry" in result
        assert "banana" not in result

    def test_search_with_multiline_match(self):
        store = RewindStore()
        original = "key: value1\nother: value2\nkey: value3"
        hash_id = store.store(original, "c")
        result = store.search(hash_id, ["key"])
        lines = result.split("\n")
        assert len(lines) == 2
        assert all("key" in line for line in lines)


# ===========================================================================
# Marker tests
# ===========================================================================

class TestEmbedMarker:
    def test_appends_marker_to_text(self):
        result = embed_marker("compressed text", 10, 3, "a" * 24)
        assert result.startswith("compressed text")
        assert f"hash={'a' * 24}" in result

    def test_marker_format_plural_items(self):
        result = embed_marker("text", 5, 2, "b" * 24)
        assert "5 items compressed to 2" in result

    def test_marker_format_singular_item(self):
        result = embed_marker("text", 1, 1, "c" * 24)
        assert "1 item compressed to 1" in result

    def test_marker_contains_retrieve_keyword(self):
        result = embed_marker("text", 3, 1, "d" * 24)
        assert "Retrieve:" in result

    def test_marker_newline_separator(self):
        result = embed_marker("my text", 2, 1, "e" * 24)
        assert "\n" in result
        parts = result.split("\n")
        assert parts[0] == "my text"

    def test_embed_then_extract_roundtrip(self):
        hash_id = "f" * 24
        embedded = embed_marker("original content", 7, 3, hash_id)
        markers = extract_markers(embedded)
        assert len(markers) == 1
        assert markers[0].hash_id == hash_id
        assert markers[0].original_count == 7
        assert markers[0].compressed_count == 3


class TestExtractMarkers:
    def test_no_markers_returns_empty_list(self):
        result = extract_markers("plain text with no markers")
        assert result == []

    def test_single_marker_extracted(self):
        text = "some text\n[5 items compressed to 2. Retrieve: hash={}]".format("a" * 24)
        markers = extract_markers(text)
        assert len(markers) == 1
        assert markers[0].original_count == 5
        assert markers[0].compressed_count == 2
        assert markers[0].hash_id == "a" * 24

    def test_multiple_markers_extracted(self):
        hash1 = "a" * 24
        hash2 = "b" * 24
        text = (
            f"[3 items compressed to 1. Retrieve: hash={hash1}]\n"
            f"middle text\n"
            f"[7 items compressed to 2. Retrieve: hash={hash2}]"
        )
        markers = extract_markers(text)
        assert len(markers) == 2
        assert markers[0].hash_id == hash1
        assert markers[1].hash_id == hash2

    def test_marker_info_has_span(self):
        hash_id = "c" * 24
        text = f"[4 items compressed to 2. Retrieve: hash={hash_id}]"
        markers = extract_markers(text)
        assert len(markers) == 1
        start, end = markers[0].span
        assert start == 0
        assert end == len(text)

    def test_singular_item_marker_extracted(self):
        hash_id = "d" * 24
        text = f"[1 item compressed to 1. Retrieve: hash={hash_id}]"
        markers = extract_markers(text)
        assert len(markers) == 1
        assert markers[0].original_count == 1

    def test_marker_info_is_frozen(self):
        hash_id = "e" * 24
        text = f"[2 items compressed to 1. Retrieve: hash={hash_id}]"
        markers = extract_markers(text)
        info = markers[0]
        with pytest.raises((Exception,)):
            info.hash_id = "modified"  # type: ignore[misc]


class TestHasMarkers:
    def test_returns_true_when_marker_present(self):
        hash_id = "a" * 24
        text = f"some content\n[5 items compressed to 2. Retrieve: hash={hash_id}]"
        assert has_markers(text) is True

    def test_returns_false_when_no_marker(self):
        assert has_markers("plain text without any markers") is False

    def test_returns_false_for_empty_string(self):
        assert has_markers("") is False

    def test_returns_true_for_singular_item_marker(self):
        hash_id = "b" * 24
        text = f"[1 item compressed to 1. Retrieve: hash={hash_id}]"
        assert has_markers(text) is True

    def test_partial_marker_not_detected(self):
        # Missing the Retrieve part
        assert has_markers("[5 items compressed to 2]") is False


class TestStripMarkers:
    def test_removes_marker_from_text(self):
        hash_id = "a" * 24
        text = f"important content\n[5 items compressed to 2. Retrieve: hash={hash_id}]"
        result = strip_markers(text)
        assert f"hash={hash_id}" not in result
        assert "important content" in result

    def test_removes_multiple_markers(self):
        hash1 = "a" * 24
        hash2 = "b" * 24
        text = (
            f"content\n"
            f"[3 items compressed to 1. Retrieve: hash={hash1}]\n"
            f"more content\n"
            f"[7 items compressed to 2. Retrieve: hash={hash2}]"
        )
        result = strip_markers(text)
        assert hash1 not in result
        assert hash2 not in result
        assert "content" in result
        assert "more content" in result

    def test_text_without_markers_unchanged(self):
        text = "no markers here at all"
        assert strip_markers(text) == text

    def test_empty_string_unchanged(self):
        assert strip_markers("") == ""

    def test_strip_then_has_markers_is_false(self):
        hash_id = "c" * 24
        text = f"text\n[2 items compressed to 1. Retrieve: hash={hash_id}]"
        stripped = strip_markers(text)
        assert not has_markers(stripped)


# ===========================================================================
# Retriever tests (skipped if module not implemented)
# ===========================================================================

@requires_retriever
class TestRewindToolDef:
    def test_openai_format_has_required_keys(self):
        tool_def = rewind_tool_def(provider="openai")
        assert "type" in tool_def
        assert tool_def["type"] == "function"
        assert "function" in tool_def

    def test_openai_format_function_has_name(self):
        tool_def = rewind_tool_def(provider="openai")
        fn = tool_def["function"]
        assert "name" in fn
        assert isinstance(fn["name"], str)
        assert len(fn["name"]) > 0

    def test_openai_format_function_has_description(self):
        tool_def = rewind_tool_def(provider="openai")
        fn = tool_def["function"]
        assert "description" in fn
        assert isinstance(fn["description"], str)

    def test_openai_format_has_parameters(self):
        tool_def = rewind_tool_def(provider="openai")
        fn = tool_def["function"]
        assert "parameters" in fn
        params = fn["parameters"]
        assert "properties" in params
        assert "hash_id" in params["properties"]

    def test_anthropic_format_has_required_keys(self):
        tool_def = rewind_tool_def(provider="anthropic")
        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def

    def test_anthropic_format_name_is_string(self):
        tool_def = rewind_tool_def(provider="anthropic")
        assert isinstance(tool_def["name"], str)
        assert len(tool_def["name"]) > 0

    def test_anthropic_format_input_schema_has_hash_id(self):
        tool_def = rewind_tool_def(provider="anthropic")
        schema = tool_def["input_schema"]
        assert "properties" in schema
        assert "hash_id" in schema["properties"]

    def test_default_provider_is_openai(self):
        tool_def = rewind_tool_def()
        # Default (openai) should return a function-wrapped format
        assert "type" in tool_def
        assert tool_def["type"] == "function"


@requires_retriever
class TestHandleRewind:
    def _make_store_with_content(self, content: str) -> tuple[RewindStore, str]:
        store = RewindStore()
        hash_id = store.store(content, "compressed")
        return store, hash_id

    def _call(self, store: RewindStore, hash_id: str, keywords: list | None = None) -> dict:
        """Build a tool_call dict and invoke handle_rewind."""
        args: dict = {"hash_id": hash_id}
        if keywords is not None:
            args["keywords"] = keywords
        return handle_rewind(store, {"arguments": args})

    def test_success_case_returns_original(self):
        original = "Full original text with details."
        store, hash_id = self._make_store_with_content(original)
        result = self._call(store, hash_id)
        assert result["status"] == "ok"
        assert result["content"] == original

    def test_not_found_case_returns_not_found_status(self):
        store = RewindStore()
        result = self._call(store, "a" * 24)
        assert result["status"] == "not_found"
        assert "message" in result

    def test_not_found_message_mentions_hash(self):
        store = RewindStore()
        unknown_hash = "b" * 24
        result = self._call(store, unknown_hash)
        assert unknown_hash in result["message"]

    def test_with_keywords_returns_filtered_content(self):
        original = "line one: python\nline two: javascript\nline three: python again"
        store, hash_id = self._make_store_with_content(original)
        result = self._call(store, hash_id, keywords=["python"])
        assert result["status"] == "ok"
        assert "python" in result["content"].lower()

    def test_with_keywords_excludes_non_matching_lines(self):
        original = "line one: python\nline two: javascript\nline three: python again"
        store, hash_id = self._make_store_with_content(original)
        result = self._call(store, hash_id, keywords=["python"])
        assert result["status"] == "ok"
        assert "javascript" not in result["content"]

    def test_with_empty_keywords_returns_full_content(self):
        original = "all lines\nshould be\nreturned"
        store, hash_id = self._make_store_with_content(original)
        result = self._call(store, hash_id, keywords=[])
        assert result["status"] == "ok"
        assert "all lines" in result["content"]
        assert "returned" in result["content"]

    def test_tool_call_with_input_key_anthropic_style(self):
        """handle_rewind also accepts Anthropic-style 'input' key."""
        original = "anthropic style input"
        store, hash_id = self._make_store_with_content(original)
        result = handle_rewind(store, {"input": {"hash_id": hash_id}})
        assert result["status"] == "ok"
        assert result["content"] == original

    def test_tool_call_with_json_string_arguments(self):
        """handle_rewind handles arguments as a JSON string."""
        import json
        original = "json string args"
        store, hash_id = self._make_store_with_content(original)
        result = handle_rewind(store, {"arguments": json.dumps({"hash_id": hash_id})})
        assert result["status"] == "ok"
        assert result["content"] == original
