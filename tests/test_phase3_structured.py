"""Phase 3 structured-data compression tests.

Covers Ionizer, LogCrunch, SearchCrunch, and DiffCrunch with realistic
data, edge cases, and integration with RewindStore.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import json
import sys
import os

# Ensure the scripts directory is on the path so imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

from claw_compactor.fusion.base import FusionContext
from claw_compactor.fusion.ionizer import Ionizer
from claw_compactor.fusion.log_crunch import LogCrunch
from claw_compactor.fusion.search_crunch import SearchCrunch
from claw_compactor.fusion.diff_crunch import DiffCrunch
from claw_compactor.rewind.store import RewindStore
from claw_compactor.rewind.marker import extract_markers, has_markers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(content: str, content_type: str = "text", **kwargs) -> FusionContext:
    return FusionContext(content=content, content_type=content_type, **kwargs)


def json_ctx(data) -> FusionContext:
    return make_ctx(json.dumps(data, indent=2), content_type="json")


def log_ctx(text: str) -> FusionContext:
    return make_ctx(text, content_type="log")


def search_ctx(text: str) -> FusionContext:
    return make_ctx(text, content_type="search")


def diff_ctx(text: str) -> FusionContext:
    return make_ctx(text, content_type="diff")


def make_dict_array(n: int, error_indices: set[int] | None = None) -> list[dict]:
    """Build a list of n dicts, optionally injecting error fields at given indices."""
    items = []
    for i in range(n):
        item: dict = {"id": i, "name": f"item_{i}", "value": i * 10}
        if error_indices and i in error_indices:
            item["status"] = "failed"
        items.append(item)
    return items


SAMPLE_LOG = """\
2024-01-15T10:00:00 INFO  Server started on port 8080
2024-01-15T10:00:01 DEBUG Checking configuration
2024-01-15T10:00:02 DEBUG Checking configuration
2024-01-15T10:00:03 DEBUG Checking configuration
2024-01-15T10:00:04 DEBUG Checking configuration
2024-01-15T10:00:05 ERROR Failed to connect to database: connection refused
2024-01-15T10:00:06 INFO  Retrying connection
2024-01-15T10:00:07 DEBUG Checking configuration
2024-01-15T10:00:08 DEBUG Checking configuration
2024-01-15T10:00:09 WARN  Disk usage above 80%
2024-01-15T10:00:10 INFO  Completed health check
"""

SAMPLE_TRACEBACK = """\
ERROR An unexpected error occurred
Traceback (most recent call last):
  File "/app/server.py", line 42, in handle_request
    result = process(data)
  File "/app/processor.py", line 17, in process
    return transform(item)
ValueError: invalid literal for int()
INFO Continuing after error
"""

SAMPLE_GREP = """\
src/auth.py:10:def authenticate(user, password):
src/auth.py:11:    if not user:
src/auth.py:45:    raise AuthError("invalid credentials")
src/models.py:5:class User:
src/models.py:22:    def authenticate(self):
src/utils.py:88:# authenticate helper
"""

SAMPLE_DIFF = """\
diff --git a/src/server.py b/src/server.py
index abc1234..def5678 100644
--- a/src/server.py
+++ b/src/server.py
@@ -1,10 +1,12 @@
 import os
 import sys
-import json
+import json  # added comment

 def start():
+    log.info("starting")
     host = os.environ.get("HOST", "0.0.0.0")
     port = int(os.environ.get("PORT", 8080))
     context_line_1 = True
     context_line_2 = True
     context_line_3 = True
"""


# ===========================================================================
# IONIZER TESTS
# ===========================================================================

class TestIonizer:

    def test_should_apply_json_content_type(self):
        stage = Ionizer()
        ctx = json_ctx([{"id": i} for i in range(10)])
        assert stage.should_apply(ctx) is True

    def test_should_not_apply_non_json(self):
        stage = Ionizer()
        assert stage.should_apply(make_ctx("hello", "text")) is False
        assert stage.should_apply(make_ctx("diff text", "diff")) is False

    def test_small_array_is_skipped(self):
        stage = Ionizer()
        data = [{"id": i} for i in range(4)]
        result = stage.apply(json_ctx(data))
        assert result.skipped is True
        assert result.content == json.dumps(data, indent=2)

    def test_single_item_array_is_skipped(self):
        stage = Ionizer()
        result = stage.apply(json_ctx([{"id": 1}]))
        assert result.skipped is True

    def test_empty_array_is_skipped(self):
        stage = Ionizer()
        result = stage.apply(json_ctx([]))
        assert result.skipped is True

    def test_dict_array_is_compressed(self):
        stage = Ionizer()
        data = make_dict_array(50)
        result = stage.apply(json_ctx(data))
        assert result.skipped is False
        assert result.compressed_tokens < result.original_tokens
        assert "ionizer:dict_array:" in result.markers[0]

    def test_error_items_are_preserved(self):
        stage = Ionizer()
        # Put an error item deep in the middle.
        data = make_dict_array(50, error_indices={25})
        result = stage.apply(json_ctx(data))
        # The compressed output should contain item 25's status.
        assert "failed" in result.content

    def test_schema_header_in_output(self):
        stage = Ionizer()
        data = make_dict_array(30)
        result = stage.apply(json_ctx(data))
        assert "Schema fields" in result.content or "id" in result.content

    def test_string_array_deduplication(self):
        stage = Ionizer()
        data = ["apple", "banana", "apple", "cherry", "banana", "apple", "date"]
        result = stage.apply(json_ctx(data))
        assert result.skipped is False
        # Duplicates removed.
        assert "duplicates removed" in result.content

    def test_string_array_large_sampling(self):
        stage = Ionizer()
        data = [f"log line {i}" for i in range(200)]
        result = stage.apply(json_ctx(data))
        assert result.skipped is False
        assert result.compressed_tokens <= result.original_tokens

    def test_non_array_json_is_skipped(self):
        stage = Ionizer()
        result = stage.apply(make_ctx('{"key": "value"}', "json"))
        assert result.skipped is True

    def test_malformed_json_returns_warning(self):
        stage = Ionizer()
        result = stage.apply(make_ctx("{not valid json}", "json"))
        assert result.skipped is True
        assert len(result.warnings) > 0
        assert "parse error" in result.warnings[0].lower() or "JSON" in result.warnings[0]

    def test_mixed_array_is_skipped(self):
        stage = Ionizer()
        result = stage.apply(json_ctx([1, "two", {"three": 3}]))
        assert result.skipped is True

    def test_rewind_store_integration(self):
        store = RewindStore()
        stage = Ionizer(rewind_store=store)
        data = make_dict_array(30)
        result = stage.apply(json_ctx(data))
        assert result.skipped is False
        assert has_markers(result.content)
        markers = extract_markers(result.content)
        assert len(markers) == 1
        assert markers[0].original_count == 30
        # Can retrieve the original.
        retrieved = store.retrieve(markers[0].hash_id)
        assert retrieved is not None

    def test_front_back_items_in_output(self):
        """Front and back items should always appear in the compressed output."""
        stage = Ionizer()
        data = make_dict_array(100)
        result = stage.apply(json_ctx(data))
        # Item 0 and item 99 should appear.
        assert '"id": 0' in result.content or "item_0" in result.content
        assert '"id": 99' in result.content or "item_99" in result.content

    def test_timed_apply_skip_on_wrong_type(self):
        stage = Ionizer()
        ctx = make_ctx("hello world", "text")
        result = stage.timed_apply(ctx)
        assert result.skipped is True


# ===========================================================================
# LOG CRUNCH TESTS
# ===========================================================================

class TestLogCrunch:

    def test_should_apply_log_content_type(self):
        stage = LogCrunch()
        assert stage.should_apply(log_ctx("ERROR something")) is True

    def test_should_not_apply_non_log(self):
        stage = LogCrunch()
        assert stage.should_apply(make_ctx("hello", "text")) is False

    def test_error_lines_preserved(self):
        stage = LogCrunch()
        result = stage.apply(log_ctx(SAMPLE_LOG))
        assert "ERROR" in result.content or "Failed to connect" in result.content

    def test_warn_lines_preserved(self):
        stage = LogCrunch()
        result = stage.apply(log_ctx(SAMPLE_LOG))
        assert "WARN" in result.content or "Disk usage" in result.content

    def test_repeated_debug_lines_collapsed(self):
        stage = LogCrunch(normalise_timestamps=False)
        log = "\n".join(["DEBUG Checking config"] * 10)
        result = stage.apply(log_ctx(log))
        # Should see a "repeated N times" marker.
        assert "repeated" in result.content
        assert result.compressed_tokens < result.original_tokens

    def test_stack_trace_preserved(self):
        stage = LogCrunch(normalise_timestamps=False)
        result = stage.apply(log_ctx(SAMPLE_TRACEBACK))
        assert "Traceback" in result.content
        assert "File" in result.content
        assert "ValueError" in result.content

    def test_failed_keyword_line_preserved(self):
        stage = LogCrunch(normalise_timestamps=False)
        log = "INFO Starting up\nINFO Connection failed\nINFO Done"
        result = stage.apply(log_ctx(log))
        assert "failed" in result.content.lower()

    def test_timestamp_normalisation(self):
        stage = LogCrunch(normalise_timestamps=True)
        result = stage.apply(log_ctx(SAMPLE_LOG))
        # Should contain relative timestamps like [+0.000s] or similar.
        assert "[+" in result.content

    def test_no_timestamp_normalisation_when_disabled(self):
        stage = LogCrunch(normalise_timestamps=False)
        result = stage.apply(log_ctx(SAMPLE_LOG))
        assert "2024-01-15" in result.content

    def test_empty_log(self):
        stage = LogCrunch()
        result = stage.apply(log_ctx(""))
        assert result.content == ""

    def test_single_line_log(self):
        stage = LogCrunch()
        result = stage.apply(log_ctx("INFO Server started"))
        assert "INFO Server started" in result.content

    def test_marker_in_result(self):
        stage = LogCrunch()
        result = stage.apply(log_ctx(SAMPLE_LOG))
        assert len(result.markers) > 0
        assert "log_crunch:" in result.markers[0]

    def test_compression_ratio_improves_with_repetition(self):
        stage = LogCrunch(normalise_timestamps=False)
        repeated_log = "\n".join(
            ["INFO Heartbeat OK"] * 50
            + ["ERROR Service down"]
            + ["INFO Heartbeat OK"] * 50
        )
        result = stage.apply(log_ctx(repeated_log))
        assert result.compressed_tokens < result.original_tokens

    def test_fatal_lines_preserved(self):
        stage = LogCrunch(normalise_timestamps=False)
        log = "INFO Starting\nFATAL Out of memory\nINFO Exiting"
        result = stage.apply(log_ctx(log))
        assert "FATAL" in result.content or "Out of memory" in result.content

    def test_mixed_log_levels_no_data_loss(self):
        """All ERROR/WARN lines must be in the output."""
        stage = LogCrunch(normalise_timestamps=False)
        errors = [f"ERROR error_{i}" for i in range(5)]
        warns = [f"WARN warn_{i}" for i in range(5)]
        fillers = ["DEBUG filler_line"] * 30
        all_lines = errors + fillers + warns
        result = stage.apply(log_ctx("\n".join(all_lines)))
        for e in errors:
            assert f"error_{e.split('_')[1]}" in result.content
        for w in warns:
            assert f"warn_{w.split('_')[1]}" in result.content


# ===========================================================================
# SEARCH CRUNCH TESTS
# ===========================================================================

class TestSearchCrunch:

    def test_should_apply_search_content_type(self):
        stage = SearchCrunch()
        assert stage.should_apply(search_ctx("src/a.py:1:hello")) is True

    def test_should_not_apply_non_search(self):
        stage = SearchCrunch()
        assert stage.should_apply(make_ctx("hello", "text")) is False

    def test_basic_parse_and_group(self):
        stage = SearchCrunch()
        result = stage.apply(search_ctx(SAMPLE_GREP))
        assert result.skipped is False
        assert "src/auth.py" in result.content
        assert "src/models.py" in result.content
        assert "src/utils.py" in result.content

    def test_summary_header_present(self):
        stage = SearchCrunch()
        result = stage.apply(search_ctx(SAMPLE_GREP))
        assert "matches" in result.content.lower()

    def test_deduplication_of_identical_matches(self):
        stage = SearchCrunch()
        dup_search = "\n".join([
            "src/foo.py:10:same line",
            "src/foo.py:10:same line",
            "src/foo.py:10:same line",
        ])
        result = stage.apply(search_ctx(dup_search))
        # Should appear only once.
        assert result.content.count("same line") == 1

    def test_consecutive_lines_merged(self):
        stage = SearchCrunch()
        consecutive = "\n".join([
            f"src/big.py:{i}:line content {i}" for i in range(1, 8)
        ])
        result = stage.apply(search_ctx(consecutive))
        assert "omitted" in result.content or "L1" in result.content

    def test_top_n_files_limit(self):
        stage = SearchCrunch(max_files=3)
        many_files = "\n".join(
            [f"src/file_{i}.py:1:match" for i in range(10)]
        )
        result = stage.apply(search_ctx(many_files))
        assert "additional file" in result.content or "omitted" in result.content

    def test_empty_input_skipped(self):
        stage = SearchCrunch()
        result = stage.apply(search_ctx(""))
        assert result.skipped is True

    def test_non_grep_lines_handled(self):
        stage = SearchCrunch()
        mixed = "src/foo.py:1:hello\nsome random line without colon format\nsrc/bar.py:5:world"
        result = stage.apply(search_ctx(mixed))
        assert "foo.py" in result.content
        assert "bar.py" in result.content

    def test_sort_by_match_count(self):
        stage = SearchCrunch(max_files=2)
        # file_b has more matches.
        many = (
            "src/file_a.py:1:one match\n"
            + "\n".join([f"src/file_b.py:{i}:match" for i in range(1, 10)])
        )
        result = stage.apply(search_ctx(many))
        # file_b should appear before file_a.
        b_pos = result.content.find("file_b.py")
        a_pos = result.content.find("file_a.py")
        assert b_pos < a_pos

    def test_markers_set(self):
        stage = SearchCrunch()
        result = stage.apply(search_ctx(SAMPLE_GREP))
        assert len(result.markers) > 0
        assert "search_crunch:" in result.markers[0]

    def test_separator_lines_ignored(self):
        stage = SearchCrunch()
        with_sep = "src/a.py:1:hello\n--\nsrc/b.py:2:world"
        result = stage.apply(search_ctx(with_sep))
        assert result.skipped is False
        assert "a.py" in result.content
        assert "b.py" in result.content

    def test_large_result_set_truncated_per_file(self):
        stage = SearchCrunch(max_matches_per_file=5)
        many_matches = "\n".join(
            [f"src/big.py:{i}:match_{i}" for i in range(1, 51)]
        )
        result = stage.apply(search_ctx(many_matches))
        assert "more matches not shown" in result.content

    def test_single_file_single_match(self):
        stage = SearchCrunch()
        result = stage.apply(search_ctx("src/only.py:42:found it"))
        assert "only.py" in result.content
        assert "42" in result.content
        assert "found it" in result.content


# ===========================================================================
# DIFF CRUNCH TESTS
# ===========================================================================

class TestDiffCrunch:

    def test_should_apply_diff_content_type(self):
        stage = DiffCrunch()
        assert stage.should_apply(diff_ctx("--- a/file\n+++ b/file\n")) is True

    def test_should_not_apply_non_diff(self):
        stage = DiffCrunch()
        assert stage.should_apply(make_ctx("hello", "text")) is False

    def test_added_lines_preserved(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert '+import json  # added comment' in result.content
        assert '+    log.info("starting")' in result.content

    def test_removed_lines_preserved(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert '-import json' in result.content

    def test_hunk_header_preserved(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert '@@' in result.content

    def test_file_headers_preserved(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert '--- a/src/server.py' in result.content
        assert '+++ b/src/server.py' in result.content

    def test_context_lines_compressed(self):
        stage = DiffCrunch(context_keep=1)
        long_diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,15 +1,16 @@\n"
            + "\n".join([f" context_{i}" for i in range(10)])
            + "\n+new line\n"
            + "\n".join([f" context_{i}" for i in range(10, 20)])
        )
        result = stage.apply(diff_ctx(long_diff))
        assert "unchanged line" in result.content
        # Original has 20 context lines but compressed should have far fewer.
        context_lines_in_output = sum(
            1 for ln in result.content.splitlines() if ln.startswith(" context_")
        )
        assert context_lines_in_output < 20

    def test_empty_diff(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(""))
        assert result.content == ""

    def test_small_diff_no_compression(self):
        """A diff with only 1-2 context lines per block should not be truncated."""
        stage = DiffCrunch(context_keep=1)
        small = (
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,3 +1,4 @@\n"
            " only_context\n"
            "+added\n"
            " other_context\n"
        )
        result = stage.apply(diff_ctx(small))
        assert " only_context" in result.content
        assert " other_context" in result.content

    def test_large_diff_gets_summary(self):
        stage = DiffCrunch(large_diff_threshold=10)
        lines = (
            ["--- a/big.py", "+++ b/big.py", "@@ -1,100 +1,100 @@"]
            + [f"+added_{i}" for i in range(50)]
            + [f"-removed_{i}" for i in range(50)]
        )
        large_diff = "\n".join(lines)
        result = stage.apply(diff_ctx(large_diff))
        assert "Large diff summary" in result.content or len(result.markers) > 0

    def test_large_diff_rewind_store(self):
        store = RewindStore()
        stage = DiffCrunch(rewind_store=store, large_diff_threshold=5)
        lines = (
            ["--- a/big.py", "+++ b/big.py", "@@ -1,10 +1,10 @@"]
            + [f"+added_{i}" for i in range(10)]
            + [f"-removed_{i}" for i in range(10)]
        )
        large_diff = "\n".join(lines)
        result = stage.apply(diff_ctx(large_diff))
        assert has_markers(result.content)
        m = extract_markers(result.content)
        assert len(m) == 1
        retrieved = store.retrieve(m[0].hash_id)
        assert retrieved is not None
        assert "added_0" in retrieved

    def test_markers_in_result(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert len(result.markers) > 0
        assert "diff_crunch:" in result.markers[0]

    def test_no_newline_indicator_preserved(self):
        stage = DiffCrunch()
        diff_with_no_newline = (
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "\\ No newline at end of file\n"
            "+new\n"
        )
        result = stage.apply(diff_ctx(diff_with_no_newline))
        assert "\\ No newline" in result.content

    def test_git_diff_index_line_preserved(self):
        stage = DiffCrunch()
        result = stage.apply(diff_ctx(SAMPLE_DIFF))
        assert "index" in result.content

    def test_compression_reduces_token_count(self):
        """End-to-end: a diff with lots of context should come out smaller."""
        stage = DiffCrunch(context_keep=1)
        # Build a diff with large context blocks around small changes.
        context_block = "\n".join([f" ctx_line_{i}" for i in range(40)])
        diff_text = (
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,50 +1,51 @@\n"
            + context_block
            + "\n+new_feature_line\n"
            + context_block
        )
        result = stage.apply(diff_ctx(diff_text))
        assert result.compressed_tokens < result.original_tokens
        assert "+new_feature_line" in result.content


# ===========================================================================
# CROSS-MODULE / INTEGRATION TESTS
# ===========================================================================

class TestPhase3Integration:

    def test_ionizer_order_is_15(self):
        assert Ionizer.order == 15

    def test_log_crunch_order_is_16(self):
        assert LogCrunch.order == 16

    def test_search_crunch_order_is_17(self):
        assert SearchCrunch.order == 17

    def test_diff_crunch_order_is_18(self):
        assert DiffCrunch.order == 18

    def test_stage_names(self):
        assert Ionizer.name == "ionizer"
        assert LogCrunch.name == "log_crunch"
        assert SearchCrunch.name == "search_crunch"
        assert DiffCrunch.name == "diff_crunch"

    def test_all_stages_return_fusion_result(self):
        from claw_compactor.fusion.base import FusionResult
        stages_and_ctxs = [
            (Ionizer(), json_ctx(make_dict_array(20))),
            (LogCrunch(), log_ctx(SAMPLE_LOG)),
            (SearchCrunch(), search_ctx(SAMPLE_GREP)),
            (DiffCrunch(), diff_ctx(SAMPLE_DIFF)),
        ]
        for stage, ctx in stages_and_ctxs:
            result = stage.apply(ctx)
            assert isinstance(result, FusionResult), f"{stage.name} did not return FusionResult"

    def test_timed_apply_respects_should_apply(self):
        """timed_apply must skip when should_apply returns False."""
        stage = Ionizer()
        ctx = make_ctx("just text", "text")
        result = stage.timed_apply(ctx)
        assert result.skipped is True

    def test_compressed_content_is_string(self):
        """All stages must return string content."""
        stages_and_ctxs = [
            (Ionizer(), json_ctx(make_dict_array(20))),
            (LogCrunch(), log_ctx(SAMPLE_LOG)),
            (SearchCrunch(), search_ctx(SAMPLE_GREP)),
            (DiffCrunch(), diff_ctx(SAMPLE_DIFF)),
        ]
        for stage, ctx in stages_and_ctxs:
            result = stage.apply(ctx)
            assert isinstance(result.content, str), f"{stage.name} content is not a string"

    def test_shared_rewind_store_across_stages(self):
        """Multiple stages can share a single RewindStore."""
        store = RewindStore()
        ionizer = Ionizer(rewind_store=store)
        diff_crunch = DiffCrunch(rewind_store=store, large_diff_threshold=5)

        # Ionizer stores something.
        ionizer.apply(json_ctx(make_dict_array(30)))

        # DiffCrunch stores something else.
        lines = (
            ["--- a/f.py", "+++ b/f.py", "@@ -1,10 +1,10 @@"]
            + [f"+x_{i}" for i in range(10)]
            + [f"-y_{i}" for i in range(10)]
        )
        diff_crunch.apply(diff_ctx("\n".join(lines)))

        assert store.size >= 1
