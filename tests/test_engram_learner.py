"""Tests for EngramLearner — Engram v2 failure learning (Phase 6).

Covers:
- Error pattern classification for all 14 named patterns
- UNKNOWN classification for unmatched text
- Evidence threshold enforcement (skip rules with count < 2)
- Rule generation from FailureEvent lists
- JSONL file scanning (single and multi-file)
- Export format (MEMORY.md block)
- Edge cases: empty session dir, malformed JSONL, non-JSON lines
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.engram_learner import (
    EngramLearner,
    FailureEvent,
    CompressionRule,
    ERROR_PATTERNS,
    _MIN_EVIDENCE,
    _extract_text,
    _build_annotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def _failure(pattern: str, msg: str = "error") -> FailureEvent:
    return FailureEvent(
        pattern_name=pattern,
        raw_message=msg,
        source_file="/fake/session.jsonl",
        line_number=1,
    )


# ===========================================================================
# 1. ERROR_PATTERNS registry completeness
# ===========================================================================

class TestErrorPatternsRegistry(unittest.TestCase):
    EXPECTED_PATTERNS = [
        "FILE_NOT_FOUND", "MODULE_NOT_FOUND", "PERMISSION_DENIED",
        "TIMEOUT", "BUILD_FAILED", "TEST_FAILED", "SYNTAX_ERROR",
        "TYPE_ERROR", "IMPORT_ERROR", "CONNECTION_ERROR", "AUTH_FAILED",
        "RATE_LIMITED", "OUT_OF_MEMORY", "DISK_FULL",
    ]

    def test_all_14_patterns_present(self):
        for name in self.EXPECTED_PATTERNS:
            self.assertIn(name, ERROR_PATTERNS, f"Missing pattern: {name}")

    def test_each_pattern_has_at_least_one_regex(self):
        for name, compiled in ERROR_PATTERNS.items():
            self.assertGreater(len(compiled), 0, f"Pattern {name} has no regexes")

    def test_patterns_are_compiled(self):
        import re
        for name, compiled in ERROR_PATTERNS.items():
            for pat in compiled:
                self.assertIsInstance(pat, re.Pattern, f"{name}: expected compiled pattern")


# ===========================================================================
# 2. classify_failure — all 14 pattern types
# ===========================================================================

class TestClassifyFailure(unittest.TestCase):
    def setUp(self):
        self.learner = EngramLearner()

    def _classify(self, text: str) -> str:
        return self.learner.classify_failure({"content": text})

    def test_file_not_found(self):
        self.assertEqual(self._classify("No such file or directory: /tmp/foo.py"), "FILE_NOT_FOUND")

    def test_module_not_found(self):
        self.assertEqual(self._classify("ModuleNotFoundError: No module named 'requests'"), "MODULE_NOT_FOUND")

    def test_permission_denied(self):
        self.assertEqual(self._classify("PermissionError: [Errno 13] Permission denied"), "PERMISSION_DENIED")

    def test_timeout(self):
        self.assertEqual(self._classify("TimeoutError: the operation timed out after 30s"), "TIMEOUT")

    def test_build_failed(self):
        self.assertEqual(self._classify("Build failed with exit code 1"), "BUILD_FAILED")

    def test_test_failed(self):
        self.assertEqual(self._classify("FAILED tests/test_foo.py::test_bar"), "TEST_FAILED")

    def test_syntax_error(self):
        self.assertEqual(self._classify("SyntaxError: invalid syntax on line 42"), "SYNTAX_ERROR")

    def test_type_error(self):
        self.assertEqual(self._classify("TypeError: unsupported operand type(s) for +: 'int' and 'str'"), "TYPE_ERROR")

    def test_import_error(self):
        self.assertEqual(self._classify("ImportError: cannot import name 'foo' from 'bar'"), "IMPORT_ERROR")

    def test_connection_error(self):
        self.assertEqual(self._classify("ConnectionError: connection refused to 127.0.0.1:8080"), "CONNECTION_ERROR")

    def test_auth_failed(self):
        self.assertEqual(self._classify("401 Unauthorized: invalid credentials"), "AUTH_FAILED")

    def test_rate_limited(self):
        self.assertEqual(self._classify("429 Too Many Requests: rate limit exceeded"), "RATE_LIMITED")

    def test_out_of_memory(self):
        self.assertEqual(self._classify("MemoryError: out of memory"), "OUT_OF_MEMORY")

    def test_disk_full(self):
        self.assertEqual(self._classify("No space left on device: /var/log"), "DISK_FULL")

    def test_unknown_pattern(self):
        self.assertEqual(self._classify("Everything is fine, no errors here."), "UNKNOWN")

    def test_empty_string_is_unknown(self):
        self.assertEqual(self._classify(""), "UNKNOWN")

    def test_case_insensitive_matching(self):
        # "permission denied" in lowercase
        self.assertEqual(self._classify("ERROR: permission denied when writing config"), "PERMISSION_DENIED")


# ===========================================================================
# 3. scan_session — JSONL file scanning
# ===========================================================================

class TestScanSession(unittest.TestCase):
    def setUp(self):
        self.learner = EngramLearner()

    def test_scan_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            events = self.learner.scan_session(d)
        self.assertEqual(events, [])

    def test_scan_nonexistent_directory(self):
        events = self.learner.scan_session("/nonexistent/path/xyz")
        self.assertEqual(events, [])

    def test_scan_single_file_finds_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            _write_jsonl(p, [
                {"role": "assistant", "content": "FileNotFoundError: /tmp/missing.py"},
                {"role": "user", "content": "Try again please."},
            ])
            events = self.learner.scan_session(d)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pattern_name, "FILE_NOT_FOUND")

    def test_scan_multiple_files(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.jsonl"
            p2 = Path(d) / "b.jsonl"
            _write_jsonl(p1, [{"content": "TimeoutError: timed out"}])
            _write_jsonl(p2, [{"content": "MemoryError: out of memory"}])
            events = self.learner.scan_session(d)

        patterns = {e.pattern_name for e in events}
        self.assertIn("TIMEOUT", patterns)
        self.assertIn("OUT_OF_MEMORY", patterns)

    def test_scan_skips_non_error_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            _write_jsonl(p, [
                {"role": "user", "content": "Hello, can you help me?"},
                {"role": "assistant", "content": "Sure! Here is the answer."},
            ])
            events = self.learner.scan_session(d)

        self.assertEqual(events, [])

    def test_scan_handles_malformed_jsonl(self):
        """Malformed JSON lines should not crash the scanner."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corrupt.jsonl"
            p.write_text("not valid json\n{\"content\": \"PermissionError: denied\"}\n", encoding="utf-8")
            events = self.learner.scan_session(d)

        # The valid line should be found
        self.assertTrue(any(e.pattern_name == "PERMISSION_DENIED" for e in events))

    def test_scan_records_source_file_and_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            _write_jsonl(p, [
                {"content": "SyntaxError: invalid syntax"},
            ])
            events = self.learner.scan_session(d)

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].source_file.endswith("session.jsonl"))
        self.assertEqual(events[0].line_number, 1)

    def test_scan_nested_subdirectory(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "thread-1"
            sub.mkdir()
            p = sub / "events.jsonl"
            _write_jsonl(p, [{"content": "ECONNREFUSED: connection refused"}])
            events = self.learner.scan_session(d)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pattern_name, "CONNECTION_ERROR")


# ===========================================================================
# 4. generate_rules — evidence threshold
# ===========================================================================

class TestGenerateRules(unittest.TestCase):
    def setUp(self):
        self.learner = EngramLearner()

    def test_single_event_no_rule(self):
        failures = [_failure("FILE_NOT_FOUND")]
        rules = self.learner.generate_rules(failures)
        self.assertEqual(rules, [])

    def test_two_events_generates_rule(self):
        failures = [_failure("TIMEOUT"), _failure("TIMEOUT")]
        rules = self.learner.generate_rules(failures)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].pattern_name, "TIMEOUT")
        self.assertEqual(rules[0].evidence_count, 2)

    def test_evidence_threshold_is_min_evidence(self):
        # Exactly _MIN_EVIDENCE-1 → no rule
        failures = [_failure("BUILD_FAILED")] * (_MIN_EVIDENCE - 1)
        rules = self.learner.generate_rules(failures)
        self.assertEqual(rules, [])

    def test_multiple_patterns_each_need_threshold(self):
        failures = (
            [_failure("SYNTAX_ERROR")] * 3
            + [_failure("TYPE_ERROR")] * 1  # below threshold
            + [_failure("IMPORT_ERROR")] * 2
        )
        rules = self.learner.generate_rules(failures)
        rule_names = {r.pattern_name for r in rules}
        self.assertIn("SYNTAX_ERROR", rule_names)
        self.assertIn("IMPORT_ERROR", rule_names)
        self.assertNotIn("TYPE_ERROR", rule_names)

    def test_rules_sorted_by_evidence_descending(self):
        failures = (
            [_failure("DISK_FULL")] * 5
            + [_failure("AUTH_FAILED")] * 2
            + [_failure("RATE_LIMITED")] * 10
        )
        rules = self.learner.generate_rules(failures)
        counts = [r.evidence_count for r in rules]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_rule_contains_up_to_3_examples(self):
        failures = [
            _failure("OUT_OF_MEMORY", f"MemoryError #{i}") for i in range(10)
        ]
        rules = self.learner.generate_rules(failures)
        self.assertEqual(len(rules), 1)
        self.assertLessEqual(len(rules[0].example_messages), 3)

    def test_empty_failures_returns_empty_rules(self):
        rules = self.learner.generate_rules([])
        self.assertEqual(rules, [])

    def test_rule_description_non_empty(self):
        failures = [_failure("CONNECTION_ERROR")] * 2
        rules = self.learner.generate_rules(failures)
        self.assertTrue(rules[0].description, "Description should not be empty")

    def test_rule_suggested_annotation_non_empty(self):
        failures = [_failure("MODULE_NOT_FOUND")] * 2
        rules = self.learner.generate_rules(failures)
        self.assertTrue(rules[0].suggested_annotation, "Annotation should not be empty")


# ===========================================================================
# 5. export_rules — MEMORY.md format
# ===========================================================================

class TestExportRules(unittest.TestCase):
    def setUp(self):
        self.learner = EngramLearner()

    def test_empty_rules_returns_empty_string(self):
        self.assertEqual(self.learner.export_rules([]), "")

    def test_exported_contains_pattern_name(self):
        rule = CompressionRule(
            pattern_name="TIMEOUT",
            description="An operation timed out.",
            evidence_count=3,
            example_messages=("TimeoutError: timed out",),
            suggested_annotation="[TIMEOUT] occurred 3 times.",
        )
        md = self.learner.export_rules([rule])
        self.assertIn("TIMEOUT", md)

    def test_exported_contains_evidence_count(self):
        rule = CompressionRule(
            pattern_name="AUTH_FAILED",
            description="Auth rejected.",
            evidence_count=5,
            example_messages=(),
            suggested_annotation="[AUTH_FAILED] occurred 5 times.",
        )
        md = self.learner.export_rules([rule])
        self.assertIn("5", md)

    def test_exported_contains_examples(self):
        rule = CompressionRule(
            pattern_name="BUILD_FAILED",
            description="Build error.",
            evidence_count=2,
            example_messages=("Build failed at step compile",),
            suggested_annotation="",
        )
        md = self.learner.export_rules([rule])
        self.assertIn("Build failed at step compile", md)

    def test_exported_has_markdown_header(self):
        rule = CompressionRule(
            pattern_name="DISK_FULL",
            description="No space.",
            evidence_count=2,
            example_messages=(),
            suggested_annotation="[DISK_FULL] no space.",
        )
        md = self.learner.export_rules([rule])
        self.assertIn("##", md)

    def test_multiple_rules_all_present(self):
        rules = [
            CompressionRule("TIMEOUT", "Timed out.", 4, (), "[TIMEOUT]"),
            CompressionRule("RATE_LIMITED", "Rate limited.", 2, (), "[RATE_LIMITED]"),
        ]
        md = self.learner.export_rules(rules)
        self.assertIn("TIMEOUT", md)
        self.assertIn("RATE_LIMITED", md)


# ===========================================================================
# 6. Integration: scan → generate → export
# ===========================================================================

class TestEngramLearnerIntegration(unittest.TestCase):
    def test_full_pipeline_generates_rules_and_exports(self):
        learner = EngramLearner()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            lines = [
                {"role": "assistant", "content": "ModuleNotFoundError: No module named 'PIL'"},
                {"role": "assistant", "content": "No module named 'numpy' found"},
                {"role": "system", "content": "Build failed with exit code 2"},
                {"role": "system", "content": "Build failed: compilation error"},
            ]
            _write_jsonl(p, lines)

            failures = learner.scan_session(d)
            rules = learner.generate_rules(failures)
            md = learner.export_rules(rules)

        self.assertGreaterEqual(len(rules), 1)
        self.assertIn("MODULE_NOT_FOUND", md)
        self.assertIn("BUILD_FAILED", md)
