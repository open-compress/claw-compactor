"""Tests for ContentDetector and Cortex FusionStage (Phase 2).

ContentDetector classifies raw text into content types (code, json, log, diff,
search, text) and optionally detects sections within mixed content.

Cortex is a FusionStage that runs ContentDetector and stores the detected
content_type (and language) into FusionResult.context_updates so downstream
stages can specialise their behaviour.

The modules under test (lib.fusion.content_detector, lib.fusion.cortex) are
Phase 2 modules and may not yet exist on disk. Imports are attempted at module
level; individual tests will fail with ImportError if the modules are missing,
which is the expected red state before implementation.
"""

import sys
from pathlib import Path

import pytest

# conftest.py already inserts the scripts directory, but be explicit for
# standalone test-file runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.content_detector import ContentDetector  # noqa: E402
from claw_compactor.fusion.cortex import Cortex  # noqa: E402
from claw_compactor.fusion.base import FusionContext, FusionResult  # noqa: E402


# ===========================================================================
# ContentDetector
# ===========================================================================

class TestContentDetectorCodeDetection:
    """Detect source code and identify the programming language."""

    def test_markdown_fence_python(self):
        text = "```python\ndef foo(): pass\n```"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "python"
        assert result.confidence >= 0.9

    def test_shebang_python(self):
        text = "#!/usr/bin/env python3\nimport sys\nprint(sys.argv)"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "python"

    def test_keywords_python(self):
        text = "def greet(name):\n    import os\n    class Foo:\n        pass"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "python"

    def test_keywords_javascript(self):
        text = "const x = 1;\nfunction hello() {}\nexport default hello;"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "javascript"

    def test_keywords_go(self):
        text = "package main\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "go"

    def test_keywords_rust(self):
        text = "use std::io;\n\nfn main() {\n    impl Foo {}\n}"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "code"
        assert result.language == "rust"


class TestContentDetectorJsonDetection:
    """Detect JSON objects and arrays."""

    def test_json_object(self):
        text = '{"key": "value", "count": 42}'
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "json"

    def test_json_array(self):
        text = "[1, 2, 3]"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "json"

    def test_invalid_json_not_classified_as_json(self):
        text = "{not json}"
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type != "json"


class TestContentDetectorSpecialFormats:
    """Detect log output, diffs, and search results."""

    def test_log_detection(self):
        text = (
            "2024-01-15 10:00:01 INFO  Starting server\n"
            "2024-01-15 10:00:02 ERROR Connection refused\n"
            "2024-01-15 10:00:03 INFO  Retrying...\n"
        )
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "log"

    def test_diff_detection(self):
        text = (
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " def foo():\n"
            "-    pass\n"
            "+    return 42\n"
        )
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "diff"

    def test_search_result_detection(self):
        text = (
            "src/foo.py:42:def greet(name):\n"
            "src/bar.py:17:    return value\n"
            "tests/test_foo.py:99:    assert result == 1\n"
        )
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "search"


class TestContentDetectorPlainText:
    """Fallback to plain text for unrecognised content."""

    def test_plain_text(self):
        text = (
            "This is a regular English paragraph about software development. "
            "It contains no special markers, no code fences, and no structured data."
        )
        detector = ContentDetector()
        result = detector.detect(text)
        assert result.content_type == "text"

    def test_empty_string_is_text(self):
        detector = ContentDetector()
        result = detector.detect("")
        assert result.content_type == "text"

    def test_short_ambiguous_text_is_text(self):
        detector = ContentDetector()
        result = detector.detect("hello")
        assert result.content_type == "text"
        assert result.confidence < 0.9


class TestContentDetectorSections:
    """detect_sections splits mixed content into typed sections."""

    def test_mixed_content_returns_multiple_sections(self):
        text = (
            "Here is some explanation text.\n\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n\n"
            "And here is more text after the code block."
        )
        detector = ContentDetector()
        sections = detector.detect_sections(text)
        # Must have at least two distinct sections: text and code
        assert len(sections) >= 2
        types = [s.content_type for s in sections]
        assert "text" in types
        assert "code" in types


# ===========================================================================
# Cortex FusionStage
# ===========================================================================

class TestCortex:
    """Cortex runs ContentDetector and records findings in context_updates."""

    def test_detects_python_code_and_sets_context_updates(self):
        cortex = Cortex()
        ctx = FusionContext(
            content="def foo():\n    import os\n    class Bar:\n        pass"
        )
        result = cortex.apply(ctx)
        assert isinstance(result, FusionResult)
        assert result.context_updates.get("content_type") == "code"

    def test_detects_json_and_sets_context_updates(self):
        cortex = Cortex()
        ctx = FusionContext(content='{"status": "ok", "count": 7}')
        result = cortex.apply(ctx)
        assert result.context_updates.get("content_type") == "json"

    def test_should_apply_false_when_content_type_already_set(self):
        cortex = Cortex()
        ctx = FusionContext(content="def foo(): pass", content_type="code")
        assert cortex.should_apply(ctx) is False

    def test_should_apply_true_when_content_type_is_text_default(self):
        cortex = Cortex()
        ctx = FusionContext(content="def foo(): pass")
        # Default content_type is "text", so Cortex should run
        assert cortex.should_apply(ctx) is True

    def test_content_unchanged_after_cortex(self):
        cortex = Cortex()
        original = "const x = 1;\nfunction hello() {}\nexport default hello;"
        ctx = FusionContext(content=original)
        result = cortex.apply(ctx)
        # Cortex is a detector only — it must not modify content
        assert result.content == original

    def test_cortex_is_fusion_stage_subclass(self):
        from claw_compactor.fusion.base import FusionStage
        assert isinstance(Cortex(), FusionStage)

    def test_cortex_sets_language_for_python_code(self):
        cortex = Cortex()
        ctx = FusionContext(content="def foo():\n    pass\nimport sys")
        result = cortex.apply(ctx)
        # language should be propagated into context_updates when detectable
        assert result.context_updates.get("language") == "python"

    def test_timed_apply_skips_when_content_type_already_set(self):
        cortex = Cortex()
        ctx = FusionContext(content='{"x": 1}', content_type="json")
        result = cortex.timed_apply(ctx)
        assert result.skipped is True
