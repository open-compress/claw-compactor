"""Tests for StructuralCollapse FusionStage.

Covers:
  - Import block collapse: Python, JS/TS, Java
  - Repeated assertion / log line collapse
  - Mixed content handling
  - Stats accuracy (markers)
  - should_apply gating
  - Token count accuracy
  - Edge cases

The module under test lives at lib/fusion/structural_collapse.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.structural_collapse import (
    StructuralCollapse,
    _extract_template,
    _find_import_blocks,
    _find_repeated_runs,
    _apply_collapse,
    _format_import_summary,
    _format_repeated_summary,
)
from claw_compactor.fusion.base import FusionContext, FusionResult, FusionStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(content: str, content_type: str = "code") -> FusionContext:
    return FusionContext(content=content, content_type=content_type)


def _collapse(content: str, content_type: str = "code") -> FusionResult:
    stage = StructuralCollapse()
    ctx = _ctx(content, content_type)
    return stage.apply(ctx)


def _lines(text: str) -> list[str]:
    return text.splitlines()


# ===========================================================================
# Stage identity
# ===========================================================================

class TestStageIdentity:
    def test_name(self):
        assert StructuralCollapse.name == "structural_collapse"

    def test_order(self):
        assert StructuralCollapse.order == 20

    def test_is_fusion_stage_subclass(self):
        assert isinstance(StructuralCollapse(), FusionStage)

    def test_apply_returns_fusion_result(self):
        content = "\n".join([f"import mod{i}" for i in range(15)])
        result = _collapse(content)
        assert isinstance(result, FusionResult)


# ===========================================================================
# should_apply gating
# ===========================================================================

class TestShouldApply:
    def test_applies_to_code_with_enough_lines(self):
        stage = StructuralCollapse()
        ctx = _ctx("def foo(): pass\n" * 12, "code")
        assert stage.should_apply(ctx) is True

    def test_applies_to_text_with_enough_lines(self):
        stage = StructuralCollapse()
        ctx = _ctx("some line\n" * 15, "text")
        assert stage.should_apply(ctx) is True

    def test_skips_log_content_type(self):
        stage = StructuralCollapse()
        ctx = FusionContext(content="some line\n" * 20, content_type="log")
        assert stage.should_apply(ctx) is False

    def test_skips_json_content_type(self):
        stage = StructuralCollapse()
        ctx = FusionContext(content='{"a":1}\n' * 20, content_type="json")
        assert stage.should_apply(ctx) is False

    def test_skips_when_fewer_than_10_lines(self):
        stage = StructuralCollapse()
        short = "\n".join([f"import x{i}" for i in range(5)])
        ctx = _ctx(short, "code")
        assert stage.should_apply(ctx) is False

    def test_applies_when_exactly_10_newlines(self):
        # 10 newlines = 11 lines after split, but count('\n') == 10
        stage = StructuralCollapse()
        content = "\n".join(["line"] * 11)  # 10 newlines
        ctx = _ctx(content, "code")
        assert stage.should_apply(ctx) is True


# ===========================================================================
# Python import collapse
# ===========================================================================

class TestPythonImportCollapse:
    def _py_imports(self, names: list[str]) -> str:
        return "\n".join([f"import {n}" for n in names])

    def test_collapses_3_or_more_python_imports(self):
        code = "\n".join([
            "import os",
            "import sys",
            "import json",
            "import logging",
        ])
        result = _collapse(code + "\n" * 8)  # pad to meet min-lines
        assert "[imports:" in result.content

    def test_python_import_summary_contains_module_names(self):
        imports = [
            "import os",
            "import sys",
            "import json",
            "import logging",
            "from pathlib import Path",
            "from typing import Dict, List, Optional",
        ]
        padding = ["x = 1"] * 6
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        summary_line = next(
            (l for l in result.content.splitlines() if l.startswith("[imports:")), None
        )
        assert summary_line is not None
        assert "os" in summary_line
        assert "sys" in summary_line
        assert "json" in summary_line
        assert "Path" in summary_line

    def test_python_from_import_names_extracted(self):
        lines = [
            "from pathlib import Path",
            "from typing import Dict, List, Optional",
            "from collections import defaultdict",
            "x = 1",
        ] + ["y = 2"] * 8
        code = "\n".join(lines)
        result = _collapse(code, "code")
        assert "Path" in result.content or "[imports:" in result.content

    def test_only_3_imports_collapsed(self):
        imports = ["import os", "import sys", "import json"]
        padding = ["pass"] * 9
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content

    def test_2_imports_not_collapsed(self):
        imports = ["import os", "import sys"]
        padding = ["x = 1"] * 10
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        # 2 imports should not be collapsed
        assert "import os" in result.content
        assert "import sys" in result.content

    def test_full_spec_example(self):
        """Reproduce the exact spec example."""
        imports = [
            "import os",
            "import sys",
            "import json",
            "import logging",
            "from pathlib import Path",
            "from typing import Dict, List, Optional",
        ]
        # Add padding lines to pass min_lines gate
        padding = ["x = 1"] * 6
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content
        # The 6 import lines should be replaced by 1 summary line
        orig_lines = len(imports) + len(padding)
        new_lines = len([l for l in result.content.splitlines() if l.strip()])
        # Should be significantly shorter
        assert new_lines < orig_lines

    def test_non_consecutive_imports_each_block_handled(self):
        """Two import blocks separated by code should each be collapsed."""
        block1 = ["import os", "import sys", "import json"]
        middle = ["x = 1", "y = 2"]
        block2 = ["import logging", "import re", "import pathlib"]
        padding = ["pass"] * 4
        code = "\n".join(block1 + middle + block2 + padding)
        result = _collapse(code, "code")
        # Both blocks collapsed → two [imports:...] summaries
        summary_count = sum(1 for l in result.content.splitlines() if l.startswith("[imports:"))
        assert summary_count == 2


# ===========================================================================
# JavaScript / TypeScript import collapse
# ===========================================================================

class TestJSImportCollapse:
    def test_js_esm_imports_collapsed(self):
        imports = [
            "import React from 'react'",
            "import { useState, useEffect } from 'react'",
            "import { BrowserRouter } from 'react-router-dom'",
            "import axios from 'axios'",
        ]
        padding = ["const x = 1;"] * 8
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content

    def test_js_require_imports_collapsed(self):
        imports = [
            "const fs = require('fs')",
            "const path = require('path')",
            "const http = require('http')",
            "const crypto = require('crypto')",
        ]
        padding = ["const x = 1;"] * 8
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content

    def test_js_named_imports_names_in_summary(self):
        imports = [
            "import { alpha } from './alpha'",
            "import { beta } from './beta'",
            "import { gamma } from './gamma'",
        ]
        padding = ["let x = 0;"] * 9
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        summary = next(
            (l for l in result.content.splitlines() if l.startswith("[imports:")), None
        )
        assert summary is not None
        assert "alpha" in summary
        assert "beta" in summary
        assert "gamma" in summary

    def test_ts_imports_collapsed(self):
        imports = [
            "import { Component } from '@angular/core'",
            "import { OnInit } from '@angular/core'",
            "import { HttpClient } from '@angular/common/http'",
            "import { Router } from '@angular/router'",
        ]
        padding = ["export class Foo {}"] * 8
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content


# ===========================================================================
# Java import collapse
# ===========================================================================

class TestJavaImportCollapse:
    def test_java_imports_collapsed(self):
        imports = [
            "import java.util.List;",
            "import java.util.ArrayList;",
            "import java.util.HashMap;",
            "import java.io.IOException;",
        ]
        padding = ["public class Foo {}"] * 8
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content

    def test_java_import_simple_names_in_summary(self):
        imports = [
            "import java.util.List;",
            "import java.util.Map;",
            "import java.io.File;",
        ]
        padding = ["class X {}"] * 9
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        summary = next(
            (l for l in result.content.splitlines() if l.startswith("[imports:")), None
        )
        assert summary is not None
        # Simple class names should appear
        assert "List" in summary or "Map" in summary or "File" in summary

    def test_java_wildcard_import(self):
        imports = [
            "import java.util.*;",
            "import java.io.*;",
            "import java.nio.file.*;",
        ]
        padding = ["public class App {}"] * 9
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content


# ===========================================================================
# Repeated line collapse
# ===========================================================================

class TestRepeatedLineCollapse:
    def test_collapses_5_similar_assertions(self):
        """Spec example: 5 assertEqual lines → first + summary + last."""
        assertions = [
            'self.assertEqual(result["a"], "1")',
            'self.assertEqual(result["b"], "2")',
            'self.assertEqual(result["c"], "3")',
            'self.assertEqual(result["d"], "4")',
            'self.assertEqual(result["e"], "5")',
        ]
        padding = ["pass"] * 7
        code = "\n".join(padding + assertions)
        result = _collapse(code, "code")
        lines = result.content.splitlines()
        # First assertion kept
        assert 'self.assertEqual(result["a"], "1")' in lines
        # Last assertion kept
        assert 'self.assertEqual(result["e"], "5")' in lines
        # Summary line present
        assert any("[..." in l for l in lines)

    def test_summary_counts_middle_lines(self):
        """With 5 lines, summary should say 3 similar lines."""
        assertions = [
            'self.assertEqual(result["a"], "1")',
            'self.assertEqual(result["b"], "2")',
            'self.assertEqual(result["c"], "3")',
            'self.assertEqual(result["d"], "4")',
            'self.assertEqual(result["e"], "5")',
        ]
        padding = ["pass"] * 7
        code = "\n".join(padding + assertions)
        result = _collapse(code, "code")
        summary = next(
            (l for l in result.content.splitlines() if "[..." in l), None
        )
        assert summary is not None
        assert "3" in summary  # 5 - 2 = 3 middle lines

    def test_collapses_repeated_log_lines(self):
        logs = [
            'logger.info("Processing item %s", item_1)',
            'logger.info("Processing item %s", item_2)',
            'logger.info("Processing item %s", item_3)',
            'logger.info("Processing item %s", item_4)',
            'logger.info("Processing item %s", item_5)',
        ]
        padding = ["x = 1"] * 7
        code = "\n".join(padding + logs)
        result = _collapse(code, "code")
        assert any("[..." in l for l in result.content.splitlines())
        # First and last preserved
        assert "item_1" in result.content
        assert "item_5" in result.content

    def test_exactly_3_repeated_lines_collapses(self):
        lines_content = [
            'assert response["x"] == expected_x',
            'assert response["y"] == expected_y',
            'assert response["z"] == expected_z',
        ]
        padding = ["setup()"] * 9
        code = "\n".join(padding + lines_content)
        result = _collapse(code, "code")
        assert any("[..." in l for l in result.content.splitlines())

    def test_only_2_repeated_lines_not_collapsed(self):
        lines_content = [
            'self.assertEqual(result["a"], "1")',
            'self.assertEqual(result["b"], "2")',
        ]
        padding = ["def test_foo(self):"] + ["pass"] * 10
        code = "\n".join(padding + lines_content)
        result = _collapse(code, "code")
        # Both lines must appear — no summary
        assert 'self.assertEqual(result["a"], "1")' in result.content
        assert 'self.assertEqual(result["b"], "2")' in result.content
        assert not any("[..." in l for l in result.content.splitlines())

    def test_repeated_env_config_lines_collapsed(self):
        env_lines = [
            'ENV_VAR_HOST = os.environ.get("HOST", "localhost")',
            'ENV_VAR_PORT = os.environ.get("PORT", "8080")',
            'ENV_VAR_USER = os.environ.get("USER", "admin")',
            'ENV_VAR_PASS = os.environ.get("PASS", "secret")',
            'ENV_VAR_DB   = os.environ.get("DB",   "mydb")',
        ]
        padding = ["# config"] + ["import os"] * 2 + [""] + ["x = 1"] * 5
        code = "\n".join(padding + env_lines)
        result = _collapse(code, "code")
        assert any("[..." in l for l in result.content.splitlines())

    def test_non_repeated_code_lines_preserved(self):
        code_lines = [
            "def foo():",
            "    x = 1",
            "    y = x + 2",
            "    return y",
        ]
        padding = ["pass"] * 8
        code = "\n".join(padding + code_lines)
        result = _collapse(code, "code")
        for line in code_lines:
            assert line in result.content


# ===========================================================================
# Mixed content handling
# ===========================================================================

class TestMixedContent:
    def test_import_block_followed_by_repeated_assertions(self):
        """Import block and assertion block both compressed in same input."""
        imports = [
            "import os",
            "import sys",
            "import json",
            "import logging",
        ]
        assertions = [
            'self.assertEqual(res["a"], 1)',
            'self.assertEqual(res["b"], 2)',
            'self.assertEqual(res["c"], 3)',
            'self.assertEqual(res["d"], 4)',
        ]
        code = "\n".join(imports + [""] + ["def test_x(self):"] + assertions)
        result = _collapse(code, "code")
        assert "[imports:" in result.content
        assert any("[..." in l for l in result.content.splitlines())

    def test_code_between_import_blocks_preserved(self):
        imports = ["import os", "import sys", "import json"]
        middle = ["x = setup()", "y = configure()"]
        imports2 = ["import re", "import pathlib", "import shutil"]
        padding = ["pass"] * 4
        code = "\n".join(imports + middle + imports2 + padding)
        result = _collapse(code, "code")
        assert "x = setup()" in result.content
        assert "y = configure()" in result.content

    def test_text_content_type_repeated_lines(self):
        log_lines = [
            "ERROR: connection timeout to host-01",
            "ERROR: connection timeout to host-02",
            "ERROR: connection timeout to host-03",
            "ERROR: connection timeout to host-04",
        ]
        padding = ["System startup"] * 8
        code = "\n".join(padding + log_lines)
        result = _collapse(code, "text")
        assert any("[..." in l for l in result.content.splitlines())

    def test_unique_lines_all_preserved(self):
        """Content with no patterns should pass through unchanged."""
        lines_content = [
            "The quick brown fox",
            "jumped over the lazy dog",
            "while the cat sat on the mat",
            "and watched the birds fly by",
            "in the clear blue sky above",
            "the rolling green hills",
            "of the English countryside",
            "on a warm summer afternoon",
            "as the sun began to set",
            "painting the sky orange",
            "and red and purple",
        ]
        code = "\n".join(lines_content)
        result = _collapse(code, "text")
        for line in lines_content:
            assert line in result.content

    def test_trailing_newline_preserved(self):
        imports = ["import os", "import sys", "import json"] + ["x = 1"] * 9
        code = "\n".join(imports) + "\n"
        result = _collapse(code, "code")
        assert result.content.endswith("\n")


# ===========================================================================
# Stats accuracy (markers)
# ===========================================================================

class TestStatsAccuracy:
    def test_marker_present_when_imports_collapsed(self):
        imports = ["import os", "import sys", "import json"] + ["pass"] * 9
        code = "\n".join(imports)
        result = _collapse(code, "code")
        assert any("structural_collapse:imports" in m for m in result.markers)

    def test_marker_present_when_repeated_runs_collapsed(self):
        assertions = [
            'self.assertEqual(x["a"], 1)',
            'self.assertEqual(x["b"], 2)',
            'self.assertEqual(x["c"], 3)',
        ]
        padding = ["pass"] * 9
        code = "\n".join(padding + assertions)
        result = _collapse(code, "code")
        assert any("structural_collapse:repeated" in m for m in result.markers)

    def test_no_markers_when_nothing_collapsed(self):
        lines_content = ["line_{}".format(i) for i in range(12)]
        code = "\n".join(lines_content)
        result = _collapse(code, "code")
        # No import blocks, no repeated templates
        import_markers = [m for m in result.markers if "imports" in m]
        repeated_markers = [m for m in result.markers if "repeated" in m]
        assert not import_markers
        assert not repeated_markers

    def test_lines_marker_shows_reduction(self):
        imports = [
            "import os", "import sys", "import json",
            "import logging", "import re", "import pathlib",
        ]
        padding = ["pass"] * 6
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        lines_markers = [m for m in result.markers if "lines:" in m]
        assert lines_markers  # at least one line-count marker
        marker = lines_markers[0]
        # Parse "X->Y" and verify Y < X
        parts = marker.split(":")[-1]
        before, after = parts.split("->")
        assert int(after) < int(before)

    def test_original_tokens_positive(self):
        code = "\n".join(["import os", "import sys", "import json"] + ["x = 1"] * 9)
        result = _collapse(code, "code")
        assert result.original_tokens > 0

    def test_compressed_tokens_lte_original_on_compressible_input(self):
        imports = [f"import module_{i}" for i in range(10)]
        padding = ["pass"] * 3
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert result.compressed_tokens <= result.original_tokens


# ===========================================================================
# Template extraction unit tests
# ===========================================================================

class TestExtractTemplate:
    def test_replaces_quoted_strings(self):
        t = _extract_template('self.assertEqual(result["a"], "1")')
        assert '"a"' not in t
        assert '"1"' not in t
        assert '<VAR>' in t

    def test_replaces_numbers(self):
        t = _extract_template('x = process(42, 3.14)')
        assert '42' not in t
        assert '3.14' not in t
        assert '<VAR>' in t

    def test_identical_structure_same_template(self):
        t1 = _extract_template('self.assertEqual(result["a"], "1")')
        t2 = _extract_template('self.assertEqual(result["b"], "2")')
        t3 = _extract_template('self.assertEqual(result["c"], "3")')
        assert t1 == t2 == t3

    def test_different_structure_different_template(self):
        t1 = _extract_template('self.assertEqual(result["a"], "1")')
        t2 = _extract_template('self.assertIn(result["a"], expected)')
        assert t1 != t2

    def test_preserves_structural_keywords(self):
        t = _extract_template('self.assertEqual(result["key"], "value")')
        assert 'self.assertEqual' in t


# ===========================================================================
# Find import blocks unit tests
# ===========================================================================

class TestFindImportBlocks:
    def test_finds_python_block(self):
        lines = ["import os", "import sys", "import json"]
        blocks = _find_import_blocks(lines)
        assert len(blocks) == 1
        assert blocks[0].start == 0
        assert blocks[0].end == 2
        assert blocks[0].lang == 'python'

    def test_finds_two_separated_blocks(self):
        lines = [
            "import os", "import sys", "import json",
            "x = 1",
            "import re", "import pathlib", "import shutil",
        ]
        blocks = _find_import_blocks(lines)
        assert len(blocks) == 2

    def test_does_not_find_block_of_2(self):
        lines = ["import os", "import sys"]
        blocks = _find_import_blocks(lines)
        assert len(blocks) == 0

    def test_java_block_found(self):
        lines = [
            "import java.util.List;",
            "import java.util.Map;",
            "import java.io.File;",
        ]
        blocks = _find_import_blocks(lines)
        assert len(blocks) == 1
        assert blocks[0].lang == 'java'

    def test_names_accumulated(self):
        lines = ["import os", "import sys", "import json"]
        blocks = _find_import_blocks(lines)
        assert "os" in blocks[0].names
        assert "sys" in blocks[0].names
        assert "json" in blocks[0].names


# ===========================================================================
# Find repeated runs unit tests
# ===========================================================================

class TestFindRepeatedRuns:
    def test_finds_run_of_5(self):
        lines = [
            'self.assertEqual(result["a"], "1")',
            'self.assertEqual(result["b"], "2")',
            'self.assertEqual(result["c"], "3")',
            'self.assertEqual(result["d"], "4")',
            'self.assertEqual(result["e"], "5")',
        ]
        runs = _find_repeated_runs(lines)
        assert len(runs) == 1
        assert runs[0].count == 5
        assert runs[0].start == 0
        assert runs[0].end == 4

    def test_run_of_3_found(self):
        lines = [
            'assert x["a"] == 1',
            'assert x["b"] == 2',
            'assert x["c"] == 3',
        ]
        runs = _find_repeated_runs(lines)
        assert len(runs) == 1
        assert runs[0].count == 3

    def test_run_of_2_not_found(self):
        lines = [
            'self.assertEqual(x["a"], 1)',
            'self.assertEqual(x["b"], 2)',
        ]
        runs = _find_repeated_runs(lines)
        assert len(runs) == 0

    def test_two_separate_runs(self):
        lines = [
            'assertEqual(a["x"], 1)',
            'assertEqual(a["y"], 2)',
            'assertEqual(a["z"], 3)',
            'do_something()',
            'log.info("event_a happened")',
            'log.info("event_b happened")',
            'log.info("event_c happened")',
        ]
        runs = _find_repeated_runs(lines)
        assert len(runs) == 2


# ===========================================================================
# Apply collapse integration
# ===========================================================================

class TestApplyCollapse:
    def test_import_block_replaced_by_single_summary(self):
        lines = [
            "import os",
            "import sys",
            "import json",
            "import logging",
            "x = 1",
        ]
        output, stats = _apply_collapse(lines)
        assert stats.import_blocks_collapsed == 1
        import_lines = [l for l in output if l.startswith("[imports:")]
        assert len(import_lines) == 1

    def test_repeated_run_replaced_by_3_lines(self):
        lines = [
            "prefix_code_line_here = setup()",
            'self.assertEqual(result["a"], "1")',
            'self.assertEqual(result["b"], "2")',
            'self.assertEqual(result["c"], "3")',
            'self.assertEqual(result["d"], "4")',
            'self.assertEqual(result["e"], "5")',
        ]
        output, stats = _apply_collapse(lines)
        assert stats.repeated_runs_collapsed == 1
        # Exactly 3 lines replace 5: first, summary, last
        summary_lines = [l for l in output if "[..." in l]
        assert len(summary_lines) == 1

    def test_lines_after_less_than_lines_before_on_compression(self):
        lines = ["import os", "import sys", "import json", "import re", "x = 1"]
        output, stats = _apply_collapse(lines)
        assert stats.lines_after <= stats.lines_before

    def test_no_collapse_on_unique_lines(self):
        lines = ["alpha = 1", "beta = 2", "gamma = 3"]
        output, stats = _apply_collapse(lines)
        assert stats.import_blocks_collapsed == 0
        assert stats.repeated_runs_collapsed == 0
        assert output == lines


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_content(self):
        stage = StructuralCollapse()
        ctx = _ctx("", "code")
        # should_apply returns False for empty (< 10 lines), but apply still works
        result = stage.apply(ctx)
        assert isinstance(result, FusionResult)
        assert result.content == ""

    def test_single_line_content(self):
        result = _collapse("import os", "code")
        assert "import os" in result.content

    def test_large_import_block_truncates_names(self):
        imports = [f"import module_{i:02d}" for i in range(25)]
        padding = ["pass"] * 5
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        summary = next(
            (l for l in result.content.splitlines() if l.startswith("[imports:")), None
        )
        assert summary is not None
        # Truncation marker present
        assert "+{0}more".format(25 - 20) in summary or "more" in summary

    def test_import_names_deduplication_not_required_but_summary_present(self):
        """Duplicate module names in summary are acceptable — just verify summary exists."""
        imports = [
            "from os import path",
            "from os import getcwd",
            "from os import listdir",
        ]
        padding = ["x = 1"] * 9
        code = "\n".join(imports + padding)
        result = _collapse(code, "code")
        assert "[imports:" in result.content

    def test_content_unchanged_when_not_compressible(self):
        """Unique code lines should be preserved verbatim."""
        code_lines = [
            "def alpha():",
            "    return 1",
            "def beta():",
            "    return 2",
            "def gamma():",
            "    return 3",
            "def delta():",
            "    return 4",
            "def epsilon():",
            "    return 5",
            "x = alpha() + beta()",
        ]
        code = "\n".join(code_lines)
        result = _collapse(code, "code")
        for func in ["alpha", "beta", "gamma", "delta", "epsilon"]:
            assert func in result.content

    def test_repeated_run_first_line_exact(self):
        """The first line of a repeated run must appear verbatim in output."""
        first = 'logger.warning("Retry attempt %d for job_%s", 1, job_a)'
        assertions = [
            first,
            'logger.warning("Retry attempt %d for job_%s", 2, job_b)',
            'logger.warning("Retry attempt %d for job_%s", 3, job_c)',
            'logger.warning("Retry attempt %d for job_%s", 4, job_d)',
        ]
        padding = ["setup_logging()"] * 8
        code = "\n".join(padding + assertions)
        result = _collapse(code, "code")
        assert first in result.content

    def test_repeated_run_last_line_exact(self):
        """The last line of a repeated run must appear verbatim in output."""
        last = 'logger.warning("Retry attempt %d for job_%s", 4, job_d)'
        assertions = [
            'logger.warning("Retry attempt %d for job_%s", 1, job_a)',
            'logger.warning("Retry attempt %d for job_%s", 2, job_b)',
            'logger.warning("Retry attempt %d for job_%s", 3, job_c)',
            last,
        ]
        padding = ["setup_logging()"] * 8
        code = "\n".join(padding + assertions)
        result = _collapse(code, "code")
        assert last in result.content

    def test_pipeline_integration(self):
        """StructuralCollapse integrates correctly with FusionPipeline."""
        from claw_compactor.fusion.pipeline import FusionPipeline

        imports = [
            "import os",
            "import sys",
            "import json",
            "import logging",
            "import re",
        ]
        padding = [
            "def main():",
            "    pass",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
            "x = 1",
        ]
        code = "\n".join(imports + padding)
        pipeline = FusionPipeline([StructuralCollapse()])
        ctx = FusionContext(content=code, content_type="code")
        pipeline_result = pipeline.run(ctx)
        assert "[imports:" in pipeline_result.content
