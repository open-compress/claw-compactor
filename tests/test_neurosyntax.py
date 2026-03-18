"""Tests for Neurosyntax code compressor FusionStage (Phase 2).

Neurosyntax applies safe, structure-preserving compression to source code:
  - removes pure comment lines (# comment, // comment, /* ... */)
  - keeps semantically significant comments (type: ignore, noqa, TODO,
    eslint-disable, pragma, etc.)
  - collapses consecutive blank lines to at most one
  - collapses multi-line Python docstrings to their opening line
  - strips trailing whitespace from every line
  - preserves all identifiers (no shortening)
  - does not touch string literals

The module under test (lib.fusion.neurosyntax) is a Phase 2 module and may
not yet exist on disk. Imports at module level will produce the expected red
state before implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.fusion.neurosyntax import Neurosyntax  # noqa: E402
from claw_compactor.fusion.base import FusionContext, FusionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compress(code: str, language: str = "python") -> str:
    """Run Neurosyntax on a code snippet and return the compressed text."""
    ns = Neurosyntax()
    ctx = FusionContext(content=code, content_type="code", language=language)
    result = ns.apply(ctx)
    return result.content


# ===========================================================================
# Comment removal
# ===========================================================================

class TestCommentRemoval:
    def test_removes_pure_python_comment_lines(self):
        code = "# this is a comment\ndef foo():\n    return 42\n"
        output = _compress(code)
        assert "# this is a comment" not in output
        assert "def foo" in output

    def test_keeps_type_ignore_comment(self):
        code = "x = foo()  # type: ignore\ndef bar(): pass\n"
        output = _compress(code)
        assert "# type: ignore" in output

    def test_keeps_noqa_comment(self):
        code = "import os  # noqa\ndef bar(): pass\n"
        output = _compress(code)
        assert "# noqa" in output

    def test_keeps_todo_comment(self):
        code = "# TODO: fix this later\ndef foo():\n    pass\n"
        output = _compress(code)
        assert "# TODO:" in output

    def test_removes_pure_js_comment_lines(self):
        code = "// this is a comment\nconst x = 1;\n"
        output = _compress(code, language="javascript")
        assert "// this is a comment" not in output
        assert "const x = 1" in output

    def test_keeps_eslint_disable_pragma(self):
        code = "// eslint-disable-next-line no-console\nconsole.log('hi');\n"
        output = _compress(code, language="javascript")
        assert "eslint-disable" in output

    def test_removes_block_comment(self):
        code = "/* this is a block comment */\nconst y = 2;\n"
        output = _compress(code, language="javascript")
        assert "this is a block comment" not in output
        assert "const y = 2" in output

    def test_keeps_important_block_comment(self):
        # pragma-style block comments should survive
        code = "/* eslint-disable */\nconst z = 3;\n"
        output = _compress(code, language="javascript")
        assert "eslint-disable" in output

    def test_mixed_comments_only_removes_pure_comment_lines(self):
        code = (
            "# pure comment — remove\n"
            "x = 1  # inline comment on real code — keep line\n"
            "def foo():\n"
            "    # another pure comment — remove\n"
            "    return x\n"
        )
        output = _compress(code)
        assert "pure comment" not in output
        assert "another pure comment" not in output
        assert "x = 1" in output
        assert "def foo" in output
        assert "return x" in output


# ===========================================================================
# Blank line collapsing
# ===========================================================================

class TestBlankLineCollapsing:
    def test_multiple_blank_lines_collapsed_to_one(self):
        code = "def foo():\n    pass\n\n\n\ndef bar():\n    pass\n"
        output = _compress(code)
        # More than one consecutive blank line must not appear
        assert "\n\n\n" not in output

    def test_single_blank_line_preserved(self):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        output = _compress(code)
        assert "def foo" in output
        assert "def bar" in output


# ===========================================================================
# Docstring collapsing
# ===========================================================================

class TestDocstringCollapsing:
    def test_multiline_docstring_collapsed_to_first_line(self):
        code = (
            'def foo():\n'
            '    """This is the summary line.\n'
            '\n'
            '    This is extra detail that can be dropped.\n'
            '    More detail here.\n'
            '    """\n'
            '    return 42\n'
        )
        output = _compress(code)
        assert "This is the summary line" in output
        assert "This is extra detail" not in output
        assert "return 42" in output


# ===========================================================================
# Import preservation
# ===========================================================================

class TestImportPreservation:
    def test_consecutive_import_lines_preserved(self):
        code = "import os\nimport sys\nimport json\n\ndef foo(): pass\n"
        output = _compress(code)
        assert "import os" in output
        assert "import sys" in output
        assert "import json" in output


# ===========================================================================
# Whitespace
# ===========================================================================

class TestTrailingWhitespace:
    def test_trailing_whitespace_stripped(self):
        code = "def foo():   \n    return 1   \n"
        output = _compress(code)
        for line in output.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace on line: {line!r}"


# ===========================================================================
# Safety: no identifier shortening, no string mutation
# ===========================================================================

class TestSafetyGuarantees:
    def test_string_literals_untouched(self):
        # A string that happens to contain import-like text must not be altered
        code = 'x = "import os; import sys"\ndef foo(): pass\n'
        output = _compress(code)
        assert '"import os; import sys"' in output

    def test_no_identifier_shortening(self):
        code = (
            "def calculate_average_value(numbers):\n"
            "    total_sum = sum(numbers)\n"
            "    class DataProcessor:\n"
            "        pass\n"
            "    return total_sum / len(numbers)\n"
        )
        output = _compress(code)
        assert "calculate_average_value" in output
        assert "total_sum" in output
        assert "DataProcessor" in output


# ===========================================================================
# should_apply gating
# ===========================================================================

class TestShouldApply:
    def test_should_apply_false_for_text_content_type(self):
        ns = Neurosyntax()
        ctx = FusionContext(content="some plain text", content_type="text")
        assert ns.should_apply(ctx) is False

    def test_should_apply_true_for_code_content_type(self):
        ns = Neurosyntax()
        ctx = FusionContext(content="def foo(): pass", content_type="code")
        assert ns.should_apply(ctx) is True

    def test_should_apply_false_for_json_content_type(self):
        ns = Neurosyntax()
        ctx = FusionContext(content='{"x": 1}', content_type="json")
        assert ns.should_apply(ctx) is False


# ===========================================================================
# Token counts
# ===========================================================================

class TestTokenCounts:
    def test_result_has_positive_original_tokens(self):
        code = "# comment\ndef foo():\n    return 42\n"
        ns = Neurosyntax()
        ctx = FusionContext(content=code, content_type="code")
        result = ns.apply(ctx)
        assert result.original_tokens > 0

    def test_compressed_tokens_lte_original_tokens(self):
        code = (
            "# comment line 1\n"
            "# comment line 2\n"
            "# comment line 3\n"
            "def foo():\n"
            "    return 42\n"
        )
        ns = Neurosyntax()
        ctx = FusionContext(content=code, content_type="code")
        result = ns.apply(ctx)
        assert result.compressed_tokens <= result.original_tokens


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_code_returns_empty_result(self):
        output = _compress("")
        assert output == ""

    def test_neurosyntax_returns_fusion_result(self):
        ns = Neurosyntax()
        ctx = FusionContext(content="def foo(): pass\n", content_type="code")
        result = ns.apply(ctx)
        assert isinstance(result, FusionResult)

    def test_is_fusion_stage_subclass(self):
        from claw_compactor.fusion.base import FusionStage
        assert isinstance(Neurosyntax(), FusionStage)


# ===========================================================================
# Real-world samples
# ===========================================================================

class TestRealWorldSamples:
    def test_realistic_python_function(self):
        code = (
            "# Module-level comment about this file\n"
            "import os\n"
            "import sys\n"
            "from typing import List\n"
            "\n"
            "\n"
            "# Helper utility\n"
            "def process_items(items: List[str], max_count: int = 100) -> List[str]:\n"
            '    """Process a list of items, filtering and transforming them.\n'
            "\n"
            "    This function filters items based on length and applies\n"
            "    a transformation to each qualifying item.\n"
            '    """\n'
            "    # filter step\n"
            "    filtered = [item for item in items if len(item) <= max_count]\n"
            "    result = []\n"
            "    for item in filtered:\n"
            "        # normalise each item\n"
            "        normalised = item.strip().lower()\n"
            "        if normalised:  # noqa: SIM102\n"
            "            result.append(normalised)\n"
            "    return result\n"
            "\n"
            "\n"
            "def main():\n"
            "    data = ['Hello ', ' World', 'foo']\n"
            "    print(process_items(data))\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        output = _compress(code)
        # Structure preserved
        assert "def process_items" in output
        assert "def main" in output
        assert "import os" in output
        # Pure comments removed
        assert "Module-level comment" not in output
        assert "Helper utility" not in output
        assert "filter step" not in output
        assert "normalise each item" not in output
        # Important comment kept
        assert "noqa" in output
        # Docstring summary kept, details dropped
        assert "Process a list of items" in output
        assert "This function filters" not in output
        # No triple consecutive blank lines
        assert "\n\n\n" not in output

    def test_realistic_javascript_module(self):
        code = (
            "// Module: user authentication helpers\n"
            "// Author: dev team\n"
            "\n"
            "/* eslint-disable no-console */\n"
            "\n"
            "const MAX_RETRIES = 3;\n"
            "\n"
            "// Validate email format\n"
            "function validateEmail(email) {\n"
            "    // Simple regex check\n"
            "    const re = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n"
            "    return re.test(email);\n"
            "}\n"
            "\n"
            "// Authenticate user with retries\n"
            "async function authenticateUser(credentials) {\n"
            "    let attempt = 0;\n"
            "    while (attempt < MAX_RETRIES) {\n"
            "        attempt++;\n"
            "        const result = await tryLogin(credentials);\n"
            "        if (result.ok) return result;\n"
            "    }\n"
            "    return null;\n"
            "}\n"
            "\n"
            "export { validateEmail, authenticateUser };\n"
        )
        output = _compress(code, language="javascript")
        # Key identifiers preserved
        assert "validateEmail" in output
        assert "authenticateUser" in output
        assert "MAX_RETRIES" in output
        assert "export" in output
        # Pure comments removed
        assert "Module: user authentication" not in output
        assert "Author:" not in output
        assert "Validate email format" not in output
        assert "Simple regex check" not in output
        assert "Authenticate user with retries" not in output
        # Pragma kept
        assert "eslint-disable" in output


# ===========================================================================
# Integration: Cortex + Neurosyntax in FusionPipeline
# ===========================================================================

def test_cortex_neurosyntax_pipeline():
    from claw_compactor.fusion import FusionPipeline, FusionContext
    from claw_compactor.fusion.cortex import Cortex
    from claw_compactor.fusion.neurosyntax import Neurosyntax

    pipeline = FusionPipeline([Cortex(), Neurosyntax()])
    ctx = FusionContext(content="```python\n# A pure comment\ndef foo():\n    return 42\n```")
    result = pipeline.run(ctx)
    # The function definition must survive
    assert "def foo" in result.content
    # The pure comment should be stripped by Neurosyntax
    assert "A pure comment" not in result.content
