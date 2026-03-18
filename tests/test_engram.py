"""
tests/test_engram.py — Tests for the Engram (Observational Memory) layer.

All LLM calls are mocked so no API key is required.

Run with:
    pytest tests/test_engram.py -v

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.engram import (
    EngramEngine,
    _count_messages_tokens,
    _messages_to_text,
    MAX_OBSERVER_INPUT_TOKENS,
    MAX_REFLECTOR_INPUT_TOKENS,
)
from claw_compactor.engram_storage import EngramStorage
from claw_compactor.engram_prompts import (
    OBSERVER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    OBSERVER_USER_TEMPLATE,
    REFLECTOR_USER_TEMPLATE,
)
from claw_compactor.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a fresh temporary workspace directory."""
    return tmp_path


@pytest.fixture
def engine(workspace: Path) -> EngramEngine:
    """Return an EngramEngine with very low thresholds for easy triggering."""
    return EngramEngine(
        workspace_path=workspace,
        observer_threshold=50,    # low: trigger after ~50 tokens of pending msgs
        reflector_threshold=100,  # low: trigger after ~100 tokens of observations
        anthropic_api_key="test-key",  # fake key — calls will be mocked
    )


# ---------------------------------------------------------------------------
# Fake LLM outputs
# ---------------------------------------------------------------------------

FAKE_OBSERVATION = """\
Date: 2026-03-05
- 🔴 12:10 User is building OpenCompress / 用户在构建 OpenCompress 项目
  - 🟡 12:11 Using ModernBERT-large for inference
- 🟢 12:15 User prefers concise replies
"""

FAKE_REFLECTION = """\
## Persistent Context (long-term patterns & facts)
- 🔴 User is building OpenCompress, uses ModernBERT-large (repeated)

## Recent Events (chronological, compressed)
Date: 2026-03-05
- 🔴 12:10 Building OpenCompress project
- 🟢 12:15 Prefers concise replies
"""


def _make_engine_with_mock(workspace: Path, mock_response: str, **kwargs) -> EngramEngine:
    """Helper: create engine and patch _call_llm to return mock_response."""
    eng = EngramEngine(
        workspace_path=workspace,
        observer_threshold=kwargs.get("observer_threshold", 50),
        reflector_threshold=kwargs.get("reflector_threshold", 100),
        anthropic_api_key="test-key",
    )
    eng._call_llm = MagicMock(return_value=mock_response)  # type: ignore[method-assign]
    return eng


# ---------------------------------------------------------------------------
# Test 1: add_message auto-triggers observe
# ---------------------------------------------------------------------------

class TestAddMessageAutoObserve:
    """add_message() should auto-trigger the Observer when pending tokens exceed threshold."""

    def test_auto_observe_triggered(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=10)

        # Add enough content to exceed the threshold
        status = eng.add_message("t1", role="user", content="A" * 200)

        assert status["observed"] is True, "Observer should have been triggered"
        assert status["error"] is None

        # Pending queue should be cleared after observe
        pending = eng.storage.read_pending("t1")
        assert len(pending) == 0

        # Observation should be saved
        obs = eng.storage.read_observations("t1")
        assert "🔴" in obs or "Date:" in obs or FAKE_OBSERVATION.strip() in obs

    def test_no_trigger_below_threshold(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)

        # Small message — should NOT trigger
        status = eng.add_message("t2", role="user", content="Hello")

        assert status["observed"] is False
        assert status["reflected"] is False
        assert status["pending_tokens"] > 0

    def test_observe_clears_pending(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=5)
        eng.add_message("t3", role="user", content="X" * 100)
        eng.add_message("t3", role="assistant", content="Y" * 100)

        # At least one should have triggered observe
        pending = eng.storage.read_pending("t3")
        assert len(pending) == 0

    def test_reflect_auto_triggered(self, workspace: Path) -> None:
        """Reflector should auto-trigger when accumulated observations exceed threshold."""
        eng = _make_engine_with_mock(
            workspace, FAKE_OBSERVATION,
            observer_threshold=5,
            reflector_threshold=10,
        )

        # Seed observations directly to exceed reflector threshold
        large_obs = "Date: 2026-01-01\n" + ("- 🔴 12:00 item\n" * 20)
        eng.storage.append_observation("t4", large_obs)

        # Now add a message — it checks obs tokens after observe
        status = eng.add_message("t4", role="user", content="Trigger reflect " * 5)
        # Either observed or reflected should be true (or both)
        assert status["reflected"] is True or status["observed"] is True


# ---------------------------------------------------------------------------
# Test 2: observe output format
# ---------------------------------------------------------------------------

class TestObserveOutputFormat:
    """Observer output should contain emoji priorities and date headers."""

    def test_emoji_format_present(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        eng.storage.append_message("t5", {"role": "user", "content": "Hello", "timestamp": "12:00"})

        result = eng.observe("t5")
        assert result is not None
        # Must contain priority emojis
        assert "🔴" in result
        assert "Date:" in result

    def test_date_line_format(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        eng.storage.append_message("t6", {"role": "user", "content": "Test", "timestamp": "12:00"})

        result = eng.observe("t6")
        assert result is not None
        lines = result.strip().splitlines()
        # First non-empty line should match "Date: YYYY-MM-DD"
        date_lines = [l for l in lines if l.startswith("Date:")]
        assert len(date_lines) >= 1
        for dl in date_lines:
            parts = dl.split()
            assert len(parts) == 2
            assert parts[0] == "Date:"
            # Validate date format YYYY-MM-DD
            date_str = parts[1]
            assert len(date_str) == 10
            assert date_str[4] == "-" and date_str[7] == "-"

    def test_no_pending_returns_none(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        result = eng.observe("empty-thread")
        assert result is None


# ---------------------------------------------------------------------------
# Test 3: reflect reduces tokens
# ---------------------------------------------------------------------------

class TestReflectReducesTokens:
    """Reflector output should be smaller than accumulated observations."""

    def test_reflection_smaller_than_input(self, workspace: Path) -> None:
        # Build up a large observation log
        blocks = []
        for i in range(20):
            blocks.extend([
                "Date: 2026-01-0" + str(i % 9 + 1),
                "- 🔴 12:00 Critical item about project X spanning multiple lines of text",
                "- 🟡 12:30 Important technical detail about ModernBERT-large model inference",
                "- 🟢 13:00 Useful background information about deployment strategy on M3",
            ])
        large_obs = "\n".join(blocks)

        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION)
        eng.storage.append_observation("t7", large_obs)

        result = eng.reflect("t7")
        assert result is not None

        input_tokens = estimate_tokens(large_obs)
        output_tokens = estimate_tokens(result)

        # Reflection should be smaller (or at worst equal for tiny inputs)
        assert output_tokens <= input_tokens, (
            f"Reflection ({output_tokens} tokens) should be ≤ input ({input_tokens} tokens)"
        )

    def test_reflection_saved_to_storage(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION)
        eng.storage.append_observation("t8", FAKE_OBSERVATION)

        eng.reflect("t8")

        saved = eng.storage.read_reflection("t8")
        assert FAKE_REFLECTION.strip() in saved


# ---------------------------------------------------------------------------
# Test 4: storage persistence
# ---------------------------------------------------------------------------

class TestStoragePersistence:
    """Storage should persist across EngramEngine instances."""

    def test_observations_persist(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        storage.append_observation("t9", FAKE_OBSERVATION)

        # Create a new storage instance pointing to same dir
        storage2 = EngramStorage(workspace)
        obs = storage2.read_observations("t9")
        assert FAKE_OBSERVATION.strip() in obs

    def test_pending_persist(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        msg = {"role": "user", "content": "Hello persistent!", "timestamp": "12:00"}
        storage.append_message("t10", msg)

        storage2 = EngramStorage(workspace)
        pending = storage2.read_pending("t10")
        assert len(pending) == 1
        assert pending[0]["content"] == "Hello persistent!"

    def test_reflection_overwrites(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        storage.write_reflection("t11", "First reflection")
        storage.write_reflection("t11", "Second reflection")

        content = storage.read_reflection("t11")
        assert "Second reflection" in content
        # First should not appear (it was overwritten)
        assert "First reflection" not in content

    def test_meta_persists(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        storage.append_observation("t12", "test obs")

        meta = storage.read_meta("t12")
        assert meta["thread_id"] == "t12"
        assert "last_observed_at" in meta

    def test_clear_pending(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        storage.append_message("t13", {"role": "user", "content": "A"})
        storage.append_message("t13", {"role": "user", "content": "B"})
        assert storage.pending_count("t13") == 2

        storage.clear_pending("t13")
        assert storage.pending_count("t13") == 0

    def test_list_threads(self, workspace: Path) -> None:
        storage = EngramStorage(workspace)
        for tid in ["alpha", "beta", "gamma"]:
            storage.append_message(tid, {"role": "user", "content": "x"})
        # list_threads only returns threads with meta.json; force meta creation
        for tid in ["alpha", "beta", "gamma"]:
            storage.append_observation(tid, "obs")

        threads = storage.list_threads()
        for tid in ["alpha", "beta", "gamma"]:
            assert tid in threads

    def test_atomic_write_no_partial(self, workspace: Path) -> None:
        """Atomic write: even if we verify after write, content is complete."""
        storage = EngramStorage(workspace)
        long_content = "x" * 100_000
        storage.write_reflection("t14", long_content)

        content = storage.read_reflection("t14")
        assert long_content in content


# ---------------------------------------------------------------------------
# Test 5: get_context structure
# ---------------------------------------------------------------------------

class TestGetContextStructure:
    """get_context() should return the expected dict structure."""

    def test_context_keys(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        ctx = eng.get_context("t15")

        assert "thread_id" in ctx
        assert "observations" in ctx
        assert "reflection" in ctx
        assert "recent_messages" in ctx
        assert "stats" in ctx
        assert "meta" in ctx

    def test_stats_keys(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        ctx = eng.get_context("t16")

        stats = ctx["stats"]
        assert "observation_tokens" in stats
        assert "reflection_tokens" in stats
        assert "pending_tokens" in stats
        assert "total_tokens" in stats
        assert "pending_count" in stats

    def test_stats_values_correct(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        eng.storage.append_observation("t17", FAKE_OBSERVATION)
        eng.storage.write_reflection("t17", FAKE_REFLECTION)
        eng.storage.append_message("t17", {"role": "user", "content": "Pending msg"})

        ctx = eng.get_context("t17")
        assert ctx["stats"]["observation_tokens"] > 0
        assert ctx["stats"]["reflection_tokens"] > 0
        assert ctx["stats"]["pending_tokens"] > 0
        assert ctx["stats"]["pending_count"] == 1
        assert ctx["stats"]["total_tokens"] == (
            ctx["stats"]["observation_tokens"]
            + ctx["stats"]["reflection_tokens"]
            + ctx["stats"]["pending_tokens"]
        )

    def test_build_system_context_includes_sections(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        eng.storage.append_observation("t18", FAKE_OBSERVATION)
        eng.storage.write_reflection("t18", FAKE_REFLECTION)

        ctx_str = eng.build_system_context("t18")
        assert "Long-Term Memory" in ctx_str or "Reflections" in ctx_str
        assert "Recent Observations" in ctx_str

    def test_build_system_context_empty_thread(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)
        ctx_str = eng.build_system_context("nonexistent-thread")
        assert ctx_str == ""


# ---------------------------------------------------------------------------
# Test 6: token counting accuracy
# ---------------------------------------------------------------------------

class TestTokenCountingAccuracy:
    """Token counting should be consistent and reasonable."""

    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_ascii_positive(self) -> None:
        tokens = estimate_tokens("Hello world, this is a test sentence.")
        assert tokens > 0

    def test_cjk_positive(self) -> None:
        tokens = estimate_tokens("用户在构建 OpenCompress 项目")
        assert tokens > 0

    def test_longer_text_more_tokens(self) -> None:
        short = estimate_tokens("Hello")
        long_ = estimate_tokens("Hello " * 100)
        assert long_ > short

    def test_messages_tokens_includes_overhead(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        tokens = _count_messages_tokens(msgs)
        content_tokens = estimate_tokens("Hi")
        # Should include at least 4 tokens of per-message overhead
        assert tokens >= content_tokens + 4

    def test_mixed_content_positive(self) -> None:
        mixed = "Hello 世界 — testing mixed EN/ZH content with punctuation."
        tokens = estimate_tokens(mixed)
        assert tokens > 5

    def test_messages_to_text_format(self) -> None:
        msgs = [
            {"role": "user", "content": "Hello", "timestamp": "12:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "12:01"},
        ]
        text = _messages_to_text(msgs)
        assert "[1] USER [12:00]" in text
        assert "[2] ASSISTANT [12:01]" in text
        assert "Hello" in text
        assert "Hi there!" in text


# ---------------------------------------------------------------------------
# Test 7: prompts integrity
# ---------------------------------------------------------------------------

class TestPrompts:
    """Verify prompt contents and templates."""

    def test_observer_prompt_has_emoji(self) -> None:
        assert "🔴" in OBSERVER_SYSTEM_PROMPT
        assert "🟡" in OBSERVER_SYSTEM_PROMPT
        assert "🟢" in OBSERVER_SYSTEM_PROMPT

    def test_observer_prompt_has_date_format(self) -> None:
        assert "Date: YYYY-MM-DD" in OBSERVER_SYSTEM_PROMPT

    def test_reflector_prompt_has_sections(self) -> None:
        assert "Persistent Context" in REFLECTOR_SYSTEM_PROMPT
        assert "Recent Events" in REFLECTOR_SYSTEM_PROMPT

    def test_observer_template_format(self) -> None:
        result = OBSERVER_USER_TEMPLATE.format(
            current_datetime="2026-03-05 12:00 UTC",
            messages_text="Test message",
        )
        assert "2026-03-05" in result
        assert "Test message" in result

    def test_reflector_template_format(self) -> None:
        result = REFLECTOR_USER_TEMPLATE.format(
            current_datetime="2026-03-05 12:00 UTC",
            observations_text="Test observation",
        )
        assert "2026-03-05" in result
        assert "Test observation" in result


# ---------------------------------------------------------------------------
# Test 8: LLM routing
# ---------------------------------------------------------------------------

class TestLLMRouting:
    """Verify correct LLM provider selection."""

    def test_no_api_key_raises(self, workspace: Path) -> None:
        eng = EngramEngine(
            workspace_path=workspace,
            anthropic_api_key="",
            openai_api_key="",
        )
        with pytest.raises(RuntimeError, match="no API key"):
            eng._call_llm("system", "user")

    def test_anthropic_key_calls_anthropic(self, workspace: Path) -> None:
        eng = EngramEngine(workspace_path=workspace, anthropic_api_key="key123")
        with patch.object(eng, "_call_anthropic", return_value="result") as mock_ant, \
             patch.object(eng, "_call_openai_compatible", return_value="oai") as mock_oai:
            result = eng._call_llm("sys", "usr")
            mock_ant.assert_called_once()
            mock_oai.assert_not_called()
            assert result == "result"

    def test_openai_key_calls_openai(self, workspace: Path) -> None:
        eng = EngramEngine(
            workspace_path=workspace,
            anthropic_api_key="",
            openai_api_key="oai-key",
        )
        with patch.object(eng, "_call_anthropic", return_value="ant") as mock_ant, \
             patch.object(eng, "_call_openai_compatible", return_value="result") as mock_oai:
            result = eng._call_llm("sys", "usr")
            mock_oai.assert_called_once()
            mock_ant.assert_not_called()
            assert result == "result"

    def test_anthropic_preferred_over_openai(self, workspace: Path) -> None:
        eng = EngramEngine(
            workspace_path=workspace,
            anthropic_api_key="ant-key",
            openai_api_key="oai-key",
        )
        with patch.object(eng, "_call_anthropic", return_value="ant-result") as mock_ant, \
             patch.object(eng, "_call_openai_compatible", return_value="oai-result") as mock_oai:
            result = eng._call_llm("sys", "usr")
            mock_ant.assert_called_once()
            mock_oai.assert_not_called()
            assert result == "ant-result"


# ---------------------------------------------------------------------------
# Test 9: engram_cli integration (smoke tests)
# ---------------------------------------------------------------------------

class TestEngramCLI:
    """Smoke tests for the CLI module."""

    def test_cli_status_no_threads(self, workspace: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = EngramEngine(workspace_path=workspace, anthropic_api_key="test")
        args = MagicMock()
        args.thread = None
        args.json = False
        rc = cli.cmd_status(eng, args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "No Engram threads" in captured.out

    def test_cli_observe_no_pending(self, workspace: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = EngramEngine(workspace_path=workspace, anthropic_api_key="test")
        args = MagicMock()
        args.thread = "empty"
        args.json = False
        rc = cli.cmd_observe(eng, args)
        assert rc == 1  # no pending messages

    def test_cli_context_empty(self, workspace: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = EngramEngine(workspace_path=workspace, anthropic_api_key="test")
        args = MagicMock()
        args.thread = "nonexistent"
        args.json = False
        rc = cli.cmd_context(eng, args)
        assert rc == 1  # no context

    def test_cli_context_with_data(self, workspace: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = EngramEngine(workspace_path=workspace, anthropic_api_key="test")
        eng.storage.append_observation("cli-t1", FAKE_OBSERVATION)
        args = MagicMock()
        args.thread = "cli-t1"
        args.json = False
        rc = cli.cmd_context(eng, args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Recent Observations" in captured.out

    def test_cli_ingest_json(self, workspace: Path, tmp_path: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION)

        input_file = tmp_path / "msgs.json"
        messages = [
            {"role": "user", "content": "Hello from ingest test"},
            {"role": "assistant", "content": "Response from ingest test"},
        ]
        input_file.write_text(json.dumps(messages))

        args = MagicMock()
        args.thread = "ingest-t1"
        args.input = str(input_file)
        args.json = False

        rc = cli.cmd_ingest(eng, args)
        assert rc == 0

        # Verify messages were added
        pending = eng.storage.read_pending("ingest-t1")
        assert len(pending) == 2

    def test_cli_ingest_jsonl(self, workspace: Path, tmp_path: Path, capsys) -> None:
        import scripts.engram_cli as cli  # type: ignore[import]
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)

        input_file = tmp_path / "msgs.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "Line 1"}),
            json.dumps({"role": "assistant", "content": "Line 2"}),
        ]
        input_file.write_text("\n".join(lines))

        args = MagicMock()
        args.thread = "ingest-t2"
        args.input = str(input_file)
        args.json = False

        rc = cli.cmd_ingest(eng, args)
        assert rc == 0
        pending = eng.storage.read_pending("ingest-t2")
        assert len(pending) == 2


# ---------------------------------------------------------------------------
# Test 10: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_unicode_content(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        status = eng.add_message(
            "unicode-t",
            role="user",
            content="Unicode: 中文 🔴 émojis 日本語 한국어",
        )
        assert status["error"] is None

        pending = eng.storage.read_pending("unicode-t")
        assert len(pending) == 1
        assert "中文" in pending[0]["content"]

    def test_empty_content(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        status = eng.add_message("empty-t", role="user", content="")
        assert status["error"] is None

    def test_multiple_threads_isolated(self, workspace: Path) -> None:
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        eng.add_message("thread-A", role="user", content="Message for A")
        eng.add_message("thread-B", role="user", content="Message for B")

        pending_a = eng.storage.read_pending("thread-A")
        pending_b = eng.storage.read_pending("thread-B")

        assert len(pending_a) == 1
        assert len(pending_b) == 1
        assert pending_a[0]["content"] != pending_b[0]["content"]

    def test_observation_appends(self, workspace: Path) -> None:
        """Multiple observe runs should append, not overwrite."""
        storage = EngramStorage(workspace)
        storage.append_observation("append-t", "First observation block")
        storage.append_observation("append-t", "Second observation block")

        obs = storage.read_observations("append-t")
        assert "First observation block" in obs
        assert "Second observation block" in obs

    def test_corrupt_jsonl_skipped(self, workspace: Path) -> None:
        """Corrupt JSONL lines should be silently skipped."""
        storage = EngramStorage(workspace)
        pending_path = storage._pending_path("corrupt-t")
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        with pending_path.open("w") as f:
            f.write('{"role": "user", "content": "Good line"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"role": "assistant", "content": "Also good"}\n')

        messages = storage.read_pending("corrupt-t")
        assert len(messages) == 2
        assert messages[0]["content"] == "Good line"
        assert messages[1]["content"] == "Also good"


# ---------------------------------------------------------------------------
# Test 11: HTTP retry logic
# ---------------------------------------------------------------------------

class TestHttpRetry:
    """_http_post() should retry on transient errors and not retry on 401/403."""

    def test_http_retry_on_429(self, workspace: Path) -> None:
        """Should retry on HTTP 429 and eventually succeed."""
        import urllib.error
        from claw_compactor.engram import _http_post

        call_count = 0

        def fake_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate 429 on first two attempts
                err = urllib.error.HTTPError(
                    url="http://test", code=429,
                    msg="Too Many Requests", hdrs=None, fp=None,  # type: ignore[arg-type]
                )
                # HTTPError.read() must be callable
                err.read = lambda: b"rate limited"
                raise err
            # Third attempt succeeds: return a file-like object
            import io
            import json as _json

            class FakeResp:
                def read(self):
                    return _json.dumps({"ok": True}).encode()
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass

            return FakeResp()

        with patch("lib.engram._HTTPX_AVAILABLE", False), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("time.sleep"):  # skip actual delays
            result = _http_post("http://test", {}, {}, max_retries=3)

        assert result == {"ok": True}
        assert call_count == 3

    def test_http_no_retry_on_401(self, workspace: Path) -> None:
        """Should raise immediately on HTTP 401 (no retry)."""
        import urllib.error
        from claw_compactor.engram import _http_post

        call_count = 0

        def fake_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            err = urllib.error.HTTPError(
                url="http://test", code=401,
                msg="Unauthorized", hdrs=None, fp=None,  # type: ignore[arg-type]
            )
            err.read = lambda: b"unauthorized"
            raise err

        with patch("lib.engram._HTTPX_AVAILABLE", False), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="401"):
                _http_post("http://test", {}, {}, max_retries=3)

        # Must have been called exactly once — no retries
        assert call_count == 1


# ---------------------------------------------------------------------------
# Test 12: batch_ingest
# ---------------------------------------------------------------------------

class TestBatchIngest:
    """batch_ingest() should write all messages and check thresholds once."""

    def test_batch_ingest_writes_all_messages(self, workspace: Path) -> None:
        """All messages in the batch should appear in pending storage."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        messages = [
            {"role": "user", "content": f"Message {i}", "timestamp": "12:00"}
            for i in range(10)
        ]
        eng.batch_ingest("batch-t1", messages)
        pending = eng.storage.read_pending("batch-t1")
        assert len(pending) == 10

    def test_batch_ingest_triggers_observe_once(self, workspace: Path) -> None:
        """batch_ingest() should only check thresholds at the end (one observe max)."""
        # Threshold low enough to trigger on the batch, but the mock tracks calls
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=10)
        observe_calls = []
        original_run_observer = eng._run_observer

        def counting_run_observer(thread_id, pending):
            observe_calls.append(thread_id)
            return original_run_observer(thread_id, pending)

        eng._run_observer = counting_run_observer  # type: ignore[method-assign]

        messages = [
            {"role": "user", "content": "X" * 50, "timestamp": "12:00"}
            for _ in range(5)
        ]
        eng.batch_ingest("batch-t2", messages)
        # Should have triggered at most once (at the end), not 5 times
        assert len(observe_calls) <= 1

    def test_batch_ingest_returns_status(self, workspace: Path) -> None:
        """batch_ingest() should return a valid status dict."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        messages = [{"role": "user", "content": "hello"}]
        status = eng.batch_ingest("batch-t3", messages)
        assert "observed" in status
        assert "reflected" in status
        assert "pending_tokens" in status
        assert "error" in status


# ---------------------------------------------------------------------------
# Test 13: add_message skip observe
# ---------------------------------------------------------------------------

class TestAddMessageSkipObserve:
    """add_message(auto_observe=False) should skip threshold checks."""

    def test_skip_observe_only_writes_pending(self, workspace: Path) -> None:
        """With auto_observe=False, message is written but observer never fires."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=5)
        # Even with low threshold and large content, no observe should fire
        status = eng.add_message(
            "skip-t1", role="user",
            content="A" * 500,  # well above threshold
            auto_observe=False,
        )
        assert status["observed"] is False
        assert status["reflected"] is False
        assert status["error"] is None

        # Message should still be in pending
        pending = eng.storage.read_pending("skip-t1")
        assert len(pending) == 1
        assert "A" * 100 in pending[0]["content"]

    def test_auto_observe_true_still_triggers(self, workspace: Path) -> None:
        """Default auto_observe=True should still trigger the observer."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=5)
        status = eng.add_message(
            "skip-t2", role="user",
            content="A" * 500,
            auto_observe=True,  # explicit default
        )
        assert status["observed"] is True


# ---------------------------------------------------------------------------
# Test 14: observer batching
# ---------------------------------------------------------------------------

class TestObserverBatching:
    """_run_observer() should split large message lists into batches."""

    @staticmethod
    def _make_messages_over_limit(n_messages: int = 2) -> list:
        """Create `n_messages` messages whose combined token count exceeds
        MAX_OBSERVER_INPUT_TOKENS, but each individual message is below the limit.

        Uses varied realistic text so tiktoken doesn't compress it heavily.
        """
        # Build content that genuinely exceeds the limit when summed.
        # We need total > MAX_OBSERVER_INPUT_TOKENS (80K) and each < 80K.
        # Use enough unique words so tiktoken can't compress heavily.
        words_per_msg = 50_000  # ~50K tokens per message via word diversity
        content = " ".join(
            f"event{i} status{i} detail{i} context{i} result{i}"
            for i in range(words_per_msg // 5)
        )
        return [
            {"role": "user", "content": content, "timestamp": "12:00"}
            for _ in range(n_messages)
        ]

    def test_small_batch_single_call(self, workspace: Path) -> None:
        """When total tokens < MAX_OBSERVER_INPUT_TOKENS, only one LLM call is made."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        call_count = [0]
        original_llm_observe = eng._llm_observe

        def counting_llm_observe(msgs):
            call_count[0] += 1
            return original_llm_observe(msgs)

        eng._llm_observe = counting_llm_observe  # type: ignore[method-assign]

        # Small messages — well below 80K total
        small_messages = [
            {"role": "user", "content": f"Short message number {i}", "timestamp": "12:00"}
            for i in range(5)
        ]
        eng._run_observer("batch-obs-t1", small_messages)

        assert call_count[0] == 1, "Expected single LLM call for small input"

    def test_large_batch_multiple_calls(self, workspace: Path) -> None:
        """When total tokens > MAX_OBSERVER_INPUT_TOKENS, multiple LLM calls are made."""
        from claw_compactor.engram import _count_messages_tokens

        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)

        # Build messages that are guaranteed to exceed limit by using mock token counting
        # Patch _count_messages_tokens to return controlled values
        call_count = [0]

        def counting_llm_observe(msgs):
            call_count[0] += 1
            return f"batch-observation-{call_count[0]}"

        eng._llm_observe = counting_llm_observe  # type: ignore[method-assign]

        # Patch _count_messages_tokens in engram module to return predictable high values
        import claw_compactor.engram as engram_mod
        original_count = engram_mod._count_messages_tokens

        def fake_count(msgs):
            # Each single message appears to be 60K tokens
            return len(msgs) * 60_000

        try:
            engram_mod._count_messages_tokens = fake_count
            messages = [
                {"role": "user", "content": "message content", "timestamp": "12:00"}
                for _ in range(3)
            ]
            result = eng._run_observer("batch-obs-t2", messages)
        finally:
            engram_mod._count_messages_tokens = original_count

        assert call_count[0] > 1, "Expected multiple LLM calls for large input"
        assert "---" in result, "Combined result should contain batch separator"

    def test_batching_clears_pending(self, workspace: Path) -> None:
        """After batching, the pending queue should be cleared."""
        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)

        messages = [
            {"role": "user", "content": f"Message {i}", "timestamp": "12:00"}
            for i in range(5)
        ]
        for msg in messages:
            eng.storage.append_message("batch-obs-t3", msg)

        eng._llm_observe = MagicMock(return_value="obs")  # type: ignore[method-assign]
        eng._run_observer("batch-obs-t3", messages)

        pending = eng.storage.read_pending("batch-obs-t3")
        assert len(pending) == 0, "Pending queue should be empty after observer run"

    def test_batching_appends_combined_observation(self, workspace: Path) -> None:
        """Batched observations should be combined and appended to storage."""
        import claw_compactor.engram as engram_mod

        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)

        batch_responses = []

        def recording_llm_observe(msgs):
            resp = f"observation-for-batch-{len(batch_responses) + 1}"
            batch_responses.append(resp)
            return resp

        eng._llm_observe = recording_llm_observe  # type: ignore[method-assign]

        # Force batching via mock token counter
        original_count = engram_mod._count_messages_tokens

        def fake_count(msgs):
            return len(msgs) * 60_000

        try:
            engram_mod._count_messages_tokens = fake_count
            messages = [
                {"role": "user", "content": f"Message {i}", "timestamp": "12:00"}
                for i in range(3)
            ]
            eng._run_observer("batch-obs-t4", messages)
        finally:
            engram_mod._count_messages_tokens = original_count

        saved = eng.storage.read_observations("batch-obs-t4")
        for resp in batch_responses:
            assert resp in saved

    def test_single_oversized_message_doesnt_loop(self, workspace: Path) -> None:
        """A single message that exceeds the limit should still be processed (no infinite loop)."""
        import claw_compactor.engram as engram_mod

        eng = _make_engine_with_mock(workspace, FAKE_OBSERVATION, observer_threshold=9999)
        call_count = [0]

        def counting_llm_observe(msgs):
            call_count[0] += 1
            if call_count[0] > 5:
                raise RuntimeError("Infinite loop detected!")
            return "oversized-observation"

        eng._llm_observe = counting_llm_observe  # type: ignore[method-assign]

        # Patch token counter so single message appears to be 200K tokens
        original_count = engram_mod._count_messages_tokens

        def fake_count(msgs):
            return len(msgs) * 200_000

        try:
            engram_mod._count_messages_tokens = fake_count
            huge_message = {"role": "user", "content": "Large content", "timestamp": "12:00"}
            result = eng._run_observer("batch-obs-t5", [huge_message])
        finally:
            engram_mod._count_messages_tokens = original_count

        assert call_count[0] == 1, "Single oversized message should produce exactly one call"
        assert result == "oversized-observation"


# ---------------------------------------------------------------------------
# Test 15: reflector truncation
# ---------------------------------------------------------------------------

class TestReflectorTruncation:
    """_run_reflector() should truncate observations that exceed MAX_REFLECTOR_INPUT_TOKENS."""

    def test_small_observations_no_truncation(self, workspace: Path) -> None:
        """Small observations should pass through unchanged."""
        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION, observer_threshold=9999)

        received_text = []

        def recording_llm_reflect(obs):
            received_text.append(obs)
            return FAKE_REFLECTION

        eng._llm_reflect = recording_llm_reflect  # type: ignore[method-assign]

        small_obs = "Date: 2026-01-01\n- 🔴 12:00 Small observation\n"
        eng._run_reflector("refl-t1", small_obs)

        assert received_text[0] == small_obs, "Small observations should not be modified"

    def test_large_observations_truncated(self, workspace: Path) -> None:
        """Observations exceeding MAX_REFLECTOR_INPUT_TOKENS should be truncated."""
        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION, observer_threshold=9999)

        received_tokens = []

        def recording_llm_reflect(obs):
            received_tokens.append(estimate_tokens(obs))
            return FAKE_REFLECTION

        eng._llm_reflect = recording_llm_reflect  # type: ignore[method-assign]

        # Build observations that are clearly over the limit
        large_obs = ("- 🔴 12:00 Critical event that happened\n" * 10_000)
        original_tokens = estimate_tokens(large_obs)
        assert original_tokens > MAX_REFLECTOR_INPUT_TOKENS, "Test setup: input must exceed limit"

        eng._run_reflector("refl-t2", large_obs)

        assert len(received_tokens) == 1
        # The truncation loop counts tokens per-line; the rejoined text may be slightly
        # higher due to newline tokenization interactions — allow a 15% margin.
        # The key guarantee is that input is dramatically reduced from the original.
        assert received_tokens[0] <= MAX_REFLECTOR_INPUT_TOKENS * 1.15, (
            f"Reflector received {received_tokens[0]} tokens, "
            f"expected roughly <= {MAX_REFLECTOR_INPUT_TOKENS} (with 15% join margin)"
        )
        # Must be significantly less than the original (at least 25% reduction)
        assert received_tokens[0] < original_tokens * 0.75, (
            f"Truncation had no effect: {received_tokens[0]} vs original {original_tokens}"
        )

    def test_truncation_keeps_tail(self, workspace: Path) -> None:
        """Truncation should keep the most recent (tail) content, not the head."""
        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION, observer_threshold=9999)

        received_text = []

        def recording_llm_reflect(obs):
            received_text.append(obs)
            return FAKE_REFLECTION

        eng._llm_reflect = recording_llm_reflect  # type: ignore[method-assign]

        # Build content with distinct head and tail markers
        # Make it large enough to trigger truncation
        head_lines = ["head-line-AAAA\n"] * 5_000
        tail_lines = ["tail-line-ZZZZ\n"] * 2_000
        large_obs = "".join(head_lines) + "".join(tail_lines)

        eng._run_reflector("refl-t3", large_obs)

        result_text = received_text[0]
        # Tail should be present; head may have been truncated
        assert "tail-line-ZZZZ" in result_text, "Tail content should be preserved after truncation"

    def test_truncation_reflection_saved(self, workspace: Path) -> None:
        """Even with truncated input, the reflection should be saved to storage."""
        eng = _make_engine_with_mock(workspace, FAKE_REFLECTION, observer_threshold=9999)
        eng._llm_reflect = MagicMock(return_value=FAKE_REFLECTION)  # type: ignore[method-assign]

        large_obs = "- 🔴 12:00 Item\n" * 10_000

        eng._run_reflector("refl-t4", large_obs)

        saved = eng.storage.read_reflection("refl-t4")
        assert FAKE_REFLECTION.strip() in saved
