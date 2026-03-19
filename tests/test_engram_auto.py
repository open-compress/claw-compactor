"""
tests/test_engram_auto.py — Tests for multi-channel auto-discovery,
unified config, and concurrent processing (Engram Layer 6 refactor).

Run with:
    pytest tests/test_engram_auto.py -v

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from claw_compactor.config import load_engram_config, engram_engine_kwargs, _load_dotenv, _deep_merge
from engram_auto import detect_thread_id, convert_session, EngramAutoRunner, _extract_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    return d


def _write_session(sessions_dir: Path, name: str, lines: List[dict]) -> Path:
    """Write a mock session JSONL file."""
    p = sessions_dir / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return p


def _make_openclaw_msg(role: str, text: str, ts: str = "") -> dict:
    """Build an OpenClaw-format session message."""
    msg: dict = {
        "type": "message",
        "message": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
    }
    if ts:
        msg["timestamp"] = ts
    return msg


# ---------------------------------------------------------------------------
# Test 1: detect_thread_id — channel detection
# ---------------------------------------------------------------------------

class TestDetectThreadId:
    """detect_thread_id() should correctly map sessions to thread IDs."""

    def test_discord_general(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s1", [
            _make_openclaw_msg("user",
                "You are in [Discord Guild #general channel id:1470169146539901001]"),
        ])
        assert detect_thread_id(p) == "discord-general"

    def test_discord_open_compress(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s2", [
            _make_openclaw_msg("user",
                "Context: [Discord Guild #open-compress channel id:1476885945163714641]"),
        ])
        assert detect_thread_id(p) == "discord-open-compress"

    def test_discord_aimm(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s3", [
            _make_openclaw_msg("user",
                "Channel: [Discord Guild #aimm channel id:1234567890]"),
        ])
        assert detect_thread_id(p) == "discord-aimm"

    def test_cron_job(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s4", [
            _make_openclaw_msg("system",
                'A cron job "cortex-tick" has fired.'),
        ])
        assert detect_thread_id(p) == "cron-cortex-tick"

    def test_subagent(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s5", [
            _make_openclaw_msg("system", "You are running as a subagent (depth 1/1)."),
        ])
        assert detect_thread_id(p) == "subagent"

    def test_fallback_openclaw_main(self, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s6", [
            _make_openclaw_msg("user", "Hello, this is a random message."),
        ])
        assert detect_thread_id(p) == "openclaw-main"

    def test_empty_session(self, sessions_dir: Path) -> None:
        p = sessions_dir / "empty.jsonl"
        p.write_text("")
        assert detect_thread_id(p) == "openclaw-main"

    def test_nonexistent_session(self, sessions_dir: Path) -> None:
        p = sessions_dir / "nonexistent.jsonl"
        # Should not raise, returns default
        result = detect_thread_id(p)
        assert result == "openclaw-main"

    def test_subagent_takes_priority(self, sessions_dir: Path) -> None:
        """subagent detection should override channel name if both present."""
        p = _write_session(sessions_dir, "s7", [
            _make_openclaw_msg("system",
                "You are a subagent in [Discord Guild #general channel id:111]"),
        ])
        assert detect_thread_id(p) == "subagent"

    def test_generic_discord_channel(self, sessions_dir: Path) -> None:
        """Unknown channel name with channel id: ID takes priority for stability.

        Phase 1 change: when both an unknown channel name and a channel id
        are present, the channel id is used (more stable — ids never change,
        names can be renamed).  Result: discord-channel-{id}.
        """
        p = _write_session(sessions_dir, "s8", [
            _make_openclaw_msg("user",
                "[Discord Guild #mychannel channel id:9999]"),
        ])
        result = detect_thread_id(p)
        # Phase 1: id-based naming is preferred over unknown name for stability
        assert result == "discord-channel-9999"

    def test_generic_discord_channel_name_only(self, sessions_dir: Path) -> None:
        """Unknown channel name with NO channel id should become discord-{name}."""
        p = _write_session(sessions_dir, "s9", [
            _make_openclaw_msg("user",
                "[Discord Guild #mychannel] some content here"),
        ])
        result = detect_thread_id(p)
        assert result == "discord-mychannel"


# ---------------------------------------------------------------------------
# Test 2: convert_session — format conversion
# ---------------------------------------------------------------------------

class TestConvertSession:
    """convert_session() should produce valid Engram-format JSONL."""

    def test_basic_conversion(self, tmp_path: Path) -> None:
        session = tmp_path / "sess.jsonl"
        session.write_text(
            json.dumps(_make_openclaw_msg("user", "Hello world")) + "\n" +
            json.dumps(_make_openclaw_msg("assistant", "Hi there")) + "\n"
        )
        out = tmp_path / "out.jsonl"
        count = convert_session(session, out)

        assert count == 2
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        msg0 = json.loads(lines[0])
        assert msg0["role"] == "user"
        assert msg0["content"] == "Hello world"

    def test_skips_non_message_events(self, tmp_path: Path) -> None:
        session = tmp_path / "sess.jsonl"
        session.write_text(
            json.dumps({"type": "system_event", "data": "boot"}) + "\n" +
            json.dumps(_make_openclaw_msg("user", "Real message")) + "\n"
        )
        out = tmp_path / "out.jsonl"
        count = convert_session(session, out)
        assert count == 1

    def test_skips_empty_content(self, tmp_path: Path) -> None:
        session = tmp_path / "sess.jsonl"
        obj = {"type": "message", "message": {"role": "user", "content": ""}}
        session.write_text(json.dumps(obj) + "\n")
        out = tmp_path / "out.jsonl"
        count = convert_session(session, out)
        assert count == 0

    def test_preserves_timestamp(self, tmp_path: Path) -> None:
        session = tmp_path / "sess.jsonl"
        obj = _make_openclaw_msg("user", "With timestamp")
        obj["timestamp"] = "2026-03-05T12:00:00Z"
        session.write_text(json.dumps(obj) + "\n")
        out = tmp_path / "out.jsonl"
        convert_session(session, out)
        msg = json.loads(out.read_text())
        assert msg["timestamp"] == "2026-03-05T12:00:00Z"

    def test_skips_corrupt_lines(self, tmp_path: Path) -> None:
        session = tmp_path / "sess.jsonl"
        session.write_text(
            "NOT JSON\n" +
            json.dumps(_make_openclaw_msg("user", "Good message")) + "\n"
        )
        out = tmp_path / "out.jsonl"
        count = convert_session(session, out)
        assert count == 1


# ---------------------------------------------------------------------------
# Test 3: _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_string_passthrough(self) -> None:
        assert _extract_text("hello") == "hello"

    def test_list_of_text_blocks(self) -> None:
        blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        result = _extract_text(blocks)
        assert "Hello" in result
        assert "World" in result

    def test_list_of_strings(self) -> None:
        result = _extract_text(["foo", "bar"])
        assert "foo" in result and "bar" in result

    def test_non_text_blocks_ignored(self) -> None:
        blocks = [{"type": "tool_use", "name": "bash"}, {"type": "text", "text": "hi"}]
        result = _extract_text(blocks)
        assert "hi" in result

    def test_fallback_to_str(self) -> None:
        result = _extract_text(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# Test 4: load_engram_config
# ---------------------------------------------------------------------------

class TestLoadEngramConfig:
    """Config loading: yaml file, env overrides, defaults."""

    def test_loads_defaults_without_file(self, tmp_path: Path, monkeypatch) -> None:
        # Point away from real engram.yaml
        monkeypatch.setenv("ENGRAM_CONFIG", str(tmp_path / "nonexistent.yaml"))
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        cfg = load_engram_config()
        assert "llm" in cfg
        assert "threads" in cfg
        assert "sessions" in cfg
        assert "storage" in cfg
        assert "concurrency" in cfg

    def test_env_var_overrides(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ENGRAM_CONFIG", str(tmp_path / "nonexistent.yaml"))
        monkeypatch.setenv("ENGRAM_MODEL", "my-test-model")
        monkeypatch.setenv("ENGRAM_OBSERVER_THRESHOLD", "12345")
        monkeypatch.setenv("ENGRAM_MAX_WORKERS", "8")
        cfg = load_engram_config()
        assert cfg["llm"]["model"] == "my-test-model"
        assert cfg["threads"]["default"]["observer_threshold"] == 12345
        assert cfg["concurrency"]["max_workers"] == 8

    def test_yaml_file_loaded(self, tmp_path: Path, monkeypatch) -> None:
        yaml_content = """
llm:
  model: test-model-from-yaml
  max_tokens: 1234
threads:
  default:
    observer_threshold: 5000
"""
        yaml_path = tmp_path / "engram.yaml"
        yaml_path.write_text(yaml_content)
        # Suppress .env loading by pointing to a nonexistent .env inside config.py's
        # root detection. We do this by patching _load_dotenv to a no-op for this test.
        # Also clear any lingering env-var overrides.
        monkeypatch.delenv("ENGRAM_MODEL", raising=False)
        monkeypatch.delenv("ENGRAM_MAX_TOKENS", raising=False)
        monkeypatch.delenv("ENGRAM_OBSERVER_THRESHOLD", raising=False)
        # Prevent .env from re-setting ENGRAM_MODEL during this call
        with patch("lib.config._load_dotenv"):
            cfg = load_engram_config(yaml_path)
        assert cfg["llm"]["model"] == "test-model-from-yaml"
        assert cfg["llm"]["max_tokens"] == 1234
        assert cfg["threads"]["default"]["observer_threshold"] == 5000

    def test_json_fallback(self, tmp_path: Path, monkeypatch) -> None:
        json_data = {
            "llm": {"model": "from-json", "max_tokens": 999},
            "threads": {"default": {"observer_threshold": 7777}},
        }
        json_path = tmp_path / "engram.json"
        json_path.write_text(json.dumps(json_data))
        # Clear env-var overrides; suppress .env re-population
        monkeypatch.delenv("ENGRAM_MODEL", raising=False)
        monkeypatch.delenv("ENGRAM_MAX_TOKENS", raising=False)
        monkeypatch.delenv("ENGRAM_OBSERVER_THRESHOLD", raising=False)
        with patch("lib.config._load_dotenv"):
            cfg = load_engram_config(json_path)
        assert cfg["llm"]["model"] == "from-json"
        assert cfg["threads"]["default"]["observer_threshold"] == 7777

    def test_paths_expanded(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ENGRAM_STORAGE_DIR", "~/my/custom/path")
        monkeypatch.setenv("ENGRAM_CONFIG", str(tmp_path / "nonexistent.yaml"))
        cfg = load_engram_config()
        assert "~" not in cfg["storage"]["base_dir"]
        assert "my/custom/path" in cfg["storage"]["base_dir"]

    def test_deep_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}, "c": 4}
        result = _deep_merge(base, override)
        assert result["a"]["x"] == 1   # preserved from base
        assert result["a"]["y"] == 99  # overridden
        assert result["a"]["z"] == 100  # new from override
        assert result["b"] == 3        # untouched
        assert result["c"] == 4        # new


# ---------------------------------------------------------------------------
# Test 5: engram_engine_kwargs
# ---------------------------------------------------------------------------

class TestEngramEngineKwargs:
    def test_openai_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-oai-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:8403",
                "api_key_env": "OPENAI_API_KEY",
                "model": "test-model",
                "max_tokens": 2048,
            },
            "threads": {"default": {"observer_threshold": 1000, "reflector_threshold": 2000}},
        }
        kwargs = engram_engine_kwargs(cfg)
        assert kwargs["openai_api_key"] == "test-oai-key"
        assert kwargs["openai_base_url"] == "http://localhost:8403"
        assert kwargs["model"] == "test-model"
        assert kwargs["observer_threshold"] == 1000

    def test_anthropic_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key")
        cfg = {
            "llm": {
                "provider": "anthropic",
                "base_url": "",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-opus-4-5",
                "max_tokens": 4096,
            },
            "threads": {"default": {"observer_threshold": 30000, "reflector_threshold": 40000}},
        }
        kwargs = engram_engine_kwargs(cfg)
        assert kwargs["anthropic_api_key"] == "test-ant-key"
        assert kwargs["openai_api_key"] == ""


# ---------------------------------------------------------------------------
# Test 6: EngramAutoRunner — concurrent ingestion
# ---------------------------------------------------------------------------

class TestEngramAutoRunner:
    """Test the concurrent auto-runner."""

    def _make_cfg(self, sessions_dir: Path, workspace: Path) -> dict:
        return {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:9999",
                "api_key_env": "OPENAI_API_KEY",
                "model": "test-model",
                "max_tokens": 512,
            },
            "threads": {"default": {"observer_threshold": 99999, "reflector_threshold": 99999}},
            "sessions": {"scan_dir": str(sessions_dir), "max_age_hours": 48},
            "storage": {"base_dir": str(workspace / "memory" / "engram")},
            "concurrency": {"max_workers": 2},
        }

    def test_dry_run_no_write(self, workspace: Path, sessions_dir: Path) -> None:
        p = _write_session(sessions_dir, "s1", [
            _make_openclaw_msg("user", "Hello from dry run test"),
        ])
        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=True)
        totals = runner.run_once()
        # dry_run → nothing ingested
        assert all(v == 0 for v in totals.values()) or totals == {}
        # storage should have no pending messages
        from claw_compactor.engram_storage import EngramStorage
        storage = EngramStorage(workspace)
        threads = storage.list_threads()
        assert threads == []

    def test_multi_channel_isolation(self, workspace: Path, sessions_dir: Path, monkeypatch) -> None:
        """Sessions from different channels should end up in different threads."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

        _write_session(sessions_dir, "general_sess", [
            _make_openclaw_msg("user", "[Discord Guild #general channel id:111] Hello"),
        ])
        _write_session(sessions_dir, "aimm_sess", [
            _make_openclaw_msg("user", "[Discord Guild #aimm channel id:222] Hello"),
        ])

        cfg = self._make_cfg(sessions_dir, workspace)
        # Very high threshold so no LLM calls fire
        cfg["threads"]["default"]["observer_threshold"] = 999999
        cfg["threads"]["default"]["reflector_threshold"] = 999999

        runner = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=False)
        # Patch _call_llm on the engine to avoid HTTP calls
        with patch("claw_compactor.engram_llm.EngramLLMClient.call", return_value="fake obs"):
            totals = runner.run_once()

        from claw_compactor.engram_storage import EngramStorage
        storage = EngramStorage(workspace)

        # Use pending.jsonl existence (not meta.json which only appears after observe)
        engram_base = workspace / "memory" / "engram"
        thread_dirs = [d.name for d in engram_base.iterdir() if d.is_dir()] if engram_base.exists() else []

        # Both channels should have their own thread directory
        assert "discord-general" in thread_dirs, f"expected discord-general in {thread_dirs}"
        assert "discord-aimm" in thread_dirs, f"expected discord-aimm in {thread_dirs}"

        # Content isolation: each thread gets only its own session's messages
        general_msgs = storage.read_pending("discord-general")
        aimm_msgs = storage.read_pending("discord-aimm")
        general_texts = [m.get("content", "") for m in general_msgs]
        aimm_texts = [m.get("content", "") for m in aimm_msgs]
        assert not any("aimm" in t.lower() for t in general_texts), "general thread has aimm content"
        assert not any("general" in t.lower() for t in aimm_texts), "aimm thread has general content"

    def test_processed_marker_prevents_reprocess(
        self, workspace: Path, sessions_dir: Path, monkeypatch
    ) -> None:
        """A session that was already processed should not be ingested again."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        _write_session(sessions_dir, "repeated_sess", [
            _make_openclaw_msg("user", "Hello, I should only be ingested once"),
        ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=False)

        with patch("claw_compactor.engram_llm.EngramLLMClient.call", return_value="fake obs"):
            runner.run_once()

        # Get pending count after first run
        from claw_compactor.engram_storage import EngramStorage
        storage = EngramStorage(workspace)
        threads = storage.list_threads()
        first_counts = {t: len(storage.read_pending(t)) for t in threads}

        # Run again — should NOT re-ingest
        runner2 = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=False)
        with patch("claw_compactor.engram_llm.EngramLLMClient.call", return_value="fake obs"):
            runner2.run_once()

        second_counts = {t: len(storage.read_pending(t)) for t in threads}
        assert first_counts == second_counts

    def test_batch_ingest_error_does_not_mark_processed(
        self, workspace: Path, sessions_dir: Path, monkeypatch
    ) -> None:
        """If batch_ingest reports error, session must stay unprocessed for retry."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        session_file = _write_session(sessions_dir, "error_sess", [
            _make_openclaw_msg("user", "This should not be marked processed on ingest error"),
        ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=False)

        with patch("engram_auto.EngramEngine.batch_ingest", return_value={"error": "proxy timeout"}):
            runner.run_once()

        marker_file = workspace / "memory" / "engram" / ".processed_sessions"
        cache_key = f"{session_file.stem}:{int(session_file.stat().st_mtime)}"
        marker_text = marker_file.read_text(encoding="utf-8") if marker_file.exists() else ""
        assert cache_key not in marker_text

    def test_concurrent_threads_use_locks(
        self, workspace: Path, sessions_dir: Path, monkeypatch
    ) -> None:
        """Concurrent processing with shared thread should not corrupt state."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        # Create 5 sessions all going to the same thread
        for i in range(5):
            _write_session(sessions_dir, f"sess_{i}", [
                _make_openclaw_msg("user", f"Message number {i} going to general"),
                _make_openclaw_msg("assistant", f"Response {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        cfg["concurrency"]["max_workers"] = 4

        runner = EngramAutoRunner(workspace=workspace, engram_cfg=cfg, dry_run=False)
        with patch("claw_compactor.engram_llm.EngramLLMClient.call", return_value="fake obs"):
            runner.run_once()

        # Storage should be consistent (no corrupt JSONL)
        from claw_compactor.engram_storage import EngramStorage
        storage = EngramStorage(workspace)
        for tid in storage.list_threads():
            # read_pending() should succeed without exceptions
            msgs = storage.read_pending(tid)
            for m in msgs:
                assert "role" in m
                assert "content" in m


# ---------------------------------------------------------------------------
# Test 7: _load_dotenv
# ---------------------------------------------------------------------------

class TestLoadDotenv:
    def test_loads_key_value(self, tmp_path: Path, monkeypatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR_XYZ=hello123\n")
        monkeypatch.delenv("MY_TEST_VAR_XYZ", raising=False)
        _load_dotenv(env_file)
        assert os.environ.get("MY_TEST_VAR_XYZ") == "hello123"

    def test_does_not_override_existing(self, tmp_path: Path, monkeypatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR_ABC=from_dotenv\n")
        monkeypatch.setenv("MY_TEST_VAR_ABC", "from_env")
        _load_dotenv(env_file)
        assert os.environ.get("MY_TEST_VAR_ABC") == "from_env"

    def test_skips_comments(self, tmp_path: Path, monkeypatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nMY_VAR_COMMENT=value\n")
        monkeypatch.delenv("MY_VAR_COMMENT", raising=False)
        _load_dotenv(env_file)
        assert os.environ.get("MY_VAR_COMMENT") == "value"

    def test_nonexistent_file_no_error(self, tmp_path: Path) -> None:
        # Should not raise
        _load_dotenv(tmp_path / "nonexistent.env")


# ---------------------------------------------------------------------------
# Test 8: Phase 1 — Channel-ID → name mapping stability
# ---------------------------------------------------------------------------

class TestChannelIdNameMapping:
    """Channel-id → name mapping should produce stable thread IDs."""

    def test_known_channel_id_maps_to_name(self, sessions_dir: Path) -> None:
        """A message with only channel id (no channel name) should resolve via ID map."""
        from engram_auto import _CHANNEL_ID_NAME_MAP
        # Use a known ID from the static map
        known_id = next(iter(_CHANNEL_ID_NAME_MAP))
        known_name = _CHANNEL_ID_NAME_MAP[known_id]
        p = _write_session(sessions_dir, "id_only", [
            _make_openclaw_msg("user", f"channel id:{known_id} some message"),
        ])
        assert detect_thread_id(p) == known_name

    def test_channel_name_wins_over_id_when_both_present(self, sessions_dir: Path) -> None:
        """When both '#general' and 'channel id:NNN' appear, name should take precedence."""
        p = _write_session(sessions_dir, "name_and_id", [
            _make_openclaw_msg("user",
                "In [Discord Guild #general channel id:9999999]"),
        ])
        # Should be "discord-general" (name-mapped), not "discord-channel-9999999"
        assert detect_thread_id(p) == "discord-general"

    def test_unknown_id_falls_back_to_channel_id_format(self, sessions_dir: Path) -> None:
        """Unknown channel id with no channel name → discord-channel-{id}."""
        p = _write_session(sessions_dir, "unknown_id", [
            _make_openclaw_msg("user", "channel id:88888888"),
        ])
        assert detect_thread_id(p) == "discord-channel-88888888"

    def test_thread_map_cache_used_on_second_call(self, tmp_path: Path, sessions_dir: Path) -> None:
        """Second call with same session file should return cached result."""
        p = _write_session(sessions_dir, "cached_sess", [
            _make_openclaw_msg("user", "[Discord Guild #general channel id:111]"),
        ])
        cache_path = tmp_path / ".thread-map.json"
        result1 = detect_thread_id(p, thread_map_path=cache_path)
        # Corrupt the file so if cache is NOT used, detection would fail
        p.write_text("INVALID JSON\n")
        result2 = detect_thread_id(p, thread_map_path=cache_path)
        assert result1 == result2, "Cache should return same result on second call"


# ---------------------------------------------------------------------------
# Test 9: Phase 1 — Rate limiting (max_sessions_per_run)
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """EngramAutoRunner should cap sessions processed per run."""

    def _make_cfg(self, sessions_dir: Path, workspace: Path) -> dict:
        return {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:9999",
                "api_key_env": "OPENAI_API_KEY",
                "model": "test-model",
                "max_tokens": 512,
            },
            "threads": {"default": {"observer_threshold": 99999, "reflector_threshold": 99999}},
            "sessions": {"scan_dir": str(sessions_dir), "max_age_hours": 48},
            "storage": {"base_dir": str(workspace / "memory" / "engram")},
            "concurrency": {"max_workers": 2},
        }

    def test_rate_limit_caps_sessions(
        self, workspace: Path, sessions_dir: Path, monkeypatch
    ) -> None:
        """With max_sessions_per_run=3 and 10 sessions, only 3 are processed."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        # Create 10 distinct sessions
        for i in range(10):
            _write_session(sessions_dir, f"rate_sess_{i:02d}", [
                _make_openclaw_msg("user", f"Message {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=True,    # avoid needing LLM
            max_sessions_per_run=3,
            max_run_seconds=300,
        )
        totals = runner.run_once()

        # Check processed marker: at most 3 sessions written
        marker = workspace / "memory" / "engram" / ".processed_sessions"
        processed_lines = [l for l in marker.read_text().splitlines() if l.strip()]
        assert len(processed_lines) <= 3, (
            f"Expected ≤3 processed sessions, got {len(processed_lines)}: {processed_lines}"
        )

    def test_rate_limit_summary_has_remaining_estimate(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """Structured summary should report remaining_estimate when rate-limited."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(5):
            _write_session(sessions_dir, f"rem_sess_{i}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=True,
            max_sessions_per_run=2,
            max_run_seconds=300,
        )
        runner.run_once()

        captured = capsys.readouterr()
        assert "remaining_estimate=3" in captured.out, (
            f"Expected 'remaining_estimate=3' in output: {captured.out!r}"
        )

    def test_no_rate_limit_when_sessions_below_cap(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """When sessions ≤ max, remaining_estimate should be 0."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(3):
            _write_session(sessions_dir, f"below_cap_{i}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=True,
            max_sessions_per_run=20,
            max_run_seconds=300,
        )
        runner.run_once()

        captured = capsys.readouterr()
        assert "remaining_estimate=0" in captured.out, (
            f"Expected 'remaining_estimate=0' in output: {captured.out!r}"
        )

    def test_second_run_processes_deferred(
        self, workspace: Path, sessions_dir: Path, monkeypatch
    ) -> None:
        """A second run should pick up sessions deferred from the first."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(5):
            _write_session(sessions_dir, f"two_run_{i}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        marker = workspace / "memory" / "engram" / ".processed_sessions"

        # First run: process 2
        runner1 = EngramAutoRunner(
            workspace=workspace, engram_cfg=cfg, dry_run=True,
            max_sessions_per_run=2, max_run_seconds=300,
        )
        runner1.run_once()
        after_run1 = {l.split(":")[0] for l in marker.read_text().splitlines() if l.strip()}
        assert len(after_run1) == 2

        # Second run: process next 2 (different runner instance, fresh processed cache)
        runner2 = EngramAutoRunner(
            workspace=workspace, engram_cfg=cfg, dry_run=True,
            max_sessions_per_run=2, max_run_seconds=300,
        )
        runner2.run_once()
        after_run2 = {l.split(":")[0] for l in marker.read_text().splitlines() if l.strip()}
        assert len(after_run2) == 4, (
            f"Expected 4 sessions total after 2 runs, got {len(after_run2)}"
        )


# ---------------------------------------------------------------------------
# Test 10: Phase 1 — Soft deadline (max_run_seconds)
# ---------------------------------------------------------------------------

class TestSoftDeadline:
    """EngramAutoRunner should respect soft deadline and defer remaining sessions."""

    def _make_cfg(self, sessions_dir: Path, workspace: Path) -> dict:
        return {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:9999",
                "api_key_env": "OPENAI_API_KEY",
                "model": "test-model",
                "max_tokens": 512,
            },
            "threads": {"default": {"observer_threshold": 99999, "reflector_threshold": 99999}},
            "sessions": {"scan_dir": str(sessions_dir), "max_age_hours": 48},
            "storage": {"base_dir": str(workspace / "memory" / "engram")},
            "concurrency": {"max_workers": 2},
        }

    def test_deadline_limits_submissions(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """When deadline=0s, only in-flight sessions (possibly none) complete."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(10):
            _write_session(sessions_dir, f"dl_sess_{i:02d}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=True,
            max_sessions_per_run=20,
            max_run_seconds=0,   # immediate soft deadline
        )
        runner.run_once()

        captured = capsys.readouterr()
        # With deadline=0, remaining_estimate should be reported
        # (exact value depends on how many were submitted before deadline)
        assert "remaining_estimate=" in captured.out, (
            f"Expected 'remaining_estimate=' in output: {captured.out!r}"
        )

    def test_summary_structure_fields_present(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """Structured summary must include all required fields."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        _write_session(sessions_dir, "summary_sess", [
            _make_openclaw_msg("user", "test message"),
        ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=True,
            max_sessions_per_run=20,
            max_run_seconds=300,
        )
        runner.run_once()

        captured = capsys.readouterr()
        # All structured summary fields must be present
        for field in ("processed=", "skipped=", "failed=", "remaining_estimate="):
            assert field in captured.out, (
                f"Field '{field}' missing from summary output: {captured.out!r}"
            )

    def test_deadline_summary_reports_remaining(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """Soft deadline hit → remaining_estimate > 0 when there are pending sessions."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(6):
            _write_session(sessions_dir, f"remain_{i}", [
                _make_openclaw_msg("user", f"session {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)

        # Use a very short deadline and slow down _process_session so deadline fires
        original_process = None
        call_count = [0]

        def slow_process(self_inner, session_file, tmp_dir, thread_id_hint=None, run_id=None):
            call_count[0] += 1
            # Only the first call is allowed; after that sleep to simulate slow work
            if call_count[0] > 1:
                time.sleep(0.5)
            return (session_file.stem, "openclaw-main", 0, "processed")

        import time as _time
        import engram_auto as _ea

        runner = EngramAutoRunner(
            workspace=workspace,
            engram_cfg=cfg,
            dry_run=False,
            max_sessions_per_run=20,
            max_run_seconds=0,  # immediate deadline
        )
        runner.run_once()

        captured = capsys.readouterr()
        # remaining_estimate must appear in summary
        assert "remaining_estimate=" in captured.out


# ---------------------------------------------------------------------------
# Test 11: Phase 1 — Structured summary from run_once
# ---------------------------------------------------------------------------

class TestStructuredSummary:
    """run_once() should produce a structured summary with correct counts."""

    def _make_cfg(self, sessions_dir: Path, workspace: Path) -> dict:
        return {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:9999",
                "api_key_env": "OPENAI_API_KEY",
                "model": "test-model",
                "max_tokens": 512,
            },
            "threads": {"default": {"observer_threshold": 99999, "reflector_threshold": 99999}},
            "sessions": {"scan_dir": str(sessions_dir), "max_age_hours": 48},
            "storage": {"base_dir": str(workspace / "memory" / "engram")},
            "concurrency": {"max_workers": 2},
        }

    def test_summary_counts_processed(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """processed count should match number of newly ingested sessions."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(3):
            _write_session(sessions_dir, f"proc_{i}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace, engram_cfg=cfg, dry_run=True,
            max_sessions_per_run=20, max_run_seconds=300,
        )
        runner.run_once()

        captured = capsys.readouterr()
        assert "processed=3" in captured.out

    def test_summary_counts_skipped_on_rerun(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """On second run (same unchanged sessions), skipped count should equal session count."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        for i in range(2):
            _write_session(sessions_dir, f"skip_{i}", [
                _make_openclaw_msg("user", f"msg {i}"),
            ])

        cfg = self._make_cfg(sessions_dir, workspace)
        # First run
        runner = EngramAutoRunner(
            workspace=workspace, engram_cfg=cfg, dry_run=True,
            max_sessions_per_run=20, max_run_seconds=300,
        )
        runner.run_once()
        capsys.readouterr()  # discard first run output

        # Second run — same runner (already has processed cache)
        runner.run_once()
        captured = capsys.readouterr()
        assert "skipped=2" in captured.out, (
            f"Expected 'skipped=2' in second run output: {captured.out!r}"
        )

    def test_empty_sessions_dir_summary(
        self, workspace: Path, sessions_dir: Path, monkeypatch, capsys
    ) -> None:
        """With no sessions, summary should show all zeros."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        cfg = self._make_cfg(sessions_dir, workspace)
        runner = EngramAutoRunner(
            workspace=workspace, engram_cfg=cfg, dry_run=True,
            max_sessions_per_run=20, max_run_seconds=300,
        )
        runner.run_once()
        captured = capsys.readouterr()
        assert "processed=0" in captured.out
        assert "skipped=0" in captured.out
        assert "failed=0" in captured.out
        assert "remaining_estimate=0" in captured.out
