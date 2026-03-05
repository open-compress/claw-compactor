"""Tests for the unified entry point (mem_compress.py)."""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mem_compress import build_parser, COMMAND_MAP, _workspace_path, _count_tokens_in_workspace


class TestBuildParser:
    """Test argument parser construction."""

    def test_parser_accepts_all_commands(self):
        parser = build_parser()
        for cmd in ["compress", "estimate", "dedup", "tiers", "audit",
                     "observe", "dict", "optimize", "full", "benchmark"]:
            args = parser.parse_args(["/tmp/ws", cmd])
            assert args.command == cmd
            assert args.workspace == "/tmp/ws"

    def test_parser_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["/tmp/ws", "estimate", "--json"])
        assert args.json is True

    def test_parser_dry_run(self):
        parser = build_parser()
        args = parser.parse_args(["/tmp/ws", "compress", "--dry-run"])
        assert args.dry_run is True

    def test_parser_since(self):
        parser = build_parser()
        args = parser.parse_args(["/tmp/ws", "observe", "--since", "2026-01-01"])
        assert args.since == "2026-01-01"

    def test_parser_threshold_val(self):
        parser = build_parser()
        args = parser.parse_args(["/tmp/ws", "dedup", "--threshold-val", "0.8"])
        assert args.threshold_val == 0.8

    def test_parser_rejects_unknown_command(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/tmp/ws", "unknown"])

    def test_parser_requires_workspace(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["estimate"])


class TestCommandMap:
    """Test command routing."""

    def test_all_commands_have_handlers(self):
        expected = {"compress", "estimate", "dedup", "tiers", "audit",
                    "observe", "dict", "optimize", "full", "benchmark", "install", "auto", "engram"}
        assert set(COMMAND_MAP.keys()) == expected

    def test_handlers_are_callable(self):
        for name, handler in COMMAND_MAP.items():
            assert callable(handler), f"{name} handler not callable"


class TestWorkspacePath:
    """Test workspace validation."""

    def test_valid_workspace(self, tmp_path):
        result = _workspace_path(str(tmp_path))
        assert result == tmp_path

    def test_invalid_workspace(self):
        with pytest.raises(SystemExit):
            _workspace_path("/nonexistent/path/xyz")


class TestCountTokens:
    """Test workspace token counting."""

    def test_empty_workspace(self, tmp_path):
        tokens = _count_tokens_in_workspace(tmp_path)
        assert tokens == 0

    def test_workspace_with_files(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("Hello world this is a test of token counting")
        tokens = _count_tokens_in_workspace(tmp_path)
        assert tokens > 0

    def test_workspace_with_memory_dir(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "2026-01-01.md").write_text("Some daily notes here")
        tokens = _count_tokens_in_workspace(tmp_path)
        assert tokens > 0


class TestEstimateCommand:
    """Test estimate command."""

    def test_estimate_json(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Test\nSome content here")
        from mem_compress import cmd_estimate
        import argparse
        args = argparse.Namespace(json=True, threshold=0)
        result = cmd_estimate(tmp_path, args)
        assert result == 0

    def test_estimate_human(self, tmp_path, capsys):
        (tmp_path / "MEMORY.md").write_text("# Test\nSome content")
        from mem_compress import cmd_estimate
        import argparse
        args = argparse.Namespace(json=False, threshold=0)
        result = cmd_estimate(tmp_path, args)
        assert result == 0
        out = capsys.readouterr().out
        assert "token" in out.lower() or "Token" in out or len(out) > 0


class TestCompressCommand:
    """Test compress command."""

    def test_compress_dry_run(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "test.md").write_text("# Test\n\nHello world\n\nHello world\n")
        from mem_compress import cmd_compress
        import argparse
        args = argparse.Namespace(json=False, dry_run=True, older_than=None)
        result = cmd_compress(tmp_path, args)
        assert result == 0


class TestDedupCommand:
    """Test dedup command."""

    def test_dedup_json(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "a.md").write_text("# Section\nSome content about X and Y\n")
        (mem / "b.md").write_text("# Section\nSome content about X and Y\n")
        from mem_compress import cmd_dedup
        import argparse
        args = argparse.Namespace(json=True, auto_merge=False, threshold_val=0.6)
        result = cmd_dedup(tmp_path, args)
        assert result == 0


class TestBenchmarkCommand:
    """Test benchmark command."""

    def test_benchmark_human(self, tmp_path, capsys):
        (tmp_path / "MEMORY.md").write_text("# Memory\nSome content\n## Section\nMore content\n")
        (tmp_path / "TOOLS.md").write_text("# Tools\n| Key | Value |\n|-----|-------|\n| a | b |\n")
        from mem_compress import cmd_benchmark
        import argparse
        args = argparse.Namespace(json=False)
        result = cmd_benchmark(tmp_path, args)
        assert result == 0
        out = capsys.readouterr().out
        assert "Performance Report" in out
        assert "Rule Engine" in out
        assert "TOTAL" in out

    def test_benchmark_json(self, tmp_path, capsys):
        (tmp_path / "MEMORY.md").write_text("# Memory\nContent\n")
        from mem_compress import cmd_benchmark
        import argparse
        args = argparse.Namespace(json=True)
        result = cmd_benchmark(tmp_path, args)
        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert "total_before" in data
        assert "total_after" in data
        assert "steps" in data
