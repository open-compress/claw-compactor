"""Tests for lib.config module."""

import json
from pathlib import Path

import pytest

from claw_compactor.config import MemCompressConfig, load_config, DEFAULT_CONFIG


class TestMemCompressConfig:
    def test_defaults(self) -> None:
        cfg = MemCompressConfig()
        assert cfg.chars_per_token == 4
        assert cfg.level0_max_tokens == 200

    def test_custom_values(self) -> None:
        cfg = MemCompressConfig(chars_per_token=3)
        assert cfg.chars_per_token == 3


class TestLoadConfig:
    def test_no_config_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == DEFAULT_CONFIG["chars_per_token"]

    def test_valid_config(self, tmp_path: Path) -> None:
        (tmp_path / "claw-compactor-config.json").write_text(
            json.dumps({"chars_per_token": 3}), encoding="utf-8"
        )
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == 3

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "claw-compactor-config.json").write_text("{{bad", encoding="utf-8")
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == 4

    def test_non_dict(self, tmp_path: Path) -> None:
        (tmp_path / "claw-compactor-config.json").write_text('["arr"]', encoding="utf-8")
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == 4

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "claw-compactor-config.json").write_text(
            json.dumps({"chars_per_token": 5, "bogus": 99}), encoding="utf-8"
        )
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == 5

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "claw-compactor-config.json").write_text("", encoding="utf-8")
        cfg = load_config(tmp_path)
        assert cfg.chars_per_token == 4
