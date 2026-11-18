"""Tests for the orb's single-file TOML configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.orb import config as orb_config


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orb_config, "DATA_DIR", tmp_path / "orb")


def test_creates_default_file_on_first_run(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.toml"
    cfg = orb_config.load_config(path)
    assert path.exists()
    assert cfg == orb_config.JarvisConfig()
    assert (tmp_path / "orb").is_dir()


def test_default_file_matches_dataclass_defaults(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.toml"
    path.write_text(orb_config.DEFAULT_TOML, encoding="utf-8")
    assert orb_config.load_config(path) == orb_config.JarvisConfig()


def test_partial_override_keeps_other_defaults(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.toml"
    path.write_text(
        '[brain]\nmodel = "qwen3.5:2b"\n[orb]\nsize = 300\nunknown_key = 1\n',
        encoding="utf-8",
    )
    cfg = orb_config.load_config(path)
    assert cfg.brain.model == "qwen3.5:2b"
    assert cfg.brain.temperature == orb_config.BrainConfig().temperature
    assert cfg.orb.size == 300
    assert cfg.voice == orb_config.VoiceConfig()
