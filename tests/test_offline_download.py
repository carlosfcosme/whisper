"""Offline load_model / _download. Does not pull weights or contact the Hub."""

from __future__ import annotations

import pytest

import whisper
from whisper.device import default_device


def test_load_model_defaults_to_cpu_without_download(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_device() == "cpu"
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []


def test_download_refuses_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper._download(
            "https://example.invalid/deadbeef/tiny.pt", str(tmp_path), False
        )
    assert not (tmp_path / "tiny.pt").exists()
