"""Ticket 4: CPU-only / offline / no-store defaults."""

from pathlib import Path

import pytest

import whisper
from whisper.runtime import (
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    no_store_enabled,
    offline_enabled,
    refuse_remote_download,
)

OFFLINE_KEYS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "WHISPER_NO_STORE",
    "WHISPER_ALLOW_DOWNLOAD",
    "WHISPER_ALLOW_WEIGHT_FETCH",
    "WHISPER_ALLOW_STORE",
)


def _clear_policy_env(monkeypatch):
    for key in OFFLINE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_cpu_offline_no_store_are_defaults(monkeypatch):
    _clear_policy_env(monkeypatch)
    assert DEFAULT_DEVICE == "cpu"
    assert DEFAULT_OFFLINE is True
    assert DEFAULT_NO_STORE is True
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_OFFLINE is True
    assert whisper.DEFAULT_NO_STORE is True
    assert offline_enabled() is True
    assert no_store_enabled() is True


def test_offline_default_refuses_download_without_env(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    dest = str(tmp_path / "tiny.pt")
    with pytest.raises(RuntimeError, match="offline"):
        refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            dest,
        )
    assert not Path(dest).exists()


def test_no_store_refuses_persist_when_download_allowed(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    dest = str(tmp_path / "tiny.pt")
    with pytest.raises(RuntimeError, match="no-store"):
        refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            dest,
        )
    assert not Path(dest).exists()


def test_load_model_default_does_not_store_weights(monkeypatch, tmp_path):
    _clear_policy_env(monkeypatch)
    with pytest.raises(RuntimeError, match="offline|no-store|no Hub"):
        whisper.load_model("tiny", device="cpu", download_root=str(tmp_path))
    assert list(tmp_path.glob("*.pt")) == []


def test_no_store_can_be_opted_out(monkeypatch):
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WHISPER_NO_STORE", "0")
    assert no_store_enabled() is False
    monkeypatch.setenv("WHISPER_ALLOW_STORE", "1")
    assert no_store_enabled() is False


def test_no_keys_and_no_field_brain():
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "whisper" / "runtime.py").read_text()
    serve = (root / "whisper" / "serve.py").read_text()
    for text in (runtime, serve):
        assert "FIELD_BRAIN" not in text
        assert "Field-Brain" not in text
        assert "API_KEY" not in text
        assert "sk-" not in text
    from whisper.runtime import BIND_HOST

    assert BIND_HOST == "127.0.0.1"
    assert BIND_HOST != "0.0.0.0"
