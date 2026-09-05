import pytest

import whisper
from whisper.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    no_store_enabled,
    offline_enabled,
)


def test_ticket4_cpu_offline_no_store_defaults():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert DEFAULT_OFFLINE is True
    assert DEFAULT_NO_STORE is True
    assert whisper.DEFAULT_OFFLINE is True
    assert whisper.DEFAULT_NO_STORE is True
    assert offline_enabled() is True
    assert no_store_enabled() is True


def test_named_load_is_offline_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_NO_STORE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*")) == []


def test_no_store_refuses_persist_without_mkdir(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("WHISPER_NO_STORE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="no-store"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*")) == []
    assert not (tmp_path / "whisper").exists()


def test_offline_opt_out_still_no_store(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("WHISPER_NO_STORE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="no-store"):
        whisper.load_model("base.en")
    leftover = list(tmp_path.rglob("*.pt"))
    assert leftover == []
