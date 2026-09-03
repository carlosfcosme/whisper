import inspect
import os
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.offline import WeightDownloadError, refuse_weight_fetch


def test_download_source_has_no_urlopen():
    source = inspect.getsource(whisper._download)
    assert "urlopen" not in source
    assert "WeightDownloadError" in source


def test_named_model_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError, match="weight pull is disabled"):
        whisper.load_model("tiny")
    assert called == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_forbidden_network_url_does_not_download(
    forbidden_network_url, tmp_path, monkeypatch
):
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        refuse_weight_fetch(forbidden_network_url)
    with pytest.raises(WeightDownloadError):
        whisper._download(forbidden_network_url, str(tmp_path), False)
    assert called == []
    assert list(Path(tmp_path).rglob("*")) == []


def test_load_model_refuses_hub_name(forbidden_network_url, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises((WeightDownloadError, RuntimeError)):
        whisper.load_model(forbidden_network_url)
    assert called == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_checksum_mismatch_does_not_redownload(tmp_path):
    url = whisper._MODELS["tiny"]
    target = tmp_path / os.path.basename(url)
    target.write_bytes(b"not-the-official-checkpoint")
    with pytest.raises(WeightDownloadError, match="weight pull is disabled"):
        whisper._download(url, str(tmp_path), False)
    assert target.read_bytes() == b"not-the-official-checkpoint"


def test_urlopen_guard_blocks_wan():
    with pytest.raises(RuntimeError, match="must not download"):
        urllib.request.urlopen("https://example.com/tiny.pt")
