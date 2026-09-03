import urllib.request

import pytest

import whisper
from whisper.offline import (
    ALLOW_WEIGHT_DOWNLOAD_ENV,
    WeightDownloadError,
    refuse_weight_auto_download,
    weight_auto_download_allowed,
)


def test_auto_download_disallowed_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    assert weight_auto_download_allowed() is False


def test_offline_env_wins_over_allow(monkeypatch):
    monkeypatch.setenv(ALLOW_WEIGHT_DOWNLOAD_ENV, "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    assert weight_auto_download_allowed() is False


def test_named_model_default_load_does_not_call_urlopen(monkeypatch, tmp_path):
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urllib.request.urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    with pytest.raises(WeightDownloadError, match="disabled by default"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_local_checkpoint_path_does_not_call_download(monkeypatch, tmp_path):
    path = tmp_path / "local.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("_download must not run for a local path")

    monkeypatch.setattr(whisper, "_download", boom)
    with pytest.raises(Exception):
        whisper.load_model(str(path))
    assert calls == []


def test_download_helper_refuses_before_urlopen(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    url = "https://openaipublic.azureedge.net/main/whisper/models/deadbeef/tiny.pt"
    with pytest.raises(WeightDownloadError):
        refuse_weight_auto_download(url)
    with pytest.raises(WeightDownloadError):
        whisper._download(url, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []
