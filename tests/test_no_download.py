import urllib.request

import pytest

import whisper


def test_conftest_sets_no_download():
    import os

    assert os.environ.get("WHISPER_NO_DOWNLOAD") == "1"


def test_download_raises_without_network_when_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_NO_DOWNLOAD", "1")

    def _forbid(*args, **kwargs):
        raise AssertionError("urlopen must not be called when WHISPER_NO_DOWNLOAD=1")

    monkeypatch.setattr(urllib.request, "urlopen", _forbid)
    url = "https://example.invalid/{}/tiny.pt".format("a" * 64)
    with pytest.raises(RuntimeError, match="WHISPER_NO_DOWNLOAD"):
        whisper._download(url, str(tmp_path), in_memory=False)


def test_load_model_does_not_fetch_named_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_NO_DOWNLOAD", "1")

    def _forbid(*args, **kwargs):
        raise AssertionError("urlopen must not be called when WHISPER_NO_DOWNLOAD=1")

    monkeypatch.setattr(urllib.request, "urlopen", _forbid)
    with pytest.raises(RuntimeError, match="WHISPER_NO_DOWNLOAD"):
        whisper.load_model("tiny", download_root=str(tmp_path))
