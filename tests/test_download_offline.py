"""_download must refuse Hub and offline weight pulls."""

import pytest

import whisper


def test_download_refuses_hub_url(tmp_path):
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
            str(tmp_path),
            False,
        )


def test_download_refuses_cdn_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert called == []
