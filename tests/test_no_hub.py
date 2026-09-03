import urllib.request

import pytest

import whisper


def test_huggingface_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hub/CDN"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_whisper_cdn_download_is_blocked(tmp_path):
    url = whisper._MODELS["tiny"]
    assert "openaipublic.azureedge.net" in url
    with pytest.raises(RuntimeError, match="Offline/no-Hub|Hub/CDN"):
        whisper._download(url, str(tmp_path), False)


def test_load_model_does_not_fetch_from_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="Offline/no-Hub|Hub/CDN"):
        whisper.load_model("tiny")
