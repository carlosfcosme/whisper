import urllib.request

import pytest

import whisper


def test_hub_url_is_refused(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("Hub urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(url, str(tmp_path), False)
    assert list(tmp_path.glob("*.pt")) == []


def test_hf_co_url_is_refused(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("Hub urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    url = "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(url, str(tmp_path), False)


def test_hub_host_helper():
    assert whisper._hub_host("https://huggingface.co/x/y") == "huggingface.co"
    assert whisper._hub_host("https://cdn.huggingface.co/x") == "cdn.huggingface.co"
    assert whisper._hub_host("https://openaipublic.azureedge.net/x") is None
