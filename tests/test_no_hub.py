import urllib.request

import pytest


def test_huggingface_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hub or model weights"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_hf_co_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hub or model weights"):
        urllib.request.urlopen("https://hf.co/openai/whisper-tiny")


def test_whisper_cdn_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hub or model weights"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt"
        )


def test_offline_hub_env_is_set():
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("WHISPER_OFFLINE") == "1"
