import os
import urllib.request

import pytest


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_urlopen_to_hub_is_blocked():
    with pytest.raises(RuntimeError, match="Hub"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_urlopen_to_hf_co_is_blocked():
    with pytest.raises(RuntimeError, match="Hub"):
        urllib.request.urlopen("https://hf.co/models")


def test_urlopen_weight_cdn_is_blocked():
    with pytest.raises(RuntimeError, match="download weights"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_huggingface_hub_stays_offline_if_installed():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
