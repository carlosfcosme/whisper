import os
import urllib.request

import pytest


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_huggingface_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper")


def test_huggingface_hub_api_is_blocked_if_installed():
    huggingface_hub = pytest.importorskip("huggingface_hub")
    with pytest.raises(RuntimeError, match="huggingface_hub"):
        huggingface_hub.hf_hub_download("openai/whisper", "config.json")
