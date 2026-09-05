import os
import urllib.request

import pytest

import whisper


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
        "https://hf.co/openai/whisper-tiny",
    ],
)
def test_urlopen_blocks_hub_and_cdn(url):
    with pytest.raises(RuntimeError, match="must not Hub"):
        urllib.request.urlopen(url)


def test_load_model_named_checkpoint_does_not_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline"):
        whisper.load_model("tiny")
    leftover = list(tmp_path.rglob("*.pt"))
    assert leftover == []
