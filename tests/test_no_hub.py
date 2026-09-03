import os

import pytest

import whisper
from whisper.runtime import (
    WeightDownloadError,
    is_hf_hub_url,
    refuse_weight_auto_download,
)

HF_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    "https://hf.co/openai/whisper-tiny/resolve/main/model.safetensors",
    "https://cdn.huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
)


@pytest.mark.parametrize("url", HF_URLS)
def test_is_hf_hub_url(url):
    assert is_hf_hub_url(url) is True


def test_official_azure_url_is_not_hub():
    assert is_hf_hub_url(whisper._MODELS["tiny"]) is False


@pytest.mark.parametrize("url", HF_URLS)
def test_refuse_hub_even_when_downloads_allowed(monkeypatch, url):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_CPU_ONLY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(url)


def test_tests_set_hub_offline():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_huggingface_hub_is_patched_when_installed():
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    with pytest.raises(RuntimeError, match="Hub"):
        huggingface_hub.hf_hub_download("openai/whisper-tiny", "pytorch_model.bin")
