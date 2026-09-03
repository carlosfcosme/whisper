import urllib.request

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
    "https://cdn-lfs.huggingface.co/repos/tiny/model.safetensors",
    "https://cas-bridge.xethub.hf.co/openai/whisper-tiny/resolve/main/model.bin",
)


@pytest.mark.parametrize("url", HF_URLS)
def test_hf_hub_urls_are_detected(url):
    assert is_hf_hub_url(url) is True


def test_official_azure_url_is_not_hf_hub():
    assert is_hf_hub_url(whisper._MODELS["tiny"]) is False


@pytest.mark.parametrize("url", HF_URLS)
def test_hf_hub_download_always_refused(url, monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(url)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(url, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


def test_named_model_load_does_not_hit_hub_or_wan(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("load_model must not open the network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_huggingface_hub_is_not_a_project_dependency():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    requirements = (root / "requirements.txt").read_text()
    assert "huggingface" not in pyproject.lower()
    assert "huggingface" not in requirements.lower()
