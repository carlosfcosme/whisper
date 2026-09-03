import pytest

from whisper import _MODELS, _download
from whisper.runtime import (
    WeightDownloadError,
    is_hf_hub_url,
    refuse_weight_auto_download,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://hf.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "https://cdn-lfs.huggingface.co/repos/abc/model.safetensors",
    ],
)
def test_detects_hf_hub_urls(url):
    assert is_hf_hub_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
        "https://example.com/tiny.pt",
    ],
)
def test_non_hub_urls_are_not_flagged(url):
    assert not is_hf_hub_url(url)


def test_official_models_are_not_on_hub():
    for name, url in _MODELS.items():
        assert not is_hf_hub_url(url), name


def test_refuse_hub_url_always(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(url)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        _download(url, str(tmp_path), in_memory=False)


def test_no_weight_download_env_blocks_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    url = "https://openaipublic.azureedge.net/main/whisper/models/deadbeef/tiny.pt"
    with pytest.raises(WeightDownloadError, match="Auto-download"):
        _download(url, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []
