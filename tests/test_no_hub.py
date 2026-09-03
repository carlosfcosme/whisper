import urllib.request

import pytest

import whisper
from whisper.runtime import (
    WeightDownloadError,
    is_hf_hub_url,
    refuse_weight_auto_download,
)

HF_HUB_URL = (
    "https://"
    + "huggingface.co"
    + "/openai/whisper-tiny/resolve/main/pytorch_model.bin"
)


def _forbid_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unit tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "build_opener", boom)


def test_hf_hub_url_is_detected():
    assert is_hf_hub_url(HF_HUB_URL)
    assert not is_hf_hub_url(whisper._MODELS["tiny"])


def test_refuse_hf_hub_and_ci_auto_download(monkeypatch, tmp_path):
    _forbid_network(monkeypatch)

    with pytest.raises(WeightDownloadError, match="Hub"):
        refuse_weight_auto_download(HF_HUB_URL)

    with pytest.raises(WeightDownloadError, match="Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    with pytest.raises(WeightDownloadError):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []
