import urllib.request

import pytest

import whisper
from whisper.runtime import WeightDownloadError, refuse_weight_auto_download

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def test_hf_hub_url_is_refused():
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(HF_HUB_URL)


def test_download_refuses_hf_hub(tmp_path):
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


def test_load_model_named_checkpoint_does_not_hit_hub(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unit tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
