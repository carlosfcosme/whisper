import os
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.offline import (
    WeightDownloadError,
    is_hub_url,
    offline_enabled,
    refuse_remote_download,
    weight_pull_allowed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def test_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert offline_enabled()
    assert weight_pull_allowed() is False


def test_is_hub_url_detects_huggingface():
    assert is_hub_url("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_url("https://hf.co/openai/whisper-tiny")
    assert is_hub_url(HF_HUB_URL)
    assert not is_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    )


def test_refuse_hub_download():
    with pytest.raises(WeightDownloadError, match="no Hub"):
        refuse_remote_download(HF_HUB_URL, "/tmp/missing.pt")


def test_refuse_weight_pull(tmp_path):
    dest = str(tmp_path / "missing.pt")
    with pytest.raises(WeightDownloadError, match="no weight pull"):
        refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt", dest
        )


def test_download_named_model_does_not_pull(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(WeightDownloadError, match="no Hub|no weight pull"):
        whisper.load_model("tiny")
    assert list(tmp_path.iterdir()) == [] or not any(tmp_path.rglob("*.pt"))


def test_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_whisper_package_does_not_import_huggingface_hub():
    for path in (REPO_ROOT / "whisper").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "huggingface_hub" not in text, path
        assert "from_pretrained" not in text, path
