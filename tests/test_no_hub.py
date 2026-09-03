import os
from pathlib import Path

import pytest

from whisper import _download
from whisper.hub import HubDisabledError, assert_not_hub_url

REPO = Path(__file__).resolve().parents[1]


def test_official_azure_url_is_allowed():
    assert_not_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://hf.co/openai/whisper-tiny",
        "https://cdn-lfs.huggingface.co/repos/xx/model.safetensors",
    ],
)
def test_hub_urls_are_rejected(url):
    with pytest.raises(HubDisabledError, match="Hub"):
        assert_not_hub_url(url)


def test_download_rejects_hub_before_network(tmp_path):
    with pytest.raises(HubDisabledError):
        _download("https://huggingface.co/openai/whisper-tiny", str(tmp_path), False)
    assert list(tmp_path.iterdir()) == []


def test_package_has_no_huggingface_imports():
    for path in (REPO / "whisper").rglob("*.py"):
        if path.name == "hub.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "huggingface" not in text
        assert "from_pretrained" not in text


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
