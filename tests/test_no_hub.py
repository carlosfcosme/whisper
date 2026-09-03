"""Hugging Face Hub downloads are disabled. Does not load weights."""

import os
from pathlib import Path

import pytest

import whisper
from whisper.hub import HubError, is_hub_url, refuse_hub_url

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_official_cdn_is_not_hub():
    for name in whisper.available_models():
        assert not is_hub_url(whisper._MODELS[name])


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://hf.co/openai/whisper-tiny",
        "https://cdn-lfs.huggingface.co/repos/xx/tiny.pt",
    ],
)
def test_hub_urls_are_detected(url):
    assert is_hub_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
        "https://hf.co/openai/whisper-tiny",
        "https://cdn-lfs.huggingface.co/repos/xx/tiny.pt",
    ],
)
def test_download_refuses_hub_urls(url, tmp_path):
    with pytest.raises(HubError, match="Hub"):
        whisper._download(url, str(tmp_path), False)
    assert list(tmp_path.iterdir()) == []


def test_load_model_refuses_hub_name():
    with pytest.raises(HubError, match="Hub"):
        whisper.load_model("https://huggingface.co/openai/whisper-tiny")


def test_loopback_url_is_not_hub():
    assert not is_hub_url("http://127.0.0.1:8765/tiny.pt")
    refuse_hub_url("http://127.0.0.1:8765/tiny.pt")


def test_package_files_do_not_depend_on_hub_clients():
    for name in ("pyproject.toml", "requirements.txt"):
        text = (REPO_ROOT / name).read_text().lower()
        assert "huggingface" not in text
        assert "hf_hub" not in text
        assert "transformers" not in text
