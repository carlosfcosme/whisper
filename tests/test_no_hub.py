"""Whisper does not pull weights from the Hugging Face Hub."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.hub import HubPullError, is_hub_url, refuse_hub_pull

REPO_ROOT = Path(__file__).resolve().parents[1]

HUB_URLS = [
    "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
    "https://www.huggingface.co/openai/whisper-tiny",
    "https://hf.co/openai/whisper-tiny",
    "https://cdn-lfs.huggingface.co/repos/foo/model.safetensors",
    "https://cas-bridge.xethub.hf.co/openai/whisper-tiny/resolve/main/model.pt",
]


@pytest.mark.parametrize("url", HUB_URLS)
def test_refuse_hub_urls(url):
    assert is_hub_url(url)
    with pytest.raises(HubPullError, match="Hub"):
        refuse_hub_pull(url)


def test_official_cdn_is_not_hub():
    for url in whisper._MODELS.values():
        assert not is_hub_url(url)
        refuse_hub_pull(url)


def test_download_refuses_hub_before_network(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network should not be used for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HubPullError):
        whisper._download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
            str(tmp_path),
            False,
        )
    assert list(tmp_path.rglob("*")) == []


def test_load_model_refuses_hub_name(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network should not be used for Hub names")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HubPullError):
        whisper.load_model(
            "https://huggingface.co/openai/whisper-tiny",
            download_root=str(tmp_path),
        )


def test_no_huggingface_hub_dependency():
    manifests = [
        (REPO_ROOT / "pyproject.toml").read_text(),
        (REPO_ROOT / "requirements.txt").read_text(),
    ]
    combined = "\n".join(manifests).lower()
    assert "huggingface" not in combined
    assert "hf_hub" not in combined
