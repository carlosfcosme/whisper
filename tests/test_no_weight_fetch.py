"""Regression: tests must not fetch model weights.

Executable sovereignty check — attempting a named-checkpoint or Hub/CDN
download must raise, and no weight file may be written. Local cache/weight
paths stay gitignored. This module never downloads weights.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]

AZURE_WEIGHT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiny.pt"
)
HUB_WEIGHT_URL = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
)

WEIGHT_IGNORE_PATHS = (
    "weights/tiny.pt",
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "whisper_cache/tiny.pt",
    "checkpoints/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
)


def test_load_model_named_tiny_cannot_fetch_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline|Hub|forbidden"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.bin")) == []


def test_download_refuses_weight_url_and_writes_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="offline|Hub|forbidden"):
        whisper._download(AZURE_WEIGHT_URL, str(tmp_path), in_memory=False)
    assert list(tmp_path.glob("*")) == []


def test_urlopen_of_hub_or_cdn_weight_fails():
    with pytest.raises(RuntimeError, match="Hub|forbidden|intercept"):
        urllib.request.urlopen(AZURE_WEIGHT_URL)
    with pytest.raises(RuntimeError, match="Hub|forbidden|intercept"):
        urllib.request.urlopen(HUB_WEIGHT_URL)


def test_weight_and_cache_artifacts_are_gitignored():
    failed = []
    for path in WEIGHT_IGNORE_PATHS:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], "weight/cache paths must be gitignored: {}".format(failed)


def test_offline_env_forbids_weight_download():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert whisper.weights_download_forbidden() is True
