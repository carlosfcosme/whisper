import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

import whisper

ROOT = Path(__file__).resolve().parents[1]

# Paths that must stay untracked: checkpoints and Hub/torch caches.
_IGNORED_PATHS = (
    "tiny.pt",
    "large-v3.pt",
    "model.safetensors",
    "pytorch_model.bin",
    ".cache/whisper/tiny.pt",
    ".cache/huggingface/hub/models--openai--whisper/refs/main",
    ".huggingface/hub/models--foo/snapshots/x/config.json",
    "hub/trusted_list",
    ".tox/offline/log/offline-0.log",
)


def test_gitignore_covers_weight_and_cache_paths():
    for rel in _IGNORED_PATHS:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel],
            cwd=ROOT,
        )
        assert proc.returncode == 0, f"{rel} is not covered by .gitignore"


def test_offline_env_flags_are_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WHISPER_BIND_HOST") == "127.0.0.1"


def test_download_refuses_missing_checkpoint_without_urlopen(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    opened = []

    def boom(url, *args, **kwargs):
        opened.append(url)
        raise AssertionError("urlopen must not run in offline mode")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError, match="Offline mode"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)

    assert opened == []
    assert list(tmp_path.glob("*.pt")) == []


def test_load_model_does_not_fetch_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="Offline mode"):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/api/models",
        "https://cdn-lfs.huggingface.co/repos/example",
        "https://huggingface.co/openai/whisper-tiny",
        "https://cas-bridge.xethub.hf.co/xet/example",
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
    ],
)
def test_urlopen_blocks_hub_and_weight_hosts(url):
    with pytest.raises(RuntimeError, match="blocked Hub/weight fetch"):
        urllib.request.urlopen(url)
