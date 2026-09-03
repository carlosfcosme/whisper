"""Regression: tests and CI must not fetch model weights."""

import os
import subprocess
from pathlib import Path

import pytest

import whisper
from whisper.runtime import WeightFetchError

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
GITIGNORE = REPO_ROOT / ".gitignore"
AZURE_TINY = "https://openaipublic.azureedge.net/main/whisper/models/deadbeef/tiny.pt"
HUB_TINY = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"

REQUIRED_IGNORE = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
)


def test_named_load_raises_weight_fetch_error_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("WHISPER_NO_STORE", "1")

    def _boom(*args, **kwargs):
        raise AssertionError("urlopen must not run when a weight fetch is attempted")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(WeightFetchError, match="offline|no-store|no Hub"):
        whisper.load_model(
            "tiny", device="cpu", download_root=str(tmp_path / "whisper")
        )
    leftover = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftover == []


def test_download_miss_is_a_weight_fetch_error(tmp_path, monkeypatch):
    called = []

    def _boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("weight fetch urlopen")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    monkeypatch.setattr(os, "makedirs", _boom)
    with pytest.raises(WeightFetchError):
        whisper._download(AZURE_TINY, str(tmp_path / "cache"), in_memory=False)
    assert called == []
    assert not (tmp_path / "cache").exists()


def test_hub_weight_url_is_a_weight_fetch_error():
    with pytest.raises(WeightFetchError, match="Hub"):
        whisper.refuse_remote_download(HUB_TINY, "/tmp/tiny.pt")


def test_gitignore_covers_weight_and_cache_dirs():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in REQUIRED_IGNORE:
        assert pattern in text
    for example in IGNORE_EXAMPLES:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", example],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, "expected gitignore to match {}".format(example)


def test_ci_skips_weight_downloads():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "WHISPER_NO_STORE" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "assert_no_weight_download.py" in workflow
    assert "check_cache_weights.sh" in workflow
