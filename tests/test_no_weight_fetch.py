"""Regression: tests must not fetch model weights.

Fails if ``urlopen`` is reached for a Hub/CDN checkpoint. Does not download.
Also asserts local weight/cache paths stay gitignored.
"""

import subprocess
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.offline import WeightDownloadError

REPO_ROOT = Path(__file__).resolve().parents[1]

WEIGHT_CACHE_IGNORES = (
    "*.pt",
    "weights/",
    "cache/",
    ".cache/",
    "checkpoints/",
)

IGNORED_WEIGHT_PATHS = (
    "tiny.pt",
    "weights/tiny.pt",
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    "checkpoints/model.safetensors",
)


def _request_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, "full_url", None) or str(url)


def _fail_if_weight_fetch(attempted):
    def _boom(url, *args, **kwargs):
        target = _request_url(url)
        attempted.append(target)
        raise AssertionError("model-weight fetch attempted: {}".format(target))

    return _boom


def test_load_model_never_fetches_weights(tmp_path, monkeypatch):
    attempted = []
    boom = _fail_if_weight_fetch(attempted)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)

    with pytest.raises(WeightDownloadError, match="no Hub|no weight pull"):
        whisper.load_model("tiny")

    assert attempted == [], "weight urlopen must not run: {}".format(attempted)
    assert list(tmp_path.iterdir()) == []
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_download_refuses_before_urlopen(tmp_path, monkeypatch):
    attempted = []
    boom = _fail_if_weight_fetch(attempted)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    monkeypatch.delenv("WHISPER_ALLOW_WEIGHT_DOWNLOAD", raising=False)

    with pytest.raises(WeightDownloadError, match="no Hub|no weight pull"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)

    assert attempted == [], "weight urlopen must not run: {}".format(attempted)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("relpath", IGNORED_WEIGHT_PATHS)
def test_weight_artifacts_are_gitignored(relpath):
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, "expected git to ignore {}".format(relpath)


def test_gitignore_lists_weight_and_cache_dirs():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [pattern for pattern in WEIGHT_CACHE_IGNORES if pattern not in text]
    assert missing == [], "gitignore missing {}".format(missing)
