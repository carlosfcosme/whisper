import os
import urllib.request

import pytest

import whisper


def test_offline_env_forbids_weight_downloads():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert whisper.weights_download_forbidden() is True


def test_load_model_does_not_pull_named_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = tmp_path / "whisper"
    with pytest.raises(RuntimeError, match="offline|Refusing"):
        whisper.load_model("tiny")
    assert not cache.exists()
    assert list(tmp_path.rglob("*.pt")) == []


def test_official_checkpoint_url_is_blocked():
    url = whisper._MODELS["tiny"]
    with pytest.raises(RuntimeError, match="weight pulls|Hub"):
        urllib.request.urlopen(url)
