"""Tests must not download model checkpoints."""

import os
import urllib.request

import pytest

import whisper


def test_whisper_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"


def test_named_model_does_not_pull_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model("tiny")
    cached = list(tmp_path.rglob("*.pt"))
    assert cached == [], f"checkpoint was written: {cached}"


def test_urlopen_blocks_official_weight_cdn():
    with pytest.raises(RuntimeError, match="must not pull model weights"):
        urllib.request.urlopen(whisper._MODELS["tiny"])
