"""Tests must not pull Hub or Azure checkpoints."""

import urllib.request

import pytest

import whisper
from whisper.runtime import WeightDownloadError, weight_auto_download_allowed


def test_weight_auto_download_disabled_in_tests():
    assert weight_auto_download_allowed() is False


def test_load_model_does_not_download_weights(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("tests must not pull weights")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_transcribe_is_skipped_without_local_weights():
    assert weight_auto_download_allowed() is False
