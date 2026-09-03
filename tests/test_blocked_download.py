"""Model fetches are blocked in tests (Hub and official Azure)."""

from __future__ import annotations

import os

import pytest

from whisper import _download
from whisper.hub import DownloadBlockedError, HubDisabledError, assert_can_fetch

AZURE = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
)


def test_whisper_no_download_env_is_set():
    assert os.environ.get("WHISPER_NO_DOWNLOAD") == "1"


def test_official_checkpoint_fetch_is_blocked(tmp_path):
    with pytest.raises(DownloadBlockedError, match="WHISPER_NO_DOWNLOAD"):
        assert_can_fetch(AZURE)
    with pytest.raises(DownloadBlockedError):
        _download(AZURE, str(tmp_path), False)
    assert not any(tmp_path.rglob("*.pt"))


def test_hub_fetch_is_blocked_even_when_downloads_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_NO_DOWNLOAD", "0")
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
    with pytest.raises(HubDisabledError):
        assert_can_fetch(url)
    with pytest.raises(HubDisabledError):
        _download(url, str(tmp_path), False)
    assert list(tmp_path.iterdir()) == []
