"""Offline no-weight-download guard. Fail if download helpers run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_offline():
    path = ROOT / "whisper" / "offline.py"
    spec = importlib.util.spec_from_file_location("whisper_offline_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


offline = _load_offline()


def test_auto_download_disallowed_by_default(monkeypatch):
    monkeypatch.delenv(offline.ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_NO_WEIGHT_DOWNLOAD", raising=False)
    assert offline.weight_auto_download_allowed() is False


def test_hub_url_refused():
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(offline.WeightDownloadError, match="Hugging Face Hub"):
        offline.refuse_weight_auto_download(url)


def test_azure_cache_miss_refused_by_default(monkeypatch):
    monkeypatch.delenv(offline.ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)
    url = "https://openaipublic.azureedge.net/main/whisper/models/deadbeef/tiny.pt"
    with pytest.raises(offline.WeightDownloadError, match="disabled by default"):
        offline.refuse_weight_auto_download(url)


def test_named_model_does_not_call_urlopen(monkeypatch, tmp_path):
    import urllib.request

    whisper = pytest.importorskip("whisper")
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urlopen must not run")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(whisper.offline.WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert calls == []
