"""Ticket 4: CPU-only / no-store offline defaults. CI needs no GPU or weights."""

import json
import os
import pathlib
import tempfile
import threading
import urllib.request

import pytest
import torch

import whisper
from whisper.runtime import (
    CACHE_CONTROL_NO_STORE,
    CPU_ONLY_ENV,
    NO_STORE_CACHE_DIRNAME,
    BindError,
    WeightDownloadError,
    cache_control_no_store,
    default_device,
    default_download_root,
    home_cache_root,
    no_store_enabled,
    offline_enabled,
    weight_auto_download_allowed,
)
from whisper.serve import make_server

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def test_default_device_is_cpu_even_if_cuda_claims_available(monkeypatch):
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert torch.device(default_device()).type == "cpu"


def test_cpu_only_env_wins_over_cuda_device(monkeypatch):
    monkeypatch.setenv(CPU_ONLY_ENV, "1")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"


def test_no_store_cache_default_is_not_home_cache(monkeypatch):
    monkeypatch.setenv("WHISPER_NO_STORE", "1")
    monkeypatch.delenv("WHISPER_CACHE_DIR", raising=False)
    root = default_download_root()
    assert pathlib.Path(root).name == NO_STORE_CACHE_DIRNAME
    assert os.path.realpath(root) != os.path.realpath(home_cache_root())
    assert pathlib.Path(root).parent == pathlib.Path(tempfile.gettempdir())


def test_no_store_load_model_does_not_write_home_cache(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("WHISPER_NO_STORE", "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.delenv("WHISPER_CACHE_DIR", raising=False)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny")

    home_cache = home / ".cache" / "whisper"
    assert not home_cache.exists() or list(home_cache.iterdir()) == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_offline_refuses_hub_and_named_checkpoint(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_ci_does_not_need_gpu_or_weight_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv(CPU_ONLY_ENV, "1")
    monkeypatch.setenv("WHISPER_NO_STORE", "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert default_device() == "cpu"
    assert no_store_enabled()
    assert offline_enabled()
    assert weight_auto_download_allowed() is False

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_ci_workflow_is_cpu_offline_no_weights():
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "WHISPER_CPU_ONLY" in text
    assert "WHISPER_NO_STORE" in text
    assert "WHISPER_OFFLINE" in text
    assert "not test_transcribe" in text
    assert "+cpu" in text
    assert "requires_cuda" in text
    assert "--live" not in text
    assert "0.0.0.0" not in text
    assert "Field-Brain" not in text


def test_no_live_flag_and_no_wildcard_in_cli_source():
    transcribe = (ROOT / "whisper" / "transcribe.py").read_text()
    serve = (ROOT / "whisper" / "serve.py").read_text()
    assert "--live" not in transcribe
    assert "--live" not in serve
    assert "0.0.0.0" not in serve
    assert "Field-Brain" not in transcribe
    assert "Field-Brain" not in serve
    assert "API_KEY" not in transcribe
    assert "API_KEY" not in serve


def test_make_server_refuses_wildcard():
    with pytest.raises(BindError, match="0.0.0.0"):
        make_server(host="0.0.0.0", port=0)


def test_health_sends_cache_control_no_store():
    assert cache_control_no_store() == CACHE_CONTROL_NO_STORE == "no-store"
    httpd = make_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as response:
            headers = dict(response.headers)
            payload = json.loads(response.read().decode("utf-8"))
        cache_control = headers.get("Cache-Control") or headers.get("cache-control")
        assert cache_control == "no-store"
        assert payload["device"] == "cpu"
        assert payload["bind"] == "127.0.0.1"
        assert payload["cache_control"] == "no-store"
    finally:
        httpd.shutdown()
        httpd.server_close()
