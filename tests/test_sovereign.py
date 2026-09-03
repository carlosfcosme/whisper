"""Fail if bind is not 127.0.0.1, if Hub is contacted, or if weights are pulled."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    WeightDownloadError,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
    refuse_weight_auto_download,
)
from whisper.serve import create_server
from whisper.serve import main as serve_main

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def test_default_device_is_cpu_even_if_cuda_claims_available(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert torch.device(default_device()).type == "cpu"


def test_fails_if_bind_is_not_127_0_0_1():
    assert DEFAULT_BIND_HOST == "127.0.0.1"
    assert default_bind_host() == "127.0.0.1"
    assert whisper.default_bind_host() == "127.0.0.1"
    with pytest.raises(BindError):
        refuse_non_localhost_bind("0.0.0.0")
    with pytest.raises(BindError):
        refuse_non_localhost_bind("")
    with pytest.raises(BindError):
        create_server(host="0.0.0.0", port=0)
    assert serve_main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_start_script_binds_127_0_0_1_only():
    start = REPO_ROOT / ".cursor" / "start.sh"
    text = start.read_text()
    assert "127.0.0.1" in text
    assert "0.0.0.0" not in text


def test_live_health_bind_is_127_0_0_1():
    httpd = create_server(host="127.0.0.1", port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_fails_if_hub_is_contacted(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("Hub was contacted")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(HF_URL)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_URL, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


def test_fails_if_weights_are_pulled(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("weight URL was opened")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightDownloadError):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), in_memory=False)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    cache = tmp_path / "xdg"
    cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny")
    assert list(cache.rglob("*")) == []


def test_wildcard_socket_bind_fails_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_module_serve_refuses_all_interfaces():
    result = subprocess.run(
        [sys.executable, "-m", "whisper.serve", "--host", "0.0.0.0", "--port", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
