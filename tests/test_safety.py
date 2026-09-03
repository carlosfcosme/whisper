import importlib.util
import os
import socket
import subprocess
import sys
import threading
import urllib.request
from http.client import HTTPConnection
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.serve import DEFAULT_HOST, BindError, create_server, normalize_bind_host

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    ckpt = tmp_path / "toy.pt"
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, ckpt)
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_hub_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert whisper.weights_download_forbidden() is True


def test_default_device_stays_cpu_when_cuda_reports_available(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    ckpt = tmp_path / "toy.pt"
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, ckpt)
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_tests_are_cpu_only():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.cuda.is_available() is False


def test_remote_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="WAN is forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="WAN is forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_wan_socket_connect_is_refused():
    with pytest.raises(RuntimeError, match="WAN is forbidden"):
        socket.create_connection(("1.1.1.1", 80), timeout=1)


def test_download_refuses_when_offline(tmp_path):
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)


def test_check_no_weights_classifies_extensions():
    check = _load_script("check_no_weights")
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("weights/model.safetensors", 10) is not None
    assert check.classify("export/model.onnx", 10) is not None
    assert check.classify("libfoo.so", 100) is not None
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("README.md", 800) is None
    assert check.classify("README.md", check.MAX_FILE_BYTES + 1) is not None


def test_check_no_weights_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_check_no_weights_flags_planted_checkpoint(tmp_path):
    check = _load_script("check_no_weights")
    planted = tmp_path / "tiny.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    hits = check.find_violations(tmp_path, ["tiny.pt"])
    assert hits and hits[0][0] == "tiny.pt"


def test_serve_refuses_non_loopback():
    with pytest.raises(BindError, match="127.0.0.1"):
        normalize_bind_host("0.0.0.0")
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server("0.0.0.0", 8765)


def test_serve_binds_loopback_and_serves():
    server = create_server(DEFAULT_HOST, 0)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        body = response.read()
        conn.close()
        assert response.status == 200
        assert b"127.0.0.1" in body
        assert b'"weights": false' in body
    finally:
        server.server_close()
        thread.join(timeout=5)


def test_cli_device_default_is_cpu():
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "default: cpu" in result.stdout


def test_cli_serve_refuses_all_interfaces():
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "serve", "--host", "0.0.0.0"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "127.0.0.1" in result.stderr


def test_start_script_binds_loopback_only():
    start = (REPO_ROOT / ".cursor" / "start.sh").read_text()
    assert "127.0.0.1" in start
    assert "0.0.0.0" not in start
