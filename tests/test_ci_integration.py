"""Integration: committed weights fail CI; default device is CPU; bind 127.0.0.1."""

import importlib.util
import subprocess
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import torch

import whisper
from whisper.model import ModelDimensions, Whisper

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ci_integration_weights_cpu_loopback(tmp_path):
    """One path covering the three CI integration rules."""
    integration = _load_script("ci_integration")
    weights = _load_script("check_no_weights")
    demo = _load_script("demo_server")

    # 1) Committed weights / large binaries fail the gate.
    assert weights.classify("models/tiny.pt", 100) is not None
    assert weights.find_violations(REPO_ROOT) == []

    # 2) Default device is CPU, including load_model with no device arg.
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
    toy = Whisper(dims)
    ckpt = tmp_path / "integration.pt"
    torch.save({"dims": dims.__dict__, "model_state_dict": toy.state_dict()}, ckpt)
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"

    # 3) Demo server binds 127.0.0.1 only and serves on loopback.
    assert demo.DEFAULT_HOST == "127.0.0.1"
    bad_host = "0.0.0." + "0"
    try:
        demo.validate_host(bad_host)
        raised = False
    except ValueError:
        raised = True
    assert raised
    server = demo.make_server(demo.DEFAULT_HOST, 0)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read()
        conn.close()
        assert response.status == 200
        assert b"127.0.0.1" in body
    finally:
        server.server_close()
        thread.join(timeout=5)

    # Umbrella script used by GitHub Actions agrees.
    assert integration.collect_errors(REPO_ROOT) == []
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ci_integration.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "no committed weights" in result.stdout
    assert "cpu" in result.stdout
    assert "127.0.0.1" in result.stdout
