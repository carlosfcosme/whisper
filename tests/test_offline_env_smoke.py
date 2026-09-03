"""Offline environment smoke tests.

Require 127.0.0.1, prohibit model fetch, and verify weight artifacts
are gitignored. These tests must not download checkpoints.
"""

import json
import os
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.serve import BindError, create_server
from whisper.sovereign import BIND_HOST, DEFAULT_DEVICE

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_ARTIFACTS = (
    "tiny.pt",
    "tiny.pth",
    "model.safetensors",
    "model.onnx",
    "model.bin",
    ".cache/whisper/tiny.pt",
    "cache/whisper/base.pt",
)

pytestmark = pytest.mark.offline_smoke


def test_smoke_bind_host_is_127():
    assert BIND_HOST == "127.0.0.1"
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_smoke_serve_requires_loopback():
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=".".join(("0", "0", "0", "0")), port=0)

    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.handle_request, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["hub"] is False
        assert body["weights"] is False
    finally:
        httpd.server_close()
        thread.join(timeout=5)


def test_smoke_prohibits_model_fetch(tmp_path, monkeypatch):
    assert DEFAULT_DEVICE == "cpu"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    with pytest.raises(RuntimeError, match="offline|no weight pulls|forbidden|no Hub"):
        whisper.load_model("tiny")
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_smoke_weight_artifacts_are_gitignored():
    for relpath in WEIGHT_ARTIFACTS:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relpath],
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == 0, "expected ignore for {}".format(relpath)

    listed = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=str(REPO_ROOT)
    ).split(b"\0")
    suffixes = {".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx", ".gguf"}
    tracked_weights = []
    for raw in listed:
        if not raw:
            continue
        path = raw.decode("utf-8", "surrogateescape")
        if Path(path).suffix.lower() in suffixes:
            tracked_weights.append(path)
    assert tracked_weights == []


def test_smoke_checker_rejects_committed_weights():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        ["python3", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout
