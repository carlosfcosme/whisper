"""Remaining sovereignty tests: loopback bind, CPU default, CI no-Hub, gitignore."""

import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch

import whisper
from whisper.offline import weight_pull_allowed
from whisper.serve import (
    ALL_INTERFACES,
    LOOPBACK_BIND,
    BindError,
    require_loopback_bind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
GITIGNORE = REPO_ROOT / ".gitignore"
INSTALL = REPO_ROOT / ".cursor" / "install.sh"

CACHE_WEIGHT_IGNORE_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
    "*.bin",
    ".cache/",
    "weights/",
    "checkpoints/",
    ".huggingface/",
)

IGNORED_SAMPLE_PATHS = (
    "tiny.pt",
    "weights/encoder.bin",
    "checkpoints/model.safetensors",
    ".cache/whisper/tiny.pt",
    ".huggingface/hub/models--openai--whisper-tiny/pytorch_model.bin",
)


def _workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def test_bind_127_0_0_1_is_the_loopback_contract():
    assert LOOPBACK_BIND == "127.0.0.1"
    assert require_loopback_bind() == "127.0.0.1"
    assert require_loopback_bind("127.0.0.1") == "127.0.0.1"
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_bind(ALL_INTERFACES)


def test_bind_127_0_0_1_socket_roundtrip():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        host, port = server.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            conn, addr = server.accept()
            try:
                assert addr[0] == "127.0.0.1"
                client.sendall(b"ok")
                assert conn.recv(2) == b"ok"
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        server.close()


def test_wildcard_socket_bind_is_blocked_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((ALL_INTERFACES, 0))
    finally:
        sock.close()


def test_start_and_ci_do_not_bind_all_interfaces():
    start = (REPO_ROOT / ".cursor" / "start.sh").read_text(encoding="utf-8")
    assert "127.0.0.1" in start
    assert ALL_INTERFACES not in start
    assert ALL_INTERFACES not in _workflow_text()
    env = (REPO_ROOT / ".cursor" / "environment.json").read_text(encoding="utf-8")
    assert ALL_INTERFACES not in env


def test_cpu_only_default():
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.device(whisper.DEFAULT_DEVICE).type == "cpu"


def test_cpu_only_default_ignores_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_ci_and_install_are_cpu_wheels():
    workflow = _workflow_text()
    assert "torch==${{ matrix.pytorch-version }}+cpu" in workflow
    assert "download.pytorch.org/whl/cpu" in workflow
    install = INSTALL.read_text(encoding="utf-8")
    assert "torch==2.5.1+cpu" in install
    assert "download.pytorch.org/whl/cpu" in install


def test_ci_does_not_download_hub_or_weights():
    workflow = _workflow_text()
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'WHISPER_OFFLINE: "1"' in workflow
    assert "scripts/assert_no_weight_cache.py" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "huggingface.co" not in workflow
    assert "hf_hub_download" not in workflow
    assert "snapshot_download" not in workflow
    assert "from_pretrained" not in workflow
    assert weight_pull_allowed() is False

    install = INSTALL.read_text(encoding="utf-8")
    assert "huggingface.co" not in install
    assert "load_model" not in install
    assert "openaipublic.azureedge.net" not in install


def test_gitignore_covers_cache_and_weights():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in CACHE_WEIGHT_IGNORE_PATTERNS:
        assert pattern in text, "missing gitignore pattern: {}".format(pattern)


@pytest.mark.parametrize("relpath", IGNORED_SAMPLE_PATHS)
def test_git_ignores_cache_and_weight_paths(relpath):
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, "expected git to ignore {}".format(relpath)
