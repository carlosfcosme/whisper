"""CPU/offline tests: block weight download, require loopback, check gitignore."""

import os
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.runtime import (
    BindError,
    WeightDownloadError,
    bind_localhost,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
    refuse_weight_auto_download,
)

ROOT = Path(__file__).resolve().parents[1]
HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
AZURE_TINY = whisper._MODELS["tiny"]
GITIGNORE_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "checkpoints/",
    "*.pt",
    "*.pth",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "checkpoints/tiny.pt",
    "tiny.pt",
    "model.pth",
)


def test_cpu_only_default_ignores_cuda(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert torch.device(default_device()).type == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_weight_download_is_blocked(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("unit tests must not download weights")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        refuse_weight_auto_download(HF_HUB_URL)
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)
    with pytest.raises(WeightDownloadError):
        whisper._download(AZURE_TINY, str(tmp_path), in_memory=False)
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "tiny.pt").exists()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    assert not cache.exists()


def test_remote_urlopen_is_blocked_in_offline_tests():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(HF_HUB_URL, timeout=1)
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(AZURE_TINY, timeout=1)


def test_loopback_bind_is_required():
    assert default_bind_host() == "127.0.0.1"
    for host in ("0.0.0.0", "", "::", "8.8.8.8"):
        with pytest.raises(BindError, match="127.0.0.1"):
            refuse_non_localhost_bind(host)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        host, port = bind_localhost(server, 0)
        assert host == "127.0.0.1"
        assert port > 0
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            conn, addr = server.accept()
            try:
                assert addr[0] == "127.0.0.1"
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        server.close()


def test_cache_weight_gitignore():
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip()
    }
    missing = [pat for pat in GITIGNORE_PATTERNS if pat not in lines]
    assert missing == []

    for path in IGNORE_EXAMPLES:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=ROOT,
        )
        assert proc.returncode == 0, path

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            ".cache",
            ".cache/**",
            "cache",
            "cache/**",
            "weights",
            "weights/**",
            "checkpoints",
            "checkpoints/**",
            "*.pt",
            "*.pth",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""
