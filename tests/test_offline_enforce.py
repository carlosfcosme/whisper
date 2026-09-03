"""Executable offline guards: no weight download, no non-loopback bind.

No WAN. No secrets. Bind policy loads without torch.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
GITIGNORE_SCRIPT = REPO / "scripts" / "check_gitignore_caches.py"

MUST_IGNORE = (
    "weights/model.pt",
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    ".huggingface/hub/models--x",
    "orphan.pt",
    "orphan.safetensors",
    "orphan.bin",
)


def _load_bind():
    path = REPO / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = _load_bind()


def test_gitignore_script_passes():
    result = subprocess.run(
        [sys.executable, str(GITIGNORE_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "gitignore-caches: ok" in result.stdout


@pytest.mark.parametrize("relpath", MUST_IGNORE)
def test_git_check_ignore_covers_weights_and_caches(relpath):
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, "{} must be gitignored".format(relpath)


def test_require_bind_refuses_non_loopback():
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1("")
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(ALL_INTERFACES)
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1("8.8.8.8")
    assert bind.require_bind_127_0_0_1(None) == LOOPBACK
    assert bind.require_bind_127_0_0_1(LOOPBACK) == LOOPBACK


def test_socket_bind_refuses_wildcard():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((ALL_INTERFACES, 0))
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind(("", 0))
    finally:
        sock.close()


def test_socket_bind_allows_loopback_only():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOOPBACK, 0))
        assert sock.getsockname()[0] == LOOPBACK
    finally:
        sock.close()


def test_wan_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_named_model_does_not_download_weights():
    whisper = pytest.importorskip("whisper")
    with pytest.raises(whisper.WeightDownloadError):
        whisper.load_model("tiny")
