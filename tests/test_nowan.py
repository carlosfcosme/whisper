"""Executable no-WAN fixtures: no weight pull, bind 127.0.0.1 only."""

import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.bind import BIND_HOST, BindError, bind_tcp, require_bind_host
from whisper.fixtures import (
    WeightDownloadError,
    refuse_weight_pull,
    run_executable_checks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_autouse_fixture_blocks_wan_urlopen():
    with pytest.raises(WeightDownloadError, match="WHISPER_OFFLINE"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_autouse_fixture_blocks_named_weight_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []


def test_offline_download_helper_refuses_cdn():
    with pytest.raises(WeightDownloadError):
        refuse_weight_pull(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_loopback_socket_fixture_binds_127(loopback_socket):
    host, port = loopback_socket.getsockname()
    assert host == BIND_HOST
    assert port > 0
    loopback_socket.listen(1)
    client = socket.create_connection((host, port), timeout=1)
    try:
        accepted, addr = loopback_socket.accept()
        try:
            assert addr[0] == BIND_HOST
        finally:
            accepted.close()
    finally:
        client.close()


def test_autouse_fixture_refuses_wildcard_bind():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(BindError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_bind_tcp_and_require_host():
    assert require_bind_host() == BIND_HOST
    with pytest.raises(BindError):
        require_bind_host("0.0.0.0")
    sock = bind_tcp(0)
    try:
        assert sock.getsockname()[0] == BIND_HOST
    finally:
        sock.close()


def test_run_executable_checks_passes():
    assert run_executable_checks() == 0


def test_python_m_whisper_fixtures():
    env = os.environ.copy()
    env["WHISPER_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "whisper.fixtures"],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK: executable no-WAN fixtures" in result.stdout
