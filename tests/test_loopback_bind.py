"""Ticket 1: bind/listen only on 127.0.0.1. No weight download."""

import os
import subprocess
import threading
from pathlib import Path

import pytest

from whisper.defaults import DEFAULT_HOST, is_loopback_host, require_loopback_host
from whisper.local_server import create_server

ROOT = Path(__file__).resolve().parents[1]
BIND_SCRIPT = ROOT / ".github" / "scripts" / "fail_if_non_loopback_bind.sh"


def _run_bind_guard(root: Path):
    env = os.environ.copy()
    env["BIND_GUARD_ROOT"] = str(root)
    return subprocess.run(
        ["bash", str(BIND_SCRIPT)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_default_host_is_loopback():
    assert DEFAULT_HOST == "127.0.0.1"
    assert is_loopback_host("127.0.0.1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("::")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "localhost", "1.2.3.4"])
def test_non_loopback_host_is_rejected(host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host(host)
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(host, 0)


def test_listen_socket_is_127_0_0_1():
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.socket.getsockname()[:2]
        assert host == "127.0.0.1"
        assert port > 0
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_ci_grep_passes_on_loopback_default():
    result = _run_bind_guard(ROOT)
    assert result.returncode == 0, result.stderr


def test_ci_grep_fails_if_default_host_is_not_loopback(tmp_path):
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "defaults.py").write_text(
        'DEFAULT_HOST = "0.0.0.0"\n', encoding="utf-8"
    )
    result = _run_bind_guard(tmp_path)
    assert result.returncode == 1
    assert "127.0.0.1" in result.stderr


def test_ci_grep_fails_on_forbidden_0_0_0_0_bind(tmp_path):
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "defaults.py").write_text(
        'DEFAULT_HOST = "127.0.0.1"\n', encoding="utf-8"
    )
    (tmp_path / "whisper" / "serve.py").write_text(
        'HTTPServer(("0.0.0.0", 80), None)\n', encoding="utf-8"
    )
    result = _run_bind_guard(tmp_path)
    assert result.returncode == 1
    assert "0.0.0.0" in result.stderr


def test_ci_workflow_runs_loopback_bind_guard():
    yml = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    script = BIND_SCRIPT.read_text(encoding="utf-8")
    assert "fail_if_non_loopback_bind.sh" in yml
    assert "loopback-bind-guard" in yml
    assert "0.0.0.0" in script
