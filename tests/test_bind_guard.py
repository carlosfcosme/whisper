"""Bind guard: the server must listen on 127.0.0.1, never all-interfaces."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from whisper.bind import (
    ALL_INTERFACES,
    BindError,
    assert_bound_loopback,
    require_loopback_host,
)
from whisper.serve import create_server, serve

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
GITIGNORE = REPO_ROOT / ".gitignore"


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return


def test_bind_guard_fails_if_host_is_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_host(ALL_INTERFACES)
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=ALL_INTERFACES, port=0)


def test_bind_guard_fails_if_socket_is_all_interfaces():
    httpd = ThreadingHTTPServer((ALL_INTERFACES, 0), _QuietHandler)
    try:
        sock_host = httpd.socket.getsockname()[0]
        assert sock_host == ALL_INTERFACES
        with pytest.raises(BindError, match="all-interfaces"):
            assert_bound_loopback(httpd)
    finally:
        httpd.server_close()


def test_live_server_never_binds_all_interfaces():
    httpd = serve(port=0)
    try:
        sock_host = httpd.socket.getsockname()[0]
        bound = httpd.server_address[0]
        assert sock_host == "127.0.0.1"
        assert bound == "127.0.0.1"
        assert sock_host != ALL_INTERFACES
        assert bound != ALL_INTERFACES
        assert_bound_loopback(httpd)
    finally:
        httpd.server_close()


def test_offline_ci_and_weights_gitignore_still_hold():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "assert_no_weight_download.py" in workflow
    assert "check_no_hub.py" in workflow
    for pattern in (".cache/", "cache/", "weights/", "*.pt", "*.pth"):
        assert pattern in gitignore
