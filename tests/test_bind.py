import importlib.util
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_BIND = os.path.join(ROOT, "scripts", "check_bind_localhost.py")
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def _load_bind():
    path = os.path.join(ROOT, "whisper", "bind.py")
    spec = importlib.util.spec_from_file_location("whisper_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = _load_bind()
LOOPBACK_HOST = bind.LOOPBACK_HOST
BindError = bind.BindError


def test_require_bind_defaults_to_loopback():
    assert bind.require_bind_127_0_0_1() == LOOPBACK_HOST
    assert bind.require_bind_127_0_0_1(None) == LOOPBACK_HOST
    assert bind.require_bind_127_0_0_1("127.0.0.1") == LOOPBACK_HOST
    assert bind.require_bind_127_0_0_1("localhost") == LOOPBACK_HOST
    assert bind.require_bind_127_0_0_1("LOCALHOST.") == LOOPBACK_HOST


@pytest.mark.parametrize(
    "host",
    [
        "",
        "   ",
        ALL_INTERFACES,
        "::",
        "*",
        "[::]",
        "192.168.1.10",
        "example.com",
        "0",
    ],
)
def test_require_bind_refuses_non_loopback(host):
    with pytest.raises(BindError):
        bind.require_bind_127_0_0_1(host)


def test_empty_host_message_names_loopback():
    with pytest.raises(BindError, match="bind host is required"):
        bind.require_bind_127_0_0_1("")


def test_is_loopback_host():
    assert bind.is_loopback_host("127.0.0.1")
    assert bind.is_loopback_host("localhost")
    assert not bind.is_loopback_host(ALL_INTERFACES)
    assert not bind.is_loopback_host("")


class _Silent(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return


def test_create_loopback_httpd_binds_127_0_0_1():
    httpd = bind.create_loopback_httpd(_Silent, host=LOOPBACK_HOST, port=0)
    try:
        assert httpd.server_address[0] == LOOPBACK_HOST
        assert httpd.server_address[1] > 0
    finally:
        httpd.server_close()


def test_create_loopback_httpd_refuses_all_interfaces():
    with pytest.raises(BindError):
        bind.create_loopback_httpd(_Silent, host=ALL_INTERFACES, port=0)


def test_check_bind_localhost_script_passes():
    result = subprocess.run(
        [sys.executable, CHECK_BIND],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout
