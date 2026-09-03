import importlib.util
import socket
import sys
from ipaddress import IPv4Address
from pathlib import Path

import pytest

_BIND_PATH = Path(__file__).resolve().parents[1] / "whisper" / "bind.py"


def _load_bind():
    existing = sys.modules.get("whisper_bind_guard")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", _BIND_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["whisper_bind_guard"] = module
    spec.loader.exec_module(module)
    return module


bind = _load_bind()

_REFUSED_HOSTS = [
    "0.0.0.0",
    "::",
    "::1",
    "",
    "*",
    "localhost",
    "8.8.8.8",
    "192.168.1.1",
    "10.0.0.1",
    "127.0.0.2",
    IPv4Address("0.0.0.0"),
    b"0.0.0.0",
    None,
]


@pytest.mark.parametrize("host", _REFUSED_HOSTS)
def test_require_loopback_host_refuses_non_loopback(host):
    with pytest.raises(bind.NonLoopbackBindError):
        bind.require_loopback_host(host)
    assert not bind.is_allowed_bind_host(host)


def test_require_loopback_host_allows_127():
    assert bind.require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert bind.require_loopback_host(b"127.0.0.1") == "127.0.0.1"
    assert bind.require_loopback_host(IPv4Address("127.0.0.1")) == "127.0.0.1"
    assert bind.require_loopback_host(" 127.0.0.1 ") == "127.0.0.1"
    assert bind.is_allowed_bind_host("127.0.0.1")


def test_bind_loopback_refuses_wildcard_without_listening():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bind.NonLoopbackBindError):
            bind.bind_loopback(sock, 0, host="0.0.0.0")
        bind.assert_only_loopback_listeners()
    finally:
        sock.close()


def test_listen_loopback_getsockname_is_127():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bind.listen_loopback(sock, 0)
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
        bind.assert_only_loopback_listeners()
    finally:
        sock.close()
    bind.assert_only_loopback_listeners()


def test_bind_loopback_rejects_bad_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bind.NonLoopbackBindError):
            bind.bind_loopback(sock, -1)
        with pytest.raises(bind.NonLoopbackBindError):
            bind.bind_loopback(sock, 65536)
        with pytest.raises(bind.NonLoopbackBindError):
            bind.bind_loopback(sock, True)  # bool is a subclass of int
    finally:
        sock.close()


def test_parse_proc_net_tcp_loopback_and_wildcard():
    table = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 11111 1 0000000000000000 100 0 0 10 0
   1: 00000000:0050 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 22222 1 0000000000000000 100 0 0 10 0
"""
    rows = bind.parse_proc_net_tcp(table, ipv6=False)
    hosts = {(host, port, inode, state) for host, port, inode, state in rows}
    assert ("127.0.0.1", 8080, 11111, 0x0A) in hosts
    assert ("0.0.0.0", 80, 22222, 0x0A) in hosts


def test_assert_only_loopback_listeners_fails_on_wildcard():
    with pytest.raises(bind.NonLoopbackListenError, match="0.0.0.0:8080"):
        bind.assert_only_loopback_listeners(endpoints=[("0.0.0.0", 8080)])


def test_assert_only_loopback_listeners_fails_on_lan():
    with pytest.raises(bind.NonLoopbackListenError, match="192.168.0.5"):
        bind.assert_only_loopback_listeners(endpoints=[("192.168.0.5", 9)])


def test_assert_only_loopback_listeners_allows_127():
    bind.assert_only_loopback_listeners(endpoints=[("127.0.0.1", 9)])


def test_assert_only_loopback_listeners_uses_iter(monkeypatch):
    monkeypatch.setattr(
        bind, "iter_process_listening_endpoints", lambda: [("0.0.0.0", 1)]
    )
    with pytest.raises(bind.NonLoopbackListenError):
        bind.assert_only_loopback_listeners()


def test_package_source_has_no_wildcard_literals():
    root = Path(__file__).resolve().parents[1] / "whisper"
    assert bind.find_wildcard_host_literals(root) == []
    assert bind.find_non_loopback_bind_calls(root) == []


def test_find_wildcard_host_literals_detects_offender(tmp_path):
    (tmp_path / "bind.py").write_text('DENIED = "0.0.0.0"\n', encoding="utf-8")
    offender = tmp_path / "server.py"
    offender.write_text('HOST = "0.0.0.0"\n', encoding="utf-8")
    assert bind.find_wildcard_host_literals(tmp_path) == [str(offender)]


def test_find_non_loopback_bind_calls_detects_wildcard(tmp_path):
    src = tmp_path / "server.py"
    src.write_text('sock.bind(("0.0.0.0", 80))\n', encoding="utf-8")
    found = bind.find_non_loopback_bind_calls(tmp_path)
    assert found and "0.0.0.0" in found[0]


def test_find_non_loopback_bind_calls_positional_host(tmp_path):
    src = tmp_path / "helper.py"
    src.write_text('listen_loopback(sock, 0, "0.0.0.0")\n', encoding="utf-8")
    found = bind.find_non_loopback_bind_calls(tmp_path)
    assert found and "0.0.0.0" in found[0]
