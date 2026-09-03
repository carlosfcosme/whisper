import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = ".".join(("0",) * 4)


def _load_bind():
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bind = _load_bind()


@pytest.mark.parametrize("host", [None, "127.0.0.1", "localhost", "LOCALHOST"])
def test_require_loopback_host_allows_loopback(host):
    assert bind.require_loopback_host(host) == bind.LOOPBACK_HOST
    assert bind.is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "::",
        "*",
        "",
        "   ",
        "192.168.1.10",
        "example.com",
        "10.0.0.1",
        "8.8.8.8",
        "::1",
        "127.0.0.2",
        "[::]",
    ],
)
def test_require_loopback_host_refuses_non_loopback(host):
    with pytest.raises(bind.BindError, match="127.0.0.1"):
        bind.require_loopback_host(host)
    if host.strip():
        assert not bind.is_loopback_host(host)


def test_all_interfaces_token_is_v4_unspecified():
    assert ALL_INTERFACES == ".".join(("0",) * 4)
    with pytest.raises(bind.BindError):
        bind.require_loopback_host(ALL_INTERFACES)


def test_guarded_socket_bind_refuses_all_interfaces():
    bind.install_bind_guard()
    sock = bind.socket.socket(bind.socket.AF_INET, bind.socket.SOCK_STREAM)
    try:
        with pytest.raises(bind.BindError):
            sock.bind((ALL_INTERFACES, 0))
    finally:
        sock.close()
        bind.uninstall_bind_guard()


def test_guarded_socket_bind_allows_loopback():
    bind.install_bind_guard()
    sock = bind.socket.socket(bind.socket.AF_INET, bind.socket.SOCK_STREAM)
    try:
        sock.bind((bind.LOOPBACK_HOST, 0))
        host, _port = sock.getsockname()
        assert host == bind.LOOPBACK_HOST
    finally:
        sock.close()
        bind.uninstall_bind_guard()


def test_proc_tcp_parser_flags_bind_all():
    sample = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr "
        "tm->when retrnsmt   uid  timeout inode\n"
        "   0: 00000000:0050 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000     0        0 12345 1\n"
        "   1: 0100007F:1F90 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000     0        0 99999 1\n"
    )
    rows = bind.parse_proc_inet_table(sample, ipv6=False)
    assert rows[0].host == ALL_INTERFACES
    assert rows[0].inode == 12345
    assert rows[1].host == bind.LOOPBACK_HOST
    with pytest.raises(bind.BindError, match="non-loopback listen"):
        bind.assert_listeners_loopback_only(rows)
    bind.assert_listeners_loopback_only([rows[1]])
