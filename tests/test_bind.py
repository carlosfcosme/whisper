"""Loopback-only bind: 127.0.0.1 accepted, all-interface and WAN refused."""

from __future__ import annotations

import socket

import pytest

from tests.import_stdlib import load_bind_standalone

bind = load_bind_standalone()
LOOPBACK = bind.LOOPBACK_HOST
UNSPECIFIED = ".".join(("0",) * 4)


def test_require_loopback_defaults_to_127():
    assert bind.require_loopback_host(None) == LOOPBACK
    assert bind.require_loopback_host(LOOPBACK) == LOOPBACK
    assert bind.require_loopback_host("localhost") == LOOPBACK


@pytest.mark.probes_bind
@pytest.mark.parametrize(
    "host",
    [UNSPECIFIED, "::", "*", "", "8.8.8.8", "192.168.1.10", "::1", "127.0.0.2"],
)
def test_require_loopback_refuses_non_loopback(host):
    with pytest.raises(bind.BindError):
        bind.require_loopback_host(host)


def test_bind_127_succeeds():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOOPBACK, 0))
        host, port = sock.getsockname()[:2]
        assert host == LOOPBACK
        assert port > 0
        bind.assert_no_nonloopback_listeners()
    finally:
        sock.close()


@pytest.mark.probes_bind
def test_bind_all_interfaces_fails():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bind.BindError):
            sock.bind((UNSPECIFIED, 0))
    finally:
        sock.close()


@pytest.mark.probes_bind
def test_bind_empty_host_fails():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bind.BindError):
            sock.bind(("", 0))
    finally:
        sock.close()


def test_is_loopback_host_helper():
    assert bind.is_loopback_host(LOOPBACK)
    assert bind.is_loopback_host("localhost")
    assert not bind.is_loopback_host(UNSPECIFIED)
    assert not bind.is_loopback_host("huggingface.co")
