"""Refuse any server bind that is not 127.0.0.1."""

from typing import Iterable

LOOPBACK_HOST = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "::0", "", "*"})


def is_wildcard_host(host: str) -> bool:
    if host is None:
        return True
    cleaned = str(host).strip().strip("[]")
    return cleaned in WILDCARD_HOSTS


def is_loopback_host(host: str) -> bool:
    if host is None:
        return False
    cleaned = str(host).strip().strip("[]")
    return cleaned == LOOPBACK_HOST


def assert_loopback_bind(host: str) -> str:
    """Return host if it is 127.0.0.1; raise if it would bind all interfaces."""
    if is_wildcard_host(host):
        raise ValueError(
            "server must bind to {0}, refusing wildcard {1!r}".format(
                LOOPBACK_HOST, host
            )
        )
    if not is_loopback_host(host):
        raise ValueError(
            "server must bind to {0}, got {1!r}".format(LOOPBACK_HOST, host)
        )
    return LOOPBACK_HOST


def assert_bound_socket_is_loopback(bound_host: str) -> str:
    """Fail if an already-bound socket is listening on 0.0.0.0 or another host."""
    if is_wildcard_host(bound_host) or not is_loopback_host(bound_host):
        raise OSError(
            "refusing non-loopback bind: {0!r} (required {1})".format(
                bound_host, LOOPBACK_HOST
            )
        )
    return bound_host


def reject_wildcard_hosts(hosts: Iterable[str]) -> None:
    for host in hosts:
        if is_wildcard_host(host) or host == "0.0.0.0":
            raise ValueError("wildcard bind is not allowed: {0!r}".format(host))
