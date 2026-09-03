"""Allow loopback sockets only. Block WAN and Hub connects."""

import ipaddress

ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


class NetworkBlocked(RuntimeError):
    """Raised when a test or offline run tries to leave loopback."""


def _as_text(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value


def connect_host(address):
    """Return the host string from a socket address, or None."""
    address = _as_text(address)
    if isinstance(address, str):
        return address
    if isinstance(address, tuple) and address:
        return _as_text(address[0])
    return None


def is_loopback_connect(address):
    """True for 127.0.0.1 / localhost / AF_UNIX paths."""
    host = connect_host(address)
    if host is None:
        return False
    if isinstance(host, str) and (host.startswith("/") or host.startswith("\0")):
        return True
    host = (host or "").strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    host = host.split("%", 1)[0]
    if host.lower() in ALLOWED_LOOPBACK_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_loopback


def refuse_non_loopback(address):
    if is_loopback_connect(address):
        return
    raise NetworkBlocked(
        "blocked network: only 127.0.0.1 is allowed; got {!r}".format(address)
    )
