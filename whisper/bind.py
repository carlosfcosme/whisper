"""Serve/bind guard: listeners must use 127.0.0.1.

Stdlib only so CI can load this file without installing torch.
"""

BIND_HOST = "127.0.0.1"
WILDCARD_HOST = "0.0.0.0"


class BindError(ValueError):
    """Raised when a bind host is not 127.0.0.1."""


def require_bind_host(host):
    """Return ``127.0.0.1`` or raise ``BindError``.

    ``0.0.0.0``, ``::``, LAN, WAN, and ``localhost`` are refused.
    """
    normalized = "" if host is None else str(host).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        "bind host must be {ok}, got {got!r}. "
        "Refusing {wild} and any non-loopback address.".format(
            ok=BIND_HOST, got=host, wild=WILDCARD_HOST
        )
    )
