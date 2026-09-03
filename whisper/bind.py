"""Local bind address. Helpers must listen on 127.0.0.1 only."""

from typing import Optional

LOCALHOST = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})


def bind_host(host: Optional[str] = None) -> str:
    """Return the host to bind.

    Defaults to ``127.0.0.1``. Wildcard addresses are rejected.
    """
    resolved = LOCALHOST if host is None else str(host).strip()
    if resolved in WILDCARD_HOSTS:
        raise ValueError("bind {0} only; refused {1}".format(LOCALHOST, resolved))
    return resolved
