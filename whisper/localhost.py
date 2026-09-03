"""Localhost-only bind policy. Serve paths must use 127.0.0.1."""

BIND_HOST = "127.0.0.1"
ALLOWED_BIND_HOSTS = frozenset({"127.0.0.1", "::1"})


def serve_bind_host(host=None) -> str:
    """Return a loopback bind address, or raise if the host is not localhost."""
    if host in (None, "", "localhost"):
        return BIND_HOST
    if host not in ALLOWED_BIND_HOSTS:
        raise ValueError(
            f"serve must bind to 127.0.0.1 (got {host!r}); "
            "refusing non-localhost hosts"
        )
    return host
