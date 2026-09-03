"""Localhost-only download guard for the precache/verify path.

The Cloud Agent verify path is easy to point at the official CDN (or any
other remote host) on a cache miss. When ``WHISPER_LOCALHOST_ONLY`` is
enabled, network pulls are allowed only to loopback hosts. Remote and WAN
hosts — including ``openaipublic.azureedge.net`` — are refused, and
redirects to those hosts are refused as well.

A cache hit is not a pull and is not affected by this guard.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from ipaddress import ip_address
from typing import Optional
from urllib.parse import urlparse

LOCALHOST_ONLY_ENV = "WHISPER_LOCALHOST_ONLY"
_TRUTHY = {"1", "true", "yes", "on"}


class RemotePullError(RuntimeError):
    """Raised when a remote/WAN pull is refused in localhost-only mode."""


def localhost_only_enabled() -> bool:
    """Return True when the verify path must refuse remote/WAN pulls."""
    return os.environ.get(LOCALHOST_ONLY_ENV, "").strip().lower() in _TRUTHY


def hostname_is_localhost(host: Optional[str]) -> bool:
    """True for ``localhost`` (any case) and loopback IPs (``127.0.0.0/8``, ``::1``).

    Hostnames are not resolved, so a name that happens to point at loopback
    is still treated as remote. That avoids DNS rebinding and "wrong host"
    mistakes.
    """
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def url_is_localhost(url: str) -> bool:
    """True for ``file:`` URLs and ``http(s):`` URLs whose host is loopback."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in ("http", "https"):
        return False
    return hostname_is_localhost(parsed.hostname)


def refuse_remote_pull(url: str) -> None:
    """No-op unless localhost-only mode is on; then reject non-loopback URLs."""
    if not localhost_only_enabled():
        return
    if url_is_localhost(url):
        return
    host = urlparse(url).hostname or "<unknown>"
    raise RemotePullError(
        f"Refusing remote/WAN pull from host {host!r}. "
        f"The whisper precache/verify path is localhost-only. "
        f"Serve weights from http://127.0.0.1/... or unset {LOCALHOST_ONLY_ENV}."
    )


class LocalhostOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that would pull from a remote/WAN host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        refuse_remote_pull(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen_maybe_localhost_only(url: str):
    """``urlopen`` that refuses remote/WAN targets when localhost-only is on."""
    if not localhost_only_enabled():
        return urllib.request.urlopen(url)
    refuse_remote_pull(url)
    opener = urllib.request.build_opener(LocalhostOnlyRedirectHandler())
    try:
        return opener.open(url)
    except RemotePullError:
        raise
    except urllib.error.URLError as exc:
        raise RemotePullError(f"Localhost-only pull failed for {url!r}: {exc}") from exc
