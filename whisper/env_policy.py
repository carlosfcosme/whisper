"""Environment policy: force 127.0.0.1 binds and no default weight fetch.

Cloud Agent / local-dev setup sources ``.cursor/env.sh``, which exports
``WHISPER_BIND_HOST=127.0.0.1`` and ``WHISPER_ALLOW_WEIGHT_FETCH=0``.
``.cursor/verify.sh`` (run by CI) asserts both.

When weight fetch is denied (the default: unset or ``0``),
``whisper._download`` refuses remote/WAN URLs — including the official
Azure CDN — before opening a socket. Cache hits and ``file:`` / loopback
URLs are not fetches. Default compute is CPU when ``WHISPER_DEVICE=cpu``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from ipaddress import ip_address
from typing import Optional
from urllib.parse import urlparse

BIND_HOST = "127.0.0.1"
BIND_HOST_ENV = "WHISPER_BIND_HOST"
ALLOW_WEIGHT_FETCH_ENV = "WHISPER_ALLOW_WEIGHT_FETCH"
DEVICE_ENV = "WHISPER_DEVICE"
_TRUTHY = {"1", "true", "yes", "on"}


class BindError(ValueError):
    """Raised when a listen address is not 127.0.0.1."""


class WeightFetchError(RuntimeError):
    """Raised when a default / remote weight fetch is refused."""


def require_bind_127_0_0_1(host: Optional[str]) -> str:
    """Return ``127.0.0.1`` or raise ``BindError`` before a socket opens.

    Only the literal IPv4 loopback address is accepted. ``0.0.0.0``,
    ``::``, ``localhost``, ``::1``, LAN, and WAN addresses are rejected.
    """
    normalized = "" if host is None else str(host).strip()
    if normalized == BIND_HOST:
        return BIND_HOST
    raise BindError(
        f"bind must be {BIND_HOST} only, got {host!r}. "
        "Do not listen on 0.0.0.0 or any other interface."
    )


def resolve_device(device: Optional[str] = None) -> str:
    """Return the inference device. Default path is CPU when forced.

    ``WHISPER_DEVICE=cpu`` (environment default) always selects CPU.
    An explicit ``device`` argument wins.
    """
    import torch

    if device:
        return device
    forced = os.environ.get(DEVICE_ENV, "").strip().lower()
    if forced == "cpu":
        return "cpu"
    if forced == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def weight_fetch_allowed() -> bool:
    """True only when ``WHISPER_ALLOW_WEIGHT_FETCH`` is an explicit yes.

    Missing or any other value means no default weight fetch.
    """
    raw = os.environ.get(ALLOW_WEIGHT_FETCH_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def hostname_is_loopback(host: Optional[str]) -> bool:
    """True for ``localhost`` and loopback IPs. Hostnames are not resolved."""
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ip_address(host.strip()).is_loopback
    except ValueError:
        return False


def url_is_loopback(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    return hostname_is_loopback(parsed.hostname)


def refuse_default_weight_fetch(url: str) -> None:
    """Refuse remote/WAN checkpoint URLs unless fetch is explicitly allowed."""
    if weight_fetch_allowed():
        return
    if url_is_loopback(url):
        return
    host = urlparse(url).hostname or "<unknown>"
    raise WeightFetchError(
        f"Refusing default weight fetch from host {host!r}. "
        f"Set {ALLOW_WEIGHT_FETCH_ENV}=1 only when a WAN pull is intentional. "
        "The environment default is no weight fetch; serve checkpoints from "
        f"http://{BIND_HOST}/... or use a cache hit."
    )


def loopback_only_opener() -> urllib.request.OpenerDirector:
    """Opener that never consults HTTP(S)_PROXY.

    ``build_opener()`` would otherwise install the environment
    ``ProxyHandler``, sending an allowed ``http://127.0.0.1/...`` request
    to a remote proxy. Passing an empty ``ProxyHandler`` suppresses that
    default (urllib then drops the empty handler). Loopback pulls stay
    on-box; the redirect handler still refuses WAN targets.
    """
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _LoopbackOnlyRedirectHandler(),
    )


def urlopen_for_weights(url: str):
    """``urlopen`` that refuses a default remote weight fetch."""
    refuse_default_weight_fetch(url)
    if not weight_fetch_allowed():
        try:
            return loopback_only_opener().open(url)
        except WeightFetchError:
            raise
        except urllib.error.URLError as exc:
            raise WeightFetchError(
                f"Loopback-only weight pull failed for {url!r}: {exc}"
            ) from exc
    return urllib.request.urlopen(url)


class _LoopbackOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        refuse_default_weight_fetch(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)
