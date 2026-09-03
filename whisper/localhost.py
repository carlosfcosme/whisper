"""Installer weight-pull policy: no default download, localhost only.

Official ``load_model(name)`` fetches checkpoints from
``openaipublic.azureedge.net`` on a cache miss. The Cloud Agent installer
must not do that. When ``WHISPER_NO_WEIGHT_DOWNLOAD`` is set (the installer
default), any network pull is refused. When ``WHISPER_LOCALHOST_ONLY`` is
set, only loopback / ``file:`` URLs are allowed.

A cache hit is a local file read, not a download.
Hostnames are not resolved (no DNS).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from ipaddress import ip_address
from typing import Optional
from urllib.parse import urlparse

NO_WEIGHT_DOWNLOAD_ENV = "WHISPER_NO_WEIGHT_DOWNLOAD"
LOCALHOST_ONLY_ENV = "WHISPER_LOCALHOST_ONLY"
_TRUTHY = {"1", "true", "yes", "on"}


class WeightPullError(RuntimeError):
    """Raised when a model-weight network pull is refused."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def no_weight_download_enabled() -> bool:
    return _env_truthy(NO_WEIGHT_DOWNLOAD_ENV)


def localhost_only_enabled() -> bool:
    return _env_truthy(LOCALHOST_ONLY_ENV)


def hostname_is_localhost(host: Optional[str]) -> bool:
    """True for ``localhost`` and loopback IPs (``127.0.0.0/8``, ``::1``).

    Names are not resolved, so ``localhost.evil.com`` is remote.
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


def refuse_weight_network_pull(url: str) -> None:
    """Refuse a network fetch of weights when installer/CI guards are on.

    No-op when neither env var is set, so the public ``pip install`` path
    still downloads official checkpoints on a cache miss.
    """
    if no_weight_download_enabled():
        host = urlparse(url).hostname or "<unknown>"
        raise WeightPullError(
            "Auto-download of model weights is disabled "
            f"({NO_WEIGHT_DOWNLOAD_ENV}=1). Refusing pull from host {host!r}. "
            "Pass a local checkpoint path to load_model()."
        )
    if localhost_only_enabled() and not url_is_localhost(url):
        host = urlparse(url).hostname or "<unknown>"
        raise WeightPullError(
            f"Refusing remote/WAN weight pull from host {host!r}. "
            f"The installer path is localhost-only ({LOCALHOST_ONLY_ENV}=1)."
        )


class LocalhostOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that would pull from a remote/WAN host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if no_weight_download_enabled() or (
            localhost_only_enabled() and not url_is_localhost(newurl)
        ):
            refuse_weight_network_pull(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_weight_url(url: str):
    """``urlopen`` that honors the installer no-download / localhost guards."""
    refuse_weight_network_pull(url)
    if localhost_only_enabled():
        opener = urllib.request.build_opener(LocalhostOnlyRedirectHandler())
        try:
            return opener.open(url)
        except WeightPullError:
            raise
        except urllib.error.URLError as exc:
            raise WeightPullError(
                f"Localhost-only weight pull failed for {url!r}: {exc}"
            ) from exc
    return urllib.request.urlopen(url)
