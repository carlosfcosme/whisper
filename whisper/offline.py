"""Offline / no-WAN weight-pull policy."""

import os
from urllib.parse import urlparse

from .bind import BIND_HOST

_OFFLINE_VALUES = frozenset({"1", "true", "yes", "on"})


class WeightDownloadError(RuntimeError):
    """Raised when a model-weight pull is refused (offline / no-WAN)."""


def offline_enabled() -> bool:
    """True when ``WHISPER_OFFLINE`` requests no weight downloads."""
    return os.environ.get("WHISPER_OFFLINE", "").strip().lower() in _OFFLINE_VALUES


def is_loopback_or_file_url(url: str) -> bool:
    """True for ``file:`` URLs and ``http(s)://127.0.0.1/...``."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in ("http", "https"):
        return False
    return (parsed.hostname or "") == BIND_HOST


def refuse_weight_pull(url: str) -> None:
    """Raise ``WeightDownloadError`` for WAN / Hub / CDN weight URLs."""
    if is_loopback_or_file_url(url):
        return
    raise WeightDownloadError(
        "WHISPER_OFFLINE: refusing weight download from {0!r}".format(url)
    )
