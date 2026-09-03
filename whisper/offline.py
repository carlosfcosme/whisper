"""Offline weight-download policy.

When ``WHISPER_OFFLINE`` is set, cache misses must not hit the network.
This module does not talk to Hugging Face Hub and does not bind sockets.
"""

import os

OFFLINE_ENV = "WHISPER_OFFLINE"


class OfflineError(RuntimeError):
    """Raised when a weight download is refused."""


def downloads_allowed() -> bool:
    value = os.environ.get(OFFLINE_ENV, "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def refuse_weight_download(url: str) -> None:
    """Raise ``OfflineError`` when downloads are disabled."""
    if not downloads_allowed():
        raise OfflineError("weight download disabled (%s=1): %s" % (OFFLINE_ENV, url))
