"""Shared runtime defaults. Stdlib only — safe to import from CLI and CI."""

import os

DEFAULT_DEVICE = "cpu"
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})
_OFFLINE_ENV_VARS = ("WHISPER_OFFLINE", "HF_HUB_OFFLINE")


def weights_download_forbidden() -> bool:
    """True when Hub/weight downloads must not hit the network."""
    return any(
        os.getenv(key, "").strip().lower() in _OFFLINE_TRUTHY
        for key in _OFFLINE_ENV_VARS
    )
