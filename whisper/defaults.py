"""Sovereign runtime defaults: CPU, loopback, offline, no-store."""

from __future__ import annotations

import os

DEFAULT_DEVICE = "cpu"
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_OFFLINE = True
DEFAULT_NO_STORE = True

_FALSEY = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def offline_enabled() -> bool:
    """Remote weight fetches are off by default. Opt in with WHISPER_OFFLINE=0."""
    if os.getenv("WHISPER_OFFLINE") is not None:
        return _env_flag("WHISPER_OFFLINE", DEFAULT_OFFLINE)
    if os.getenv("HF_HUB_OFFLINE") is not None:
        return _env_flag("HF_HUB_OFFLINE", DEFAULT_OFFLINE)
    return DEFAULT_OFFLINE


def no_store_enabled() -> bool:
    """Do not persist checkpoints. Opt in to storing with WHISPER_NO_STORE=0."""
    return _env_flag("WHISPER_NO_STORE", DEFAULT_NO_STORE)
