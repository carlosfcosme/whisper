"""Offline weight policy: no Hub / Azure fetch by default.

Local cache and local checkpoint paths still work. Network download is
opt-in via WHISPER_ALLOW_DOWNLOADS=1. WHISPER_OFFLINE, HF_HUB_OFFLINE,
and TRANSFORMERS_OFFLINE always keep downloads off.

This module does not read API keys and does not import huggingface_hub.
"""

from __future__ import annotations

import os

ALLOW_DOWNLOADS_ENV = "WHISPER_ALLOW_DOWNLOADS"
OFFLINE_ENV_VARS = (
    "WHISPER_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_is_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def offline_forced() -> bool:
    return any(env_is_truthy(key) for key in OFFLINE_ENV_VARS)


def downloads_allowed() -> bool:
    """False by default. True only with an explicit opt-in and no offline flag."""
    if offline_forced():
        return False
    return env_is_truthy(ALLOW_DOWNLOADS_ENV)


def downloads_forbidden() -> bool:
    return not downloads_allowed()


def refuse_download(target: str) -> None:
    raise RuntimeError(
        "Refusing to download model weights (default is offline; "
        "no Hugging Face Hub / Azure fetch). "
        "Use a local checkpoint or set {}=1. "
        "Missing or invalid local cache: {}".format(ALLOW_DOWNLOADS_ENV, target)
    )
