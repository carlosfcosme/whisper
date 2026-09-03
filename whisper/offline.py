"""Refuse checkpoint downloads when offline flags are set."""

import os

OFFLINE_ENV_VARS = ("WHISPER_OFFLINE", "HF_HUB_OFFLINE")


def weights_offline() -> bool:
    """True when tests/CI forbid fetching model weights."""
    return any(os.environ.get(name) == "1" for name in OFFLINE_ENV_VARS)
