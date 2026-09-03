"""Local-only fixture paths.

Audio and packaged assets must be in-repo files or tempfiles. HTTP(S) and
Hugging Face Hub URLs are refused so pytest never needs WAN. This module
does not load model weights and does not read secrets.
"""

from __future__ import annotations

import os
import wave
from typing import Union

PathLike = Union[str, os.PathLike]

REMOTE_SCHEMES = ("http://", "https://")
HUB_MARKERS = ("huggingface.co", "hf.co/")
WEIGHT_DOWNLOAD_MARKERS = ("openaipublic.azureedge.net", "openaipublic")


class RemoteFixtureError(ValueError):
    """Raised when a fixture path is an HTTP(S) or Hub URL."""


def is_hub_url(path: PathLike) -> bool:
    """True for Hugging Face Hub / hf.co addresses."""
    text = str(path).strip().lower()
    return any(marker in text for marker in HUB_MARKERS)


def is_remote_fixture_url(path: PathLike) -> bool:
    """True for http(s) URLs and Hugging Face Hub / hf.co addresses."""
    text = str(path).strip().lower()
    if text.startswith(REMOTE_SCHEMES):
        return True
    return is_hub_url(text)


def is_weight_download_url(path: PathLike) -> bool:
    """True for official Whisper checkpoint CDN URLs."""
    text = str(path).strip().lower()
    return any(marker in text for marker in WEIGHT_DOWNLOAD_MARKERS)


def offline_requested() -> bool:
    """True when WHISPER_OFFLINE asks CI/tests not to pull weights."""
    flag = os.getenv("WHISPER_OFFLINE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def assert_local_fixture(path: PathLike, must_exist: bool = True) -> str:
    """Return an absolute local path, or raise ``RemoteFixtureError``.

    Remote http(s) and Hugging Face URLs are always rejected. When
    ``must_exist`` is true the path must already be a regular file
    (in-repo asset or a tempfile that has been written).
    """
    if is_remote_fixture_url(path):
        raise RemoteFixtureError(
            "Fixture path must be local (in-repo or tempfile); refused {0!r}".format(
                str(path)
            )
        )
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if is_remote_fixture_url(resolved):
        raise RemoteFixtureError(
            "Fixture path must be local (in-repo or tempfile); refused {0!r}".format(
                resolved
            )
        )
    if must_exist and not os.path.isfile(resolved):
        raise RemoteFixtureError(
            "Fixture path is not a local file: {0}".format(resolved)
        )
    return resolved


def write_tiny_wav(
    path: PathLike, seconds: float = 0.25, sample_rate: int = 16000
) -> str:
    """Write a tiny silent WAV to ``path`` and return the local file path."""
    target = assert_local_fixture(path, must_exist=False)
    frames = int(sample_rate * seconds)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with wave.open(target, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return assert_local_fixture(target, must_exist=True)
