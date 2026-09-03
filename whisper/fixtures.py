"""Local-only audio fixture paths. Remote URLs never resolve."""

import io
import os
import wave
from pathlib import Path
from typing import Union

_REMOTE_PREFIXES = ("http://", "https://")
_HUB_MARKERS = (
    "huggingface.co",
    "hf.co/",
    "hf-mirror.com",
    "huggingface_hub",
)
IN_REPO_SAMPLE_AUDIO = Path(__file__).resolve().parent.parent / "tests" / "jfk.flac"


class RemoteFixtureError(ValueError):
    """Raised when a fixture path is a remote URL."""


def is_remote_fixture_url(path: Union[str, os.PathLike]) -> bool:
    text = os.fspath(path).strip()
    lowered = text.lower()
    if lowered.startswith(_REMOTE_PREFIXES):
        return True
    return any(marker in lowered for marker in _HUB_MARKERS)


def require_local_fixture(path: Union[str, os.PathLike]) -> str:
    """Return an absolute local path, or refuse a remote/Hub URL."""
    text = os.fspath(path)
    if is_remote_fixture_url(text):
        raise RemoteFixtureError(
            "fixture paths must be in-repo or temp files, not remote URLs: %s" % text
        )
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("local fixture is not a file: %s" % resolved)
    return str(resolved)


def write_tiny_wav(
    path: Union[str, os.PathLike], n_samples: int = 160, sr: int = 16000
) -> str:
    """Write a tiny mono 16-bit PCM WAV (default 10 ms at 16 kHz)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(b"\x00\x00" * int(n_samples))
    return require_local_fixture(dest)


def tiny_wav_bytes(n_samples: int = 160, sr: int = 16000) -> bytes:
    """Return the same tiny WAV as bytes (no network, no Hub)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(b"\x00\x00" * int(n_samples))
    return buffer.getvalue()
