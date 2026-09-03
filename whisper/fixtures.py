"""Local-only test fixture paths. Never downloads. Never uses WAN.

This module is stdlib-only so CI can import it without torch or Hub.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
TINY_FIXTURE_NAME = "tone.wav"
TINY_FIXTURE_PATH = FIXTURES_DIR / TINY_FIXTURE_NAME
JFK_FIXTURE_PATH = TESTS_DIR / "jfk.flac"

# 50 ms @ 16 kHz mono 16-bit ≈ 1.6 KiB plus WAV header.
TINY_SECONDS = 0.05
TINY_SAMPLE_RATE = 16000
TINY_HZ = 440.0


class RemoteFixtureError(ValueError):
    """Raised when a fixture path is a remote URL instead of a local file."""


def is_remote_asset_url(value: str) -> bool:
    """True for http(s)/ftp URLs that are not loopback."""
    raw = (value or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https", "ftp"}:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in {"127.0.0.1", "localhost", "::1"}:
        return False
    return True


def require_local_fixture(
    path: Union[str, Path], *, root: Optional[Path] = None
) -> Path:
    """Return an existing local fixture path, or raise.

    Remote asset URLs are refused. Relative paths resolve under ``tests/``.
    The file must already exist on disk; nothing is downloaded.
    """
    raw = str(path).strip()
    if is_remote_asset_url(raw):
        raise RemoteFixtureError(
            f"refusing remote fixture URL {raw!r}; use a local path under tests/"
        )
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "ftp"}:
        # Loopback URLs are not fixture files.
        raise RemoteFixtureError(
            f"refusing non-file fixture {raw!r}; use a local path under tests/"
        )
    base = Path(root) if root is not None else TESTS_DIR
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"local fixture missing: {resolved}")
    return resolved


def write_tiny_wav(
    path: Optional[Path] = None,
    *,
    seconds: float = TINY_SECONDS,
    sample_rate: int = TINY_SAMPLE_RATE,
) -> Path:
    """Write a tiny mono WAV locally. No network. No weights. No keys."""
    target = Path(path) if path is not None else TINY_FIXTURE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(sample_rate * seconds))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            sample = int(8000 * math.sin(2.0 * math.pi * TINY_HZ * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wav.writeframes(bytes(frames))
    return target.resolve()


def tiny_fixture_path(*, generate: bool = True) -> Path:
    """Return the committed/generated tiny WAV, creating it locally if needed."""
    if TINY_FIXTURE_PATH.is_file():
        return TINY_FIXTURE_PATH.resolve()
    if not generate:
        raise FileNotFoundError(f"tiny fixture missing: {TINY_FIXTURE_PATH}")
    return write_tiny_wav(TINY_FIXTURE_PATH)


def sample_audio_path() -> Path:
    """In-repo JFK sample used by audio/transcribe tests. Local file only."""
    return require_local_fixture(JFK_FIXTURE_PATH)
