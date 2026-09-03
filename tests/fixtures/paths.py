"""Resolve audio/model fixtures to in-repo or tempfile paths only."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Union

FIXTURES_DIR = Path(__file__).resolve().parent
TESTS_DIR = FIXTURES_DIR.parent

_REMOTE_SCHEMES = frozenset({"http", "https"})


def is_remote_url(value: str) -> bool:
    """True when a path is a WAN URL (http(s) or Hub host), not a local file."""
    text = (value or "").strip()
    lowered = text.lower()
    scheme, sep, rest = lowered.partition("://")
    if sep and scheme in _REMOTE_SCHEMES:
        host = rest.split("/", 1)[0].split(":", 1)[0]
        if host in {"127.0.0.1", "localhost", "::1"}:
            return False
        return True
    if "huggingface" + ".co" in lowered or "hf.co/" in lowered:
        return True
    return False


def local_path(path: Union[str, Path]) -> Path:
    """Return a resolved local file path, or raise if the value is remote."""
    text = str(path)
    if is_remote_url(text):
        raise ValueError(
            "fixture paths must be local files (got remote URL {!r})".format(text)
        )
    resolved = Path(text).expanduser()
    if not resolved.is_absolute():
        resolved = (FIXTURES_DIR / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError("local fixture missing: {}".format(resolved))
    return resolved


def fixture_path(name: str) -> Path:
    """Path to a file under tests/fixtures/. Rejects remote URLs."""
    return local_path(FIXTURES_DIR / name)


def repo_audio_path(name: str = "jfk.flac") -> Path:
    """Path to an in-repo sample under tests/ (not a WAN URL)."""
    return local_path(TESTS_DIR / name)


def write_sine_wav(
    path: Union[str, Path],
    seconds: float = 0.25,
    freq: float = 440.0,
    sample_rate: int = 16000,
) -> Path:
    """Write a tiny mono WAV to a local path (tempfile or fixtures/)."""
    if is_remote_url(str(path)):
        raise ValueError("cannot write a fixture to a remote URL")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(sample_rate * seconds))
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(n_frames):
            sample = int(16000 * math.sin(2.0 * math.pi * freq * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))
    return dest.resolve()
