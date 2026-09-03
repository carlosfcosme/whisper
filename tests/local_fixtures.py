"""Local-only test fixture paths. Never fetch over WAN or the Hub."""

import math
import os
import re
import struct
import sys
import wave
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_AUDIO_NAME = "jfk.flac"
SAMPLE_AUDIO_PATH = str(TESTS_DIR / SAMPLE_AUDIO_NAME)

# Scanner self-tests mention forbidden URL shapes on purpose.
_SCAN_SKIP = frozenset({"local_fixtures.py", "test_local_fixtures.py"})
# Host-only strings like "huggingface.co" (Hub blocklist) are allowed.
# http(s)/hf schemes and Hub *paths* are fixture URLs and must fail.
_REMOTE_RE = re.compile(
    r"https?://|hf://|huggingface\.co/[\w.-]+|hf\.co/[\w.-]+",
    re.IGNORECASE,
)


def assert_local_fixture_path(path):
    """Raise if a fixture path is a URL, Hub ref, or missing local file."""
    text = os.fspath(path)
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "hf://")):
        raise ValueError("local fixture must not be a WAN/Hub URL: {0}".format(text))
    if "huggingface.co" in lowered or "hf.co/" in lowered:
        raise ValueError(
            "local fixture must not use Hugging Face Hub: {0}".format(text)
        )
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("local fixture file is missing: {0}".format(resolved))
    return str(resolved)


def sample_audio_path():
    """In-repo JFK clip. No download URL."""
    return assert_local_fixture_path(SAMPLE_AUDIO_PATH)


def write_tiny_wav(path, seconds=0.25, sample_rate=16000):
    """Write a short local sine WAV. No network."""
    dest = Path(path)
    n_samples = int(seconds * sample_rate)
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(
            struct.pack(
                "<h",
                int(32767 * 0.1 * math.sin(2 * math.pi * 440.0 * index / sample_rate)),
            )
            for index in range(n_samples)
        )
        handle.writeframes(frames)
    return assert_local_fixture_path(dest)


def remote_fixture_url_hits():
    """Return (file, line, text) for http(s)/Hub fixture URLs under tests/."""
    hits = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name in _SCAN_SKIP:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _REMOTE_RE.search(line):
                hits.append((path.name, lineno, line.strip()))
    return hits


def main(argv=None):
    del argv
    sample_audio_path()
    hits = remote_fixture_url_hits()
    if hits:
        for name, lineno, line in hits:
            sys.stderr.write("{0}:{1}: {2}\n".format(name, lineno, line))
        return 1
    sys.stdout.write("no remote fixture URLs\n")
    sys.stdout.write("local fixtures OK: {0}\n".format(SAMPLE_AUDIO_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
