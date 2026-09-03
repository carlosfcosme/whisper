"""Local-only fixture paths. Remote / Hub / WAN URLs are rejected.

This module is stdlib-only so CI can run ``python3 whisper/local_fixtures.py``
without installing the package or touching the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

REMOTE_SCHEMES = frozenset(
    {"http", "https", "ftp", "ftps", "s3", "gs", "hf", "huggingface"}
)
_REMOTE_MARKERS = (
    b"http://",
    b"https://",
    b"hf://",
    b"huggingface://",
    b"huggingface.co",
    b"hf.co/",
)

# name -> path relative to the tests/ directory. Values must be local files.
REGISTERED: Dict[str, str] = {
    "jfk.flac": "jfk.flac",
    "tone.wav": "fixtures/tone.wav",
    "pcm16le.raw": "fixtures/pcm16le.raw",
}


def tests_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests"


def is_remote_fixture_url(path: str) -> bool:
    """True if *path* is a WAN, Hub, or other non-local URL."""
    text = path.strip()
    lowered = text.lower()
    if "://" in text:
        scheme = text.split("://", 1)[0].lower()
        if scheme != "file":
            return True
        if scheme in REMOTE_SCHEMES:
            return True
    if any(lowered.startswith(marker.decode("ascii")) for marker in _REMOTE_MARKERS):
        return True
    if "huggingface.co" in lowered or "hf.co/" in lowered:
        return True
    return False


def assert_local_path(path: str) -> str:
    """Raise ValueError if *path* would require a remote fetch."""
    if is_remote_fixture_url(path):
        raise ValueError(
            "Remote fixture URLs are not allowed (WAN/Hub); use a local path: "
            f"{path!r}"
        )
    return path


def resolve(name: str, *, root: Optional[Path] = None) -> Path:
    """Resolve a registered fixture name (or relative path) to a local file."""
    assert_local_path(name)
    rel = REGISTERED.get(name, name)
    assert_local_path(rel)
    base = (root or tests_dir()).resolve()
    path = (base / rel).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"Fixture path escapes the tests directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Local fixture missing: {path}")
    return path


def _file_contains_remote_url(path: Path) -> bool:
    data = path.read_bytes()
    return any(marker in data for marker in _REMOTE_MARKERS)


def check_registered_fixtures(
    *,
    extra_scan: Optional[Iterable[Path]] = None,
) -> int:
    """Return 0 if every registered fixture is a local file with no remote URL."""
    errors = []
    for name, rel in REGISTERED.items():
        if is_remote_fixture_url(name) or is_remote_fixture_url(rel):
            errors.append(f"remote registry entry: {name} -> {rel}")
            continue
        try:
            path = resolve(name)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if is_remote_fixture_url(str(path)):
            errors.append(f"resolved remote path: {path}")
        if _file_contains_remote_url(path):
            errors.append(f"remote URL bytes in fixture file: {path}")

    scan_roots = [tests_dir() / "fixtures"]
    if extra_scan:
        scan_roots.extend(extra_scan)
    for folder in scan_roots:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and _file_contains_remote_url(path):
                errors.append(f"remote URL bytes in {path}")

    if errors:
        for item in errors:
            print(item, file=sys.stderr)
        return 1
    print("OK: all fixture paths are local (no WAN/Hub URLs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_registered_fixtures())
