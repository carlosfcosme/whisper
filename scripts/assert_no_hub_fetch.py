#!/usr/bin/env python3
"""Fail if CI/tests fetched from the Hugging Face Hub.

Checks:
* HF_HUB_OFFLINE / WHISPER_OFFLINE are set (when --require-offline-env)
* no Hub token env vars are present (values are never printed)
* the Hub cache has no downloaded blobs/snapshots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

_OFFLINE_ENV_NAMES = (
    "HF_HUB_OFFLINE",
    "WHISPER_OFFLINE",
)


def hub_cache_roots() -> List[Path]:
    home = Path.home()
    xdg = os.environ.get("XDG_CACHE_HOME")
    roots: List[Optional[Path]] = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface" / "modules",
        home / ".huggingface" / "hub",
        Path(xdg) / "huggingface" / "hub" if xdg else None,
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else None,
    ]
    seen = set()
    unique: List[Path] = []
    for root in roots:
        if root is None:
            continue
        resolved = root.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def find_hub_artifacts(roots: Optional[Sequence[Path]] = None) -> List[Path]:
    found: List[Path] = []
    for root in roots if roots is not None else hub_cache_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                found.append(path)
    return sorted(found)


def leaked_token_names() -> List[str]:
    """Return names of Hub token env vars that are set. Never returns values."""
    return [name for name in _TOKEN_ENV_NAMES if os.environ.get(name)]


def offline_env_missing() -> List[str]:
    missing: List[str] = []
    for name in _OFFLINE_ENV_NAMES:
        if os.environ.get(name, "").strip() not in {"1", "true", "yes", "on"}:
            missing.append(name)
    return missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if the Hub was fetched")
    parser.add_argument(
        "--require-offline-env",
        action="store_true",
        help="also fail when HF_HUB_OFFLINE/WHISPER_OFFLINE are unset",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    failed = False
    tokens = leaked_token_names()
    if tokens:
        failed = True
        sys.stderr.write(
            "ERROR: Hub token environment variables must not be set in CI/tests:\n"
        )
        for name in tokens:
            sys.stderr.write("  {}\n".format(name))
    if args.require_offline_env:
        missing = offline_env_missing()
        if missing:
            failed = True
            sys.stderr.write(
                "ERROR: offline env vars required (CI must not hit the Hub):\n"
            )
            for name in missing:
                sys.stderr.write("  {}\n".format(name))
    artifacts = find_hub_artifacts()
    if artifacts:
        failed = True
        sys.stderr.write("ERROR: Hugging Face Hub cache is not empty:\n")
        for path in artifacts:
            sys.stderr.write("  {}\n".format(path))
    if failed:
        return 1
    sys.stdout.write("OK: no Hub fetch (offline, no tokens, empty Hub cache)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
