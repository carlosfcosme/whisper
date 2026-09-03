#!/usr/bin/env python3
"""Fail CI if library or start scripts bind all interfaces (not localhost)."""

from __future__ import annotations

import sys
from pathlib import Path

WILDCARD = "0.0." + "0.0"
SCAN_GLOBS = (
    "whisper/*.py",
    ".cursor/*.sh",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = repo_root()
    hits = []
    for pattern in SCAN_GLOBS:
        for path in sorted(root.glob(pattern)):
            text = path.read_text(errors="replace")
            if WILDCARD in text:
                hits.append(str(path.relative_to(root)))
    if hits:
        print(
            "All-interfaces bind (0.0.0.0) is not allowed; use 127.0.0.1:",
            file=sys.stderr,
        )
        for hit in hits:
            print(f"  - {hit}", file=sys.stderr)
        return 1
    print("No all-interfaces bind in library/start scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
