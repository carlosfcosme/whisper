#!/usr/bin/env python3
"""Fail CI if a non-loopback all-interface bind appears outside tests.

Listen on 127.0.0.1 only. This is the source of truth for the
all-interfaces grep and is invoked from GitHub Actions and tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

FORBIDDEN = ".".join(("0", "0", "0", "0"))
SKIP_DIR_PARTS = {".git", "tests"}
SKIP_NAMES = {"check_loopback_bind.py"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def is_skipped(relpath: str) -> bool:
    parts = Path(relpath).parts
    if any(part in SKIP_DIR_PARTS for part in parts):
        return True
    name = Path(relpath).name
    if name in SKIP_NAMES:
        return True
    return name.startswith("test_") or name.endswith("_test.py")


def find_hits(root: Path) -> List[Tuple[str, int, str]]:
    hits: List[Tuple[str, int, str]] = []
    for relpath in tracked_files(root):
        if is_skipped(relpath):
            continue
        path = root / relpath
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN in line:
                hits.append((relpath, lineno, line.strip()))
    return hits


def main() -> int:
    root = repo_root()
    hits = find_hits(root)
    if hits:
        sys.stderr.write(
            "ERROR: bind/listen must be 127.0.0.1; {} is forbidden outside tests:\n".format(
                FORBIDDEN
            )
        )
        for relpath, lineno, line in hits:
            sys.stderr.write("  {}:{}: {}\n".format(relpath, lineno, line))
        return 1
    sys.stdout.write(
        "OK: no {} outside tests; bind stays 127.0.0.1\n".format(FORBIDDEN)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
