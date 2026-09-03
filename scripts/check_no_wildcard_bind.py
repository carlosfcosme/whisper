#!/usr/bin/env python3
"""Fail CI if production code/config binds to a wildcard host.

Scans application and config paths for forbidden listen patterns
(unspecified IPv4 and ``--host`` wildcard). Tests may mention those
patterns when asserting they are rejected. This script does not fetch
Hub artifacts or model weights.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Patterns that mean "listen on all interfaces".
FORBIDDEN_SUBSTRINGS = (
    "0.0.0.0",
    "--host 0.0.0.0",
    "--host=0.0.0.0",
)

SCAN_ROOTS = (
    "whisper",
    ".github",
    ".cursor",
    "scripts",
    "pyproject.toml",
)

SKIP_NAMES = frozenset(
    {
        "scripts/check_no_wildcard_bind.py",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def _under_scan_root(relpath: str) -> bool:
    posix = relpath.replace("\\", "/")
    if posix in SKIP_NAMES:
        return False
    for prefix in SCAN_ROOTS:
        if posix == prefix or posix.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        if not _under_scan_root(relpath):
            continue
        path = root / relpath
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_SUBSTRINGS:
            if pattern in text:
                violations.append((relpath, pattern))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write(
            "ERROR: forbidden wildcard bind pattern in production code/config:\n"
        )
        for relpath, pattern in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, pattern))
        sys.stderr.write("Listen only on 127.0.0.1 (never wildcard / empty host).\n")
        return 1
    sys.stdout.write("OK: no wildcard bind patterns in production code/config\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
