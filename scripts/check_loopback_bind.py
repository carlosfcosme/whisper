#!/usr/bin/env python3
"""Fail if package code binds off loopback (0.0.0.0 / :: / non-127)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "whisper"
WILDCARD_V4 = ".".join(("0", "0", "0", "0"))
FORBIDDEN_CONSTANTS = frozenset({WILDCARD_V4, "::", "[::]"})


def find_forbidden_binds() -> List[str]:
    offenders: List[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in FORBIDDEN_CONSTANTS:
                offenders.append(
                    "{}:{}: {}".format(path.relative_to(REPO), node.lineno, node.value)
                )
    return offenders


def bind_host_is_loopback() -> List[str]:
    text = (PACKAGE / "offline.py").read_text(encoding="utf-8")
    errors: List[str] = []
    if 'BIND_HOST = "127.0.0.1"' not in text and "BIND_HOST = '127.0.0.1'" not in text:
        errors.append("whisper.offline.BIND_HOST is not 127.0.0.1")
    if "def require_loopback_bind" not in text:
        errors.append("require_loopback_bind is missing")
    if "def bind_loopback" not in text:
        errors.append("bind_loopback is missing")
    return errors


def main() -> int:
    errors = bind_host_is_loopback() + find_forbidden_binds()
    if errors:
        sys.stderr.write("ERROR: loopback-only bind policy failed:\n")
        for line in errors:
            sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: bind policy is 127.0.0.1 only\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
