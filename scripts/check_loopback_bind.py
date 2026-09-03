#!/usr/bin/env python3
"""Fail CI if start scripts bind 0.0.0.0 instead of 127.0.0.1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

ALL_INTERFACES = "0.0.0.0"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_start_scripts(root: Path) -> List[Path]:
    """Return repo start scripts (start.sh / start-*.sh / environment start)."""
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        name = path.name
        if name == "start.sh" or (name.startswith("start-") and name.endswith(".sh")):
            found.append(path)
    env = root / ".cursor" / "environment.json"
    if env.is_file():
        found.append(env)
    return sorted(set(found))


def start_script_all_interface_hits(paths: Iterable[Path]) -> List[str]:
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if ALL_INTERFACES in text:
            hits.append(str(path))
    return hits


def assert_start_scripts_localhost_only(paths: Iterable[Path]) -> None:
    hits = start_script_all_interface_hits(paths)
    if hits:
        raise AssertionError(
            "{} is not allowed in start scripts: {}".format(ALL_INTERFACES, hits)
        )


def main() -> int:
    root = repo_root()
    scripts = discover_start_scripts(root)
    hits = start_script_all_interface_hits(scripts)
    if hits:
        sys.stderr.write(
            "ERROR: start scripts must bind 127.0.0.1, not {}:\n".format(ALL_INTERFACES)
        )
        for path in hits:
            sys.stderr.write("  {}\n".format(path))
        return 1
    sys.stdout.write(
        "OK: start scripts bind 127.0.0.1 (no {})\n".format(ALL_INTERFACES)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
