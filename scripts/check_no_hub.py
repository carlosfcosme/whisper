#!/usr/bin/env python3
"""Fail CI if unit tests or app code use a Hugging Face Hub client."""

from __future__ import annotations

import sys
from pathlib import Path

CLIENT_TOKENS = (
    "huggingface_hub",
    "hf_hub_download",
    "snapshot_download",
    "from_pretrained",
)

# Tests that assert Hub is refused may name the tokens they forbid.
ALLOW_NAMES = frozenset(
    {
        "test_no_hub.py",
        "check_no_hub.py",
        "check_cpu_offline.py",
        "conftest.py",
        "offline.py",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_violations(root: Path):
    violations = []
    for folder in (root / "tests", root / "whisper", root / "scripts"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            if path.name in ALLOW_NAMES:
                continue
            text = path.read_text(encoding="utf-8")
            for token in CLIENT_TOKENS:
                if token in text:
                    violations.append((str(path.relative_to(root)), token))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write("ERROR: Hugging Face Hub client is forbidden:\n")
        for relpath, token in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, token))
        return 1
    sys.stdout.write("OK: no Hub client in tests or application code\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
