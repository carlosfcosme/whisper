#!/usr/bin/env python3
"""Fail CI if unit tests reference the Hugging Face Hub."""

import sys
from pathlib import Path

HUB_TOKENS = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "from_pretrained",
    "hf.co/",
)

# Blocker / this checker may name the tokens they forbid.
ALLOW_NAMES = frozenset({"test_no_hub.py", "check_no_hub.py", "conftest.py"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_violations(root: Path):
    violations = []
    tests = root / "tests"
    if not tests.is_dir():
        return violations
    for path in tests.glob("test_*.py"):
        if path.name in ALLOW_NAMES:
            continue
        text = path.read_text(encoding="utf-8")
        for token in HUB_TOKENS:
            if token in text:
                violations.append((str(path.relative_to(root)), token))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write("ERROR: Hugging Face Hub is forbidden in tests:\n")
        for relpath, token in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, token))
        return 1
    sys.stdout.write("OK: no Hub references in unit tests\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
