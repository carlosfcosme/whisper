#!/usr/bin/env python3
"""Fail CI if weight/cache paths are not gitignored.

Enforces the .gitignore patterns that keep downloaded checkpoints and
tool caches out of git. No Hub, no weight pull, no keys.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_PATTERNS = (
    ".cache/",
    "cache/",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    "*.gguf",
    "__pycache__/",
    ".pytest_cache",
)

MUST_IGNORE = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "tiny.pt",
    "model.pth",
    "weights.safetensors",
    "checkpoint.bin",
    "__pycache__/x.pyc",
    ".pytest_cache/v/cache",
)

MUST_NOT_IGNORE = (
    "tests/jfk.flac",
    "whisper/assets/mel_filters.npz",
    "whisper/assets/multilingual.tiktoken",
    "README.md",
    ".gitignore",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gitignore_text(root: Path) -> str:
    return (root / ".gitignore").read_text(encoding="utf-8")


def check_ignore(root: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=root,
    )
    return result.returncode == 0


def find_violations(root: Path) -> list:
    errors = []
    text = gitignore_text(root)
    for pattern in REQUIRED_PATTERNS:
        if pattern not in text:
            errors.append("missing .gitignore pattern {!r}".format(pattern))
    for path in MUST_IGNORE:
        if not check_ignore(root, path):
            errors.append(
                "not ignored (weights/cache must stay out of git): {}".format(path)
            )
    for path in MUST_NOT_IGNORE:
        if check_ignore(root, path):
            errors.append("fixture/source incorrectly ignored: {}".format(path))
    return errors


def main() -> int:
    errors = find_violations(repo_root())
    if errors:
        sys.stderr.write("ERROR: gitignore does not enforce weight/cache rules\n")
        for message in errors:
            sys.stderr.write("  {}\n".format(message))
        return 1
    sys.stdout.write("OK: gitignore covers weight and cache paths\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
