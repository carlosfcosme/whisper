#!/usr/bin/env python3
"""Fail CI if .gitignore omits cache or weight patterns."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.bin",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    ".cache/",
    "**/.cache/whisper/",
    ".huggingface/",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def missing_patterns(text: str) -> list:
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in text]


def main() -> int:
    path = repo_root() / ".gitignore"
    if not path.is_file():
        sys.stderr.write("ERROR: .gitignore is missing\n")
        return 1
    missing = missing_patterns(path.read_text())
    if missing:
        sys.stderr.write("ERROR: .gitignore is missing cache/weight patterns:\n")
        for pattern in missing:
            sys.stderr.write("  {}\n".format(pattern))
        return 1
    sys.stdout.write("OK: .gitignore covers cache and weight patterns\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
