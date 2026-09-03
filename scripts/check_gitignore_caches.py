#!/usr/bin/env python3
"""Fail CI if .gitignore does not cover weight and cache paths.

Implementation check (not documentation): the ignore file must list weight
globs and local Hub/Whisper cache directories so they cannot be committed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

REQUIRED_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.bin",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    "*.gguf",
    ".cache/",
    ".cache/whisper/",
    ".cache/huggingface/",
    ".huggingface/",
    "hf_cache/",
    "whisper_cache/",
    "checkpoints/",
    ".env",
    "*.pem",
    "*.key",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gitignore_entries(root: Path) -> List[str]:
    path = root / ".gitignore"
    entries: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def missing_patterns(root: Path) -> List[str]:
    present = set(gitignore_entries(root))
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in present]


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    root = repo_root()
    missing = missing_patterns(root)
    if missing:
        sys.stderr.write(
            "ERROR: .gitignore must cover weight and cache paths; missing:\n"
        )
        for pattern in missing:
            sys.stderr.write("  {}\n".format(pattern))
        return 1
    sys.stdout.write("OK: .gitignore covers weight and cache paths\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
