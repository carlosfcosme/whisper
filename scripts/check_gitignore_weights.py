#!/usr/bin/env python3
"""Fail CI if cache/weight paths are not gitignored."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SAMPLES = (
    "tiny.pt",
    "models/large.pth",
    "checkpoint.ckpt",
    "model.safetensors",
    "weights/model.gguf",
    ".cache/whisper/tiny.pt",
    "whisper/.cache/model.pt",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_ignored(root: Path, relpath: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=root,
        check=False,
    )
    return proc.returncode == 0


def main() -> int:
    root = repo_root()
    missing = [path for path in SAMPLES if not is_ignored(root, path)]
    if missing:
        print("Cache/weight paths are not gitignored:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1
    print("Cache/weight sample paths are gitignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
