#!/usr/bin/env python3
"""CI assertion: gitignore covers weight and cache paths.

Uses ``git check-ignore`` (no network, no secrets, no weight download).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUST_IGNORE = (
    "weights/model.pt",
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    ".huggingface/hub/models--x",
    "orphan.pt",
    "orphan.pth",
    "orphan.safetensors",
    "orphan.bin",
    "orphan.onnx",
    "orphan.gguf",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def ignored(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(ROOT),
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    missing = [path for path in MUST_IGNORE if not ignored(path)]
    if missing:
        fail("gitignore must ignore: {}".format(missing))
    print("gitignore-caches: ok (weights/caches ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
