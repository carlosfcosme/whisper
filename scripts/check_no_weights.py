#!/usr/bin/env python3
"""Fail CI if model weights or large binaries are committed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".ckpt",
        ".ggml",
        ".gguf",
        ".h5",
        ".hdf5",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".weights",
        ".bin",
    }
)

# Official Whisper checkpoints start around 75 MiB. Existing fixtures stay
# under this limit (largest tracked file is ~5.7 MiB).
MAX_FILE_BYTES = 10 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str, size: int) -> Optional[str]:
    """Return a violation reason, or None if the file is allowed."""
    posix = relpath.replace("\\", "/")
    suffix = Path(posix).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    if size > MAX_FILE_BYTES:
        return "large file ({} bytes > {})".format(size, MAX_FILE_BYTES)
    return None


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        path = root / relpath
        if not path.is_file():
            continue
        reason = classify(relpath, path.stat().st_size)
        if reason:
            violations.append((relpath, reason))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write("ERROR: model weights must not be committed:\n")
        for relpath, reason in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, reason))
        sys.stderr.write("Do not add checkpoints to git.\n")
        return 1
    sys.stdout.write("OK: no model weights committed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
