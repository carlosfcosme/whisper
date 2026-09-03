#!/usr/bin/env python3
"""CI assertion: no committed model weights or cache directories.

Weights belong in a local cache, not git. Stdlib only. No network.
No Hub. No secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
        ".tflite",
        ".pb",
        ".mar",
        ".params",
        ".weights",
    }
)
CACHE_PARTS = frozenset({".cache", "cache", "weights"})
MAX_FILE_BYTES = 10 * 1024 * 1024
REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str, size: int) -> Optional[str]:
    posix = relpath.replace("\\", "/")
    parts = set(Path(posix).parts)
    suffix = Path(posix).suffix.lower()
    if parts & CACHE_PARTS:
        return "cache/weight directory path"
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
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        reason = classify(relpath, size)
        if reason:
            violations.append((relpath, reason))
    return violations


def check_gitignore(root: Path) -> None:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        fail("missing .gitignore")
    text = gitignore.read_text(encoding="utf-8")
    missing = [pattern for pattern in REQUIRED_GITIGNORE if pattern not in text]
    if missing:
        fail("gitignore missing weight/cache patterns: {}".format(missing))


def main() -> int:
    check_gitignore(ROOT)
    violations = find_violations(ROOT)
    if violations:
        for relpath, reason in violations:
            print(f"FAIL: tracked {relpath}: {reason}", file=sys.stderr)
        fail("committed weights or caches are not allowed")
    print("no-weights: ok (no tracked checkpoints or cache dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
