#!/usr/bin/env python3
"""Fail CI if model weights are committed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
        ".npz",
    }
)

ALLOWLIST = frozenset({"whisper/assets/mel_filters.npz"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str) -> Optional[str]:
    if relpath in ALLOWLIST:
        return None
    suffix = Path(relpath).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return f"committed weight file: {relpath}"
    return None


def findings(root: Path, paths: Optional[Iterable[str]] = None) -> List[str]:
    issues = []
    for relpath in paths if paths is not None else tracked_files(root):
        reason = classify(relpath)
        if reason:
            issues.append(reason)
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    issues = findings(repo_root())
    if issues:
        print("Committed model weights are not allowed:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("No committed model weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
