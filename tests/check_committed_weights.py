#!/usr/bin/env python3
"""Fail CI if model-weight files are committed (git-tracked).

Standalone stdlib script so the workflow job does not need PyTorch or Hub.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Iterable, List, Optional

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".gguf",
)


def is_weight_path(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(name.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def iter_tracked_paths(repo: Optional[str] = None) -> Iterable[str]:
    cmd = ["git", "ls-files", "-z"]
    if repo:
        cmd = ["git", "-C", repo, "ls-files", "-z"]
    raw = subprocess.check_output(cmd)
    if not raw:
        return []
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def committed_weight_paths(repo: Optional[str] = None) -> List[str]:
    return sorted(path for path in iter_tracked_paths(repo) if is_weight_path(path))


def main(repo: Optional[str] = None) -> int:
    bad = committed_weight_paths(repo)
    if bad:
        sys.stderr.write("CI fail: committed model weights:\n")
        for path in bad:
            sys.stderr.write("  %s\n" % path)
        return 1
    sys.stdout.write("OK: no committed model weights\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
