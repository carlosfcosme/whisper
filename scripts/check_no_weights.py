#!/usr/bin/env python3
"""Fail CI if model weights or cache dirs are tracked.

Checkpoints belong in a local cache (ignored by git), not in the tree.
This script does not download weights and does not talk to Hugging Face Hub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})
FORBIDDEN_PREFIXES = (".cache/", "cache/", "weights/")
REQUIRED_GITIGNORE = (".cache/", "cache/", "weights/", "*.pt", "*.pth")


def _git_ls_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def gitignore_lines(root: Path) -> set:
    path = root / ".gitignore"
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def missing_gitignore_patterns(root: Path) -> List[str]:
    lines = gitignore_lines(root)
    return [pattern for pattern in REQUIRED_GITIGNORE if pattern not in lines]


def classify_tracked(paths: Iterable[str]) -> List[str]:
    hits = []
    for relpath in paths:
        posix = relpath.replace("\\", "/")
        suffix = Path(posix).suffix.lower()
        if posix.startswith(FORBIDDEN_PREFIXES) or suffix in WEIGHT_SUFFIXES:
            hits.append(posix)
    return hits


def tracked_weight_hits(root: Path, paths: Optional[Sequence[str]] = None) -> List[str]:
    listed = list(paths) if paths is not None else _git_ls_files(root)
    return classify_tracked(listed)


def main() -> int:
    missing = missing_gitignore_patterns(ROOT)
    if missing:
        print("gitignore is missing cache/weight patterns:")
        for pattern in missing:
            print("  %s" % pattern)
        return 1
    hits = tracked_weight_hits(ROOT)
    if hits:
        print("model weights or cache artifacts must not be tracked:")
        for hit in hits:
            print("  %s" % hit)
        return 1
    print("ok: no tracked weights; cache/weight paths are gitignored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
