#!/usr/bin/env python3
"""Verify cache/weight artifacts are gitignored and not tracked."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Sequence

ARTIFACTS = (
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    "weights/model.safetensors",
    "checkpoints/model.ckpt",
    "tiny.pt",
    "model.bin",
)


def repo_root() -> str:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return out.strip()


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=repo_root(),
    )
    return result.returncode == 0


def is_tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=repo_root(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def missing_ignore_rules() -> List[str]:
    return [path for path in ARTIFACTS if not is_ignored(path)]


def tracked_artifacts() -> List[str]:
    return [path for path in ARTIFACTS if is_tracked(path)]


def porcelain_untracked(paths: Sequence[str]) -> List[str]:
    """Return planted paths that git would add as untracked (??)."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=repo_root(),
        text=True,
    )
    leaked: List[str] = []
    for line in status.splitlines():
        if line.startswith("??"):
            leaked.append(line[3:].strip())
    return leaked


def plant_and_verify() -> List[str]:
    root = repo_root()
    planted: List[str] = []
    try:
        for rel in ARTIFACTS:
            full = os.path.join(root, rel)
            parent = os.path.dirname(full)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full, "wb") as handle:
                handle.write(b"not-a-real-checkpoint")
            planted.append(full)
        return porcelain_untracked(ARTIFACTS)
    finally:
        for full in planted:
            try:
                os.remove(full)
            except OSError:
                pass


def main() -> int:
    missing = missing_ignore_rules()
    tracked = tracked_artifacts()
    leaked = plant_and_verify()
    if not missing and not tracked and not leaked:
        print("ignored artifacts ok")
        return 0
    if missing:
        print("not gitignored:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    if tracked:
        print("tracked artifacts:", file=sys.stderr)
        for path in tracked:
            print(f"  {path}", file=sys.stderr)
    if leaked:
        print("planted files showed as untracked:", file=sys.stderr)
        for path in leaked:
            print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
