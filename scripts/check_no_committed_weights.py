#!/usr/bin/env python3
"""Fail CI if the git tree tracks model weights or local cache directories."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".ckpt", ".onnx")
CACHE_PREFIXES = (".cache/", "cache/", "weights/")
REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
)


def repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def tracked_files(root: Path) -> list[str]:
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    return [p.decode() for p in raw.split(b"\0") if p]


def main() -> int:
    root = repo_root()
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    missing = [pat for pat in REQUIRED_GITIGNORE if pat not in gitignore]
    if missing:
        print(
            "ERROR: .gitignore missing cache/weight patterns: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    bad = []
    for path in tracked_files(root):
        lower = path.lower().replace("\\", "/")
        if lower.endswith(WEIGHT_SUFFIXES):
            bad.append(path)
        elif any(
            lower == prefix[:-1] or lower.startswith(prefix)
            for prefix in CACHE_PREFIXES
        ):
            bad.append(path)
    if bad:
        print("ERROR: committed weight/cache paths:", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        return 1

    print("OK: no committed weights or cache directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
