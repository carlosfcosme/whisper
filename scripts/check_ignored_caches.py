#!/usr/bin/env python3
"""Fail CI if weight/cache paths are not gitignored, or are tracked.

Offline-safe: git + .gitignore only. No Hub, no model download, no WAN.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_IGNORE_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.gguf",
)

EXAMPLE_IGNORED_PATHS = (
    ".cache/whisper/tiny.pt",
    "cache/tiny.en.pt",
    "weights/base.pt",
    "tiny.pt",
    "model.pth",
    "export.onnx",
    "model.safetensors",
    "run.ckpt",
    "local.gguf",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gitignore_patterns(root: Path):
    text = (root / ".gitignore").read_text(encoding="utf-8")
    patterns = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def tracked_files(root: Path):
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def is_ignored(root: Path, relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(root),
    )
    return result.returncode == 0


def find_violations(root: Path):
    violations = []
    present = set(gitignore_patterns(root))
    for pattern in REQUIRED_IGNORE_PATTERNS:
        if pattern not in present:
            violations.append("missing .gitignore pattern: {}".format(pattern))

    for relpath in tracked_files(root):
        posix = relpath.replace("\\", "/")
        suffix = Path(posix).suffix.lower()
        if posix.startswith(".cache/") or posix.startswith("cache/"):
            violations.append("tracked cache path: {}".format(posix))
        if posix.startswith("weights/"):
            violations.append("tracked weights path: {}".format(posix))
        if suffix in {
            ".pt",
            ".pth",
            ".onnx",
            ".safetensors",
            ".ckpt",
            ".gguf",
        }:
            violations.append("tracked weight file: {}".format(posix))

    for relpath in EXAMPLE_IGNORED_PATHS:
        if not is_ignored(root, relpath):
            violations.append("path is not ignored: {}".format(relpath))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write("ERROR: weight/cache ignore policy failed:\n")
        for item in violations:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: weight and cache paths are gitignored and untracked\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
