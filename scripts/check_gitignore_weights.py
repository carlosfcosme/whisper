#!/usr/bin/env python3
"""Fail CI if cache/weight paths can be committed.

Requires .gitignore to list cache and checkpoint patterns, and fails
if git already tracks those paths. Stdlib only: no torch, no Hub, no
weight download, no credentials.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.ckpt",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.ckpt",
)

TRACKED_GLOBS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
)


def gitignore_lines(text: str) -> set:
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def missing_gitignore_patterns(text: str) -> List[str]:
    present = gitignore_lines(text)
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in present]


def _git(args: Sequence[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def tracked_weight_paths(cwd: Path = ROOT) -> List[str]:
    proc = _git(["ls-files", "-z", "--", *TRACKED_GLOBS], cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "git ls-files failed")
    return [path for path in proc.stdout.split("\0") if path]


def unignored_examples(
    cwd: Path = ROOT, examples: Iterable[str] = IGNORE_EXAMPLES
) -> List[str]:
    failed = []
    for path in examples:
        proc = _git(["check-ignore", "-q", "--", path], cwd=cwd)
        if proc.returncode != 0:
            failed.append(path)
    return failed


def main() -> int:
    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        print("missing .gitignore", file=sys.stderr)
        return 1
    text = gitignore.read_text(encoding="utf-8")
    missing = missing_gitignore_patterns(text)
    if missing:
        print(
            "required cache/weight patterns missing from .gitignore:", file=sys.stderr
        )
        for pattern in missing:
            print("  %s" % pattern, file=sys.stderr)
        return 1
    tracked = tracked_weight_paths()
    if tracked:
        print("cache/weight paths must stay untracked:", file=sys.stderr)
        for path in tracked:
            print("  %s" % path, file=sys.stderr)
        return 1
    failed = unignored_examples()
    if failed:
        print("expected these paths to be gitignored:", file=sys.stderr)
        for path in failed:
            print("  %s" % path, file=sys.stderr)
        return 1
    print("ok: cache/weight paths are gitignored and untracked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
