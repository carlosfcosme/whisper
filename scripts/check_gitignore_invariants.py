#!/usr/bin/env python3
"""Fail if .gitignore does not keep weight/cache paths untracked."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "export/model.onnx",
)

LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "*.pt",
    "*.pth",
    "*.safetensors",
)

TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/mel_filters.npz",
    "tests/jfk.flac",
    ".gitignore",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def gitignore_lines(root: Path) -> set:
    return {
        line.strip()
        for line in (root / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def missing_patterns(root: Path) -> List[str]:
    lines = gitignore_lines(root)
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in lines]


def tracked_weight_paths(root: Path) -> List[str]:
    listed = _git(root, "ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    if listed.returncode != 0:
        raise RuntimeError(listed.stderr)
    return [path for path in listed.stdout.split("\0") if path]


def unignored_examples(root: Path) -> List[str]:
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git(root, "check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    return failed


def wrongly_ignored_assets(root: Path) -> List[str]:
    failed = []
    for path in TRACKED_ASSETS:
        result = _git(root, "check-ignore", "-q", "--", path)
        if result.returncode == 0:
            failed.append(path)
        elif not (root / path).is_file():
            failed.append(path)
    return failed


def find_violations(root: Path) -> List[Tuple[str, str]]:
    violations: List[Tuple[str, str]] = []
    for pattern in missing_patterns(root):
        violations.append((pattern, "required gitignore pattern missing"))
    for path in tracked_weight_paths(root):
        violations.append((path, "weight/cache path is tracked"))
    for path in unignored_examples(root):
        violations.append((path, "expected gitignore to match"))
    for path in wrongly_ignored_assets(root):
        violations.append((path, "tracked asset must not be gitignored"))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write("ERROR: gitignore weight/cache invariants failed:\n")
        for path, reason in violations:
            sys.stderr.write("  {}: {}\n".format(path, reason))
        return 1
    sys.stdout.write("OK: gitignore keeps weight/cache paths untracked\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
