#!/usr/bin/env python3
"""Fail CI if cache/weight artifacts are tracked in git.

Stdlib only. Does not download Hub or CDN weights and does not read secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    ".huggingface/",
    ".torch/",
    "weights/",
    "checkpoints/",
    "*.pt",
    "*.pth",
    "*.pth.tar",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
    "*.ggml",
)

CACHE_PREFIXES = (
    ".cache/",
    "cache/",
    ".huggingface/",
    ".torch/",
    "weights/",
    "checkpoints/",
)

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".pth.tar",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".ggml",
)

LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    ".huggingface",
    ".huggingface/**",
    ".torch",
    ".torch/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    "*.pt",
    "*.pth",
    "*.pth.tar",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
    "*.ggml",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/base.pt",
    "weights/model.pth",
    "checkpoints/epoch.ckpt",
    ".huggingface/hub/models--x/snapshot.bin",
    "tiny.pt",
    "model.pth",
    "model.pth.tar",
    "model.safetensors",
    "model.onnx",
    "model.gguf",
    "model.ggml",
)


def repo_root() -> Path:
    output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(output.strip())


def gitignore_patterns(root: Path) -> List[str]:
    path = root / ".gitignore"
    if not path.is_file():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def missing_gitignore_patterns(root: Path) -> List[str]:
    present = set(gitignore_patterns(root))
    return [pattern for pattern in REQUIRED_GITIGNORE if pattern not in present]


def tracked_files(root: Path, pathspecs: Sequence[str] = ()) -> List[str]:
    cmd = ["git", "ls-files", "-z"]
    if pathspecs:
        cmd.extend(["--", *pathspecs])
    output = subprocess.check_output(cmd, cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def is_weight_artifact(relpath: str) -> bool:
    posix = relpath.replace("\\", "/")
    if posix.startswith(CACHE_PREFIXES):
        return True
    lower = posix.lower()
    return any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def find_tracked_weight_artifacts(root: Path) -> List[str]:
    by_pathspec = set(tracked_files(root, LS_FILES_PATHSPECS))
    by_scan = {path for path in tracked_files(root) if is_weight_artifact(path)}
    return sorted(by_pathspec | by_scan)


def ignore_examples_not_ignored(root: Path) -> List[str]:
    failed = []
    for example in IGNORE_EXAMPLES:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", example],
            cwd=root,
        )
        if result.returncode != 0:
            failed.append(example)
    return failed


def collect_errors(root: Path) -> List[str]:
    errors: List[str] = []
    missing = missing_gitignore_patterns(root)
    if missing:
        errors.append("gitignore missing: " + ", ".join(missing))
    unignored = ignore_examples_not_ignored(root)
    if unignored:
        errors.append("not gitignored: " + ", ".join(unignored))
    tracked = find_tracked_weight_artifacts(root)
    if tracked:
        errors.append("tracked weight artifacts: " + ", ".join(tracked))
    return errors


def main(argv: Iterable[str] = ()) -> int:
    del argv
    root = repo_root()
    errors = collect_errors(root)
    if errors:
        sys.stderr.write("error: cache/weight artifacts must not be committed:\n")
        for item in errors:
            sys.stderr.write("  {0}\n".format(item))
        sys.stderr.write("Remove with: git rm --cached -- <path>\n")
        return 1
    sys.stdout.write("OK: no tracked cache or weight artifacts\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
