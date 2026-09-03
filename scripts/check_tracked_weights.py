#!/usr/bin/env python3
"""Fail if ``git ls-files`` lists weight or cache artifacts.

Does not download checkpoints, contact the Hub, or read secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# Pathspecs passed to ``git ls-files``. Directory names catch a tracked
# cache tree; globs catch checkpoint suffixes anywhere in the tree.
LS_FILES_PATHSPECS: Tuple[str, ...] = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    ".huggingface",
    ".huggingface/**",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.ggml",
    "*.gguf",
    "*.bin",
    "*.weights",
)

# Paths that ``git check-ignore`` must reject (gitignore coverage).
IGNORE_SAMPLES: Tuple[str, ...] = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.onnx",
    "model.bin",
    ".cache/huggingface/hub/models--openai--whisper/snapshots/x/pytorch_model.bin",
    ".huggingface/hub/models--openai--whisper/snapshots/x/model.safetensors",
)

REQUIRED_GITIGNORE: Tuple[str, ...] = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.bin",
    "*.weights",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gitignore_text(root: Path) -> str:
    return (root / ".gitignore").read_text(encoding="utf-8")


def missing_gitignore_patterns(root: Path) -> List[str]:
    text = gitignore_text(root)
    return [pattern for pattern in REQUIRED_GITIGNORE if pattern not in text]


def tracked_weight_paths(
    root: Path, pathspecs: Sequence[str] = LS_FILES_PATHSPECS
) -> List[str]:
    """Return tracked paths that match weight/cache pathspecs (``git ls-files``)."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--"] + list(pathspecs),
        cwd=root,
    )
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def is_ignored(root: Path, relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=root,
    )
    return result.returncode == 0


def unignored_samples(root: Path, samples: Sequence[str] = IGNORE_SAMPLES) -> List[str]:
    return [path for path in samples if not is_ignored(root, path)]


def main() -> int:
    root = repo_root()
    errors: List[str] = []

    missing = missing_gitignore_patterns(root)
    if missing:
        errors.append("gitignore missing patterns: {}".format(", ".join(missing)))

    leaked = unignored_samples(root)
    if leaked:
        errors.append("git check-ignore missed: {}".format(", ".join(leaked)))

    tracked = tracked_weight_paths(root)
    if tracked:
        errors.append("git ls-files listed weight/cache paths:")
        errors.extend("  {}".format(path) for path in tracked)

    if errors:
        sys.stderr.write("ERROR: weight/cache artifacts must not be committed:\n")
        for line in errors:
            sys.stderr.write("{}\n".format(line))
        return 1

    sys.stdout.write("OK: git ls-files has no weight/cache artifacts\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
