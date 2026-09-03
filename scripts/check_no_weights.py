#!/usr/bin/env python3
"""Fail if model weights, caches, or secrets are tracked in git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
        ".h5",
        ".hdf5",
        ".tflite",
        ".pb",
        ".mar",
        ".params",
        ".weights",
    }
)

CACHE_PREFIXES = (
    ".cache/",
    "cache/",
    "weights/",
    ".huggingface/",
    "huggingface/hub/",
)

SECRET_PREFIXES = (".env",)

IGNORE_SAMPLES = (
    "tiny.pt",
    "weights/tiny.pt",
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "model.pth",
    ".huggingface/hub/models--openai--whisper/tiny.pt",
    ".env",
    ".env.local",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str) -> Optional[str]:
    posix = relpath.replace("\\", "/")
    suffix = Path(posix).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    for prefix in CACHE_PREFIXES:
        if posix == prefix.rstrip("/") or posix.startswith(prefix):
            return "cache path ({})".format(prefix.rstrip("/"))
    for prefix in SECRET_PREFIXES:
        name = Path(posix).name
        if name == ".env" or name.startswith(".env."):
            return "secret file"
    return None


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        reason = classify(relpath)
        if reason:
            violations.append((relpath, reason))
    return violations


def assert_gitignore(root: Path) -> List[str]:
    missing = []
    for sample in IGNORE_SAMPLES:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", sample],
            cwd=root,
        )
        if result.returncode != 0:
            missing.append(sample)
    return missing


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    missing_ignore = assert_gitignore(root)
    if violations or missing_ignore:
        if violations:
            sys.stderr.write(
                "ERROR: model weights, caches, or secrets must not be committed:\n"
            )
            for relpath, reason in violations:
                sys.stderr.write("  {} — {}\n".format(relpath, reason))
        if missing_ignore:
            sys.stderr.write("ERROR: these paths must be gitignored:\n")
            for sample in missing_ignore:
                sys.stderr.write("  {}\n".format(sample))
        return 1
    sys.stdout.write("OK: no committed weights, caches, or secrets\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
