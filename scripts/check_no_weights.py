#!/usr/bin/env python3
"""CI assertion: no committed model weights, caches, or secrets.

Weights belong in a local cache, not git. Stdlib only. No network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
    }
)

CACHE_PREFIXES = (
    ".cache/",
    "cache/",
    "weights/",
    ".huggingface/",
)


def tracked_files() -> List[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    if not result.stdout:
        return []
    return [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]


def find_tracked_weights(paths: Iterable[str]) -> List[str]:
    hits = []
    for path in paths:
        lowered = path.lower()
        suffix = Path(path).suffix.lower()
        if suffix in WEIGHT_SUFFIXES:
            hits.append(path)
            continue
        if any(
            lowered == prefix.rstrip("/") or lowered.startswith(prefix)
            for prefix in CACHE_PREFIXES
        ):
            hits.append(path)
    return hits


def find_tracked_secrets(paths: Iterable[str]) -> List[str]:
    hits = []
    for path in paths:
        name = Path(path).name
        if name == ".env" or name.startswith(".env."):
            hits.append(path)
    return hits


def required_gitignore_tokens() -> Sequence[str]:
    return (".cache/", "cache/", "weights/", "*.pt")


def gitignore_missing_tokens(text: str) -> List[str]:
    missing = []
    for token in required_gitignore_tokens():
        if token not in text:
            missing.append(token)
    return missing


def main() -> int:
    paths = tracked_files()
    weights = find_tracked_weights(paths)
    secrets = find_tracked_secrets(paths)
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = gitignore_missing_tokens(gitignore)

    if weights:
        sys.stderr.write("ERROR: committed model weights or caches:\n")
        for path in weights:
            sys.stderr.write("  {}\n".format(path))
        return 1
    if secrets:
        sys.stderr.write("ERROR: committed secrets:\n")
        for path in secrets:
            sys.stderr.write("  {}\n".format(path))
        return 1
    if missing:
        sys.stderr.write("ERROR: .gitignore must ignore cache/weights:\n")
        for token in missing:
            sys.stderr.write("  {}\n".format(token))
        return 1
    sys.stdout.write("OK: no committed weights, caches, or secrets\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
