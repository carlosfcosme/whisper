#!/usr/bin/env python3
"""Fail CI if .gitignore does not cover weight and cache paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REQUIRED_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
    ".cache/",
    ".cache/whisper/",
    ".cache/huggingface/",
    ".huggingface/",
    "hf_cache/",
    "whisper_cache/",
    "checkpoints/",
    "weights/",
    "cache/",
    ".env",
    "*.pem",
    "*.key",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    ".cache/huggingface/hub/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "hf_cache/tiny.safetensors",
    "whisper_cache/tiny.pt",
    "checkpoints/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    ".env",
    ".env.local",
    "secrets.pem",
    "id_rsa.key",
)

KEEP_TRACKED = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/mel_filters.npz",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gitignore_entries(root: Path) -> List[str]:
    path = root / ".gitignore"
    entries: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def missing_patterns(root: Path) -> List[str]:
    present = set(gitignore_entries(root))
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in present]


def unignored_examples(root: Path) -> List[str]:
    failed: List[str] = []
    for path in IGNORE_EXAMPLES:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            failed.append(path)
    return failed


def wrongly_ignored_assets(root: Path) -> List[str]:
    failed: List[str] = []
    for path in KEEP_TRACKED:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=root,
            check=False,
        )
        if result.returncode == 0:
            failed.append(path)
    return failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    root = repo_root()
    missing = missing_patterns(root)
    unignored = unignored_examples(root)
    leaked = wrongly_ignored_assets(root)
    if missing or unignored or leaked:
        if missing:
            sys.stderr.write(
                "ERROR: .gitignore must cover weight and cache paths; missing:\n"
            )
            for pattern in missing:
                sys.stderr.write("  {}\n".format(pattern))
        if unignored:
            sys.stderr.write("ERROR: git check-ignore must match cache/weight paths:\n")
            for path in unignored:
                sys.stderr.write("  {}\n".format(path))
        if leaked:
            sys.stderr.write("ERROR: tracked fixtures must not be gitignored:\n")
            for path in leaked:
                sys.stderr.write("  {}\n".format(path))
        return 1
    sys.stdout.write(
        "OK: .gitignore covers weight/cache paths; git check-ignore matches\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
