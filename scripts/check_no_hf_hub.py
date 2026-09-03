#!/usr/bin/env python3
"""Fail if install, CI, or tests would fetch from the Hugging Face Hub."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Host / API tokens assembled so this file is the scanner, not a fetcher.
_HF_HOST = "huggingface" + ".co"
_HF_SHORT = "hf" + ".co"
_HF_LFS = "cdn-lfs." + _HF_HOST

DOWNLOAD_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"hf_hub_download\s*\(", "hf_hub_download()"),
    (r"snapshot_download\s*\(", "snapshot_download()"),
    (r"from_pretrained\s*\(", "from_pretrained()"),
    (re.escape("huggingface-cli download"), "huggingface-cli download"),
    (re.escape(_HF_LFS), "Hugging Face LFS host"),
)

# Install / workflow files must not contact the Hub at all.
INSTALL_PATTERNS: Sequence[Tuple[str, str]] = DOWNLOAD_PATTERNS + (
    (re.escape(_HF_HOST), "huggingface.co URL"),
    (re.escape(_HF_SHORT + "/"), "hf.co URL"),
)

SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "node_modules"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_files(root: Path, relative_dirs: Iterable[str]) -> List[Path]:
    found = []
    for rel in relative_dirs:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file():
            found.append(base)
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found.append(path)
    return found


def scan(paths: Iterable[Path], patterns: Sequence[Tuple[str, str]]) -> List[str]:
    hits = []
    compiled = [(re.compile(pat), label) for pat, label in patterns]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for regex, label in compiled:
            if regex.search(text):
                hits.append("{} ({})".format(path, label))
    return hits


def main() -> int:
    root = repo_root()
    test_hits = scan(iter_files(root, ("tests",)), DOWNLOAD_PATTERNS)
    install_hits = scan(
        iter_files(root, (".github", ".cursor")),
        INSTALL_PATTERNS,
    )
    hits = test_hits + install_hits
    if hits:
        sys.stderr.write(
            "ERROR: install/CI/tests must not fetch from Hugging Face Hub:\n"
        )
        for hit in hits:
            sys.stderr.write("  {}\n".format(hit))
        return 1
    sys.stdout.write("OK: no Hugging Face Hub fetches in install/CI/tests\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
