#!/usr/bin/env python3
"""Fail CI if package code would fetch from Hugging Face Hub."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

HUB_GREP = (
    r"hf_hub_download|snapshot_download|from huggingface_hub|import huggingface_hub"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def hub_hits(root: Path) -> str:
    result = subprocess.run(
        ["git", "grep", "-nE", HUB_GREP, "--", "whisper"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or "git grep failed")
    return result.stdout


def find_hub_api_uses(root: Path) -> List[str]:
    output = hub_hits(root)
    return [line for line in output.splitlines() if line.strip()]


def main() -> int:
    root = repo_root()
    hits = find_hub_api_uses(root)
    if hits:
        sys.stderr.write("ERROR: package code must not call Hugging Face Hub:\n")
        for line in hits:
            sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: no Hugging Face Hub fetch in whisper/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
