#!/usr/bin/env python3
"""Fail if the package talks to a model hub or requires hub credentials."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "whisper"
DEPS = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
)

HUB_CODE_RE = re.compile(
    r"huggingface_hub|hf_hub_download|snapshot_download|"
    r"from_pretrained|hf://|HUGGING_FACE_HUB_TOKEN",
    re.I,
)
HUB_DEP_RE = re.compile(r"huggingface[_-]hub|transformers\s*[>=<]", re.I)

SKIP_NAMES = {"offline.py"}


def _iter_package_py() -> List[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.is_file())


def reasons_hub_in_tree() -> List[str]:
    reasons: List[str] = []
    for path in _iter_package_py():
        if path.name in SKIP_NAMES:
            continue
        text = path.read_text(encoding="utf-8")
        if HUB_CODE_RE.search(text):
            rel = path.relative_to(REPO_ROOT).as_posix()
            reasons.append(f"Hub client / fetch API in {rel}")
    for dep in DEPS:
        if not dep.is_file():
            continue
        text = dep.read_text(encoding="utf-8")
        if HUB_DEP_RE.search(text):
            reasons.append(f"Hub/transformers dependency in {dep.name}")
    return reasons


def main() -> int:
    reasons = reasons_hub_in_tree()
    if reasons:
        print("FAIL: package must not contact model hubs:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print("OK: no Hub client, no Hub dependency, no Hub credentials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
