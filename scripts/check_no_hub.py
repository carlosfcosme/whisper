#!/usr/bin/env python3
"""CI assertion: download helpers stay unused; no Hub fetch, no secrets.

Scans application sources (whisper/, .cursor/, scripts/). Tests may name
the rejected helpers. Stdlib only. No network. No weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Call/import forms that would pull from the Hugging Face Hub.
DOWNLOAD_HELPERS = (
    "import huggingface_hub",
    "from huggingface_hub",
    "hf_hub_download(",
    "snapshot_download(",
    "hf_hub_url(",
    "cached_download(",
    "from_pretrained(",
)

# Ticket: no keys, no Field-Brain.
FORBIDDEN_SECRETS = (
    "Field-Brain",
    "FIELD_BRAIN",
    "API_KEY",
    "SECRET_KEY",
    "BEGIN RSA",
)

SCAN_DIRS = (ROOT / "whisper", ROOT / ".cursor", ROOT / "scripts")
SKIP_NAMES = {"__pycache__", ".git"}
# This file names the helpers in order to forbid them.
SKIP_FILES = {"check_no_hub.py"}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_app_files():
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.name in SKIP_FILES:
                continue
            if any(part in SKIP_NAMES for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def check_helpers_unused() -> None:
    hits = []
    for path in iter_app_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for token in DOWNLOAD_HELPERS:
            if token in text:
                hits.append("{}: {}".format(path.relative_to(ROOT), token))
    if hits:
        fail("download helpers must stay unused: {}".format(hits))


def check_no_secrets() -> None:
    hits = []
    for path in iter_app_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for token in FORBIDDEN_SECRETS:
            if token in text:
                hits.append("{}: {}".format(path.relative_to(ROOT), token))
    if hits:
        fail("secrets / Field-Brain tokens are forbidden: {}".format(hits))


def main() -> int:
    check_helpers_unused()
    check_no_secrets()
    print("no-hub: ok (download helpers unused; no secrets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
