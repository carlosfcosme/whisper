#!/usr/bin/env python3
"""CI assertion: serve/demo binds 127.0.0.1 only.

Fails if application start/serve sources contain an all-interface address
or an empty --host. Stdlib only. No network. No secrets. No weights.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))

# Application serve/listen paths. Tests may mention the rejected token.
SCAN_DIRS = (ROOT / "whisper", ROOT / ".cursor")
SKIP_NAMES = {"__pycache__"}

EMPTY_HOST = re.compile(r"""--host(?:\s+|=)(?:''|\"\"|\s*$)""")


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
            if any(part in SKIP_NAMES or part == ".git" for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def check_no_all_interfaces() -> None:
    hits = []
    for path in iter_app_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if ALL_INTERFACES in text:
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        fail(f"{ALL_INTERFACES} is not allowed in serve/listen paths: {hits}")


def check_no_empty_host() -> None:
    hits = []
    for path in iter_app_files():
        if path.suffix not in {".sh", ".py", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if EMPTY_HOST.search(text):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        fail(f"empty --host is not allowed: {hits}")


def check_start_script() -> None:
    start = ROOT / ".cursor" / "start.sh"
    if not start.is_file():
        fail("missing .cursor/start.sh")
    text = start.read_text(encoding="utf-8")
    if LOOPBACK not in text:
        fail(".cursor/start.sh must bind 127.0.0.1")
    if ALL_INTERFACES in text:
        fail(".cursor/start.sh must not bind all interfaces")
    if "--host" in text and EMPTY_HOST.search(text):
        fail(".cursor/start.sh must not pass an empty --host")


def check_bind_module() -> None:
    bind = ROOT / "whisper" / "bind.py"
    if not bind.is_file():
        fail("missing whisper/bind.py")
    text = bind.read_text(encoding="utf-8")
    if "bind host is required" not in text:
        fail("whisper/bind.py must refuse an empty host")
    if LOOPBACK not in text:
        fail("whisper/bind.py must name 127.0.0.1")


def main() -> int:
    check_no_all_interfaces()
    check_no_empty_host()
    check_start_script()
    check_bind_module()
    print("bind-localhost: ok (127.0.0.1 only; empty host refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
