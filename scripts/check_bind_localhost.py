#!/usr/bin/env python3
"""CI assertion: serve/demo binds 127.0.0.1 only.

Fails if application start/serve sources contain an all-interface address
or an empty --host. Stdlib only. No network. No secrets. No weights.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))

SCAN_DIRS = (ROOT / "whisper", ROOT / ".cursor")
SKIP_NAMES = {"__pycache__"}

EMPTY_HOST = re.compile(r"""--host(?:\s+|=)(?:''|\"\"|\s*$)""")


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_app_files(root: Path = ROOT) -> Iterable[Path]:
    for rel in ("whisper", ".cursor"):
        base = root / rel
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


def find_all_interface_hits(root: Path = ROOT) -> List[str]:
    hits = []
    for path in iter_app_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if ALL_INTERFACES in text:
            hits.append(str(path.relative_to(root)))
    return hits


def find_empty_host_hits(root: Path = ROOT) -> List[str]:
    hits = []
    for path in iter_app_files(root):
        if path.suffix not in {".sh", ".py", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if EMPTY_HOST.search(text):
            hits.append(str(path.relative_to(root)))
    return hits


def load_bind_module():
    path = ROOT / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_ci", path)
    if spec is None or spec.loader is None:
        fail("could not load whisper/bind.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_no_all_interfaces(root: Path = ROOT) -> None:
    hits = find_all_interface_hits(root)
    if hits:
        fail(f"{ALL_INTERFACES} is not allowed in serve/listen paths: {hits}")


def check_no_empty_host(root: Path = ROOT) -> None:
    hits = find_empty_host_hits(root)
    if hits:
        fail(f"empty --host is not allowed: {hits}")


def check_start_script(root: Path = ROOT) -> None:
    start = root / ".cursor" / "start.sh"
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
    bind = load_bind_module()
    if bind.require_bind_127_0_0_1(None) != LOOPBACK:
        fail("whisper/bind.py must default to 127.0.0.1")
    try:
        bind.require_bind_127_0_0_1(ALL_INTERFACES)
    except bind.BindError:
        pass
    else:
        fail("whisper/bind.py must refuse an all-interfaces host")
    try:
        bind.require_bind_127_0_0_1("")
    except bind.BindError:
        pass
    else:
        fail("whisper/bind.py must refuse an empty host")


def main() -> int:
    check_no_all_interfaces()
    check_no_empty_host()
    check_start_script()
    check_bind_module()
    print("bind-localhost: ok (127.0.0.1 only; all-interfaces refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
