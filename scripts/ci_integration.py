#!/usr/bin/env python3
"""CI integration gate: committed weights, CPU default, 127.0.0.1 binds.

One entrypoint for GitHub Actions. Fails the job when any of these is true:

* model weights or large binaries are committed
* the default inference device is not CPU
* a serve/install/test path binds off 127.0.0.1
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = repo_root() / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_committed_weights(root: Path) -> List[str]:
    check = _load_script("check_no_weights")
    return [
        "committed weight/binary: {}: {}".format(relpath, reason)
        for relpath, reason in check.find_violations(root)
    ]


def check_default_device_cpu(root: Path) -> List[str]:
    errors: List[str] = []
    init = (root / "whisper" / "__init__.py").read_text(encoding="utf-8")
    if not re.search(r"^DEFAULT_DEVICE\s*=\s*[\"']cpu[\"']", init, re.M):
        errors.append("whisper.DEFAULT_DEVICE is not cpu")
    cli = (root / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    if "default=DEFAULT_DEVICE" not in cli:
        errors.append("CLI --device does not default to DEFAULT_DEVICE (cpu)")
    return errors


def check_loopback_only(root: Path) -> List[str]:
    errors: List[str] = []
    localhost = (root / "whisper" / "localhost.py").read_text(encoding="utf-8")
    if (
        'LOOPBACK_BIND = "127.0.0.1"' not in localhost
        and "LOOPBACK_BIND = '127.0.0.1'" not in localhost
    ):
        errors.append("LOOPBACK_BIND is not 127.0.0.1")
    start = root / ".cursor" / "start.sh"
    if start.is_file():
        text = start.read_text(encoding="utf-8")
        if "127.0.0.1" not in text:
            errors.append(".cursor/start.sh does not bind 127.0.0.1")
        if "0.0.0.0" in text:
            errors.append(".cursor/start.sh mentions 0.0.0.0")
    policy = _load_script("check_install_test_policy")
    errors.extend(policy.find_bind_violations(root))
    errors.extend(policy.find_download_violations(root))
    return errors


def collect_errors(root: Optional[Path] = None) -> List[str]:
    root = root if root is not None else repo_root()
    errors: List[str] = []
    errors.extend(check_committed_weights(root))
    errors.extend(check_default_device_cpu(root))
    errors.extend(check_loopback_only(root))
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    errors = collect_errors()
    if errors:
        sys.stderr.write("ERROR: CI integration failed:\n")
        for item in errors:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write(
        "OK: CI integration — no committed weights; default device cpu; 127.0.0.1 only\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
