#!/usr/bin/env python3
"""Fail if start/serve paths bind all interfaces.

Scans start scripts and the serve/bind modules. Test files and this checker
may mention the wildcard as a needle; they are not scanned.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"
# --host that is not 127.0.0.1, plus classic all-interfaces names.
NON_LOOPBACK_RE = re.compile(
    r"0\.0\.0\.0|INADDR_ANY|IN6ADDR_ANY|" r"--host(?:=|\s+)(?!127\.0\.0\.1(?:\s|$))\S+"
)
RELATIVE_PATHS = (
    ".cursor/start.sh",
    ".cursor/environment.json",
    "whisper/serve.py",
    "whisper/bind.py",
)


def _scan_paths() -> List[Path]:
    paths = [REPO_ROOT / rel for rel in RELATIVE_PATHS]
    paths.extend(sorted(REPO_ROOT.glob("start*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("serve*.sh")))
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def reasons_wildcard_bind(
    file_texts: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return reasons a start/serve path would listen on all interfaces."""
    reasons: List[str] = []
    if file_texts is None:
        file_texts = {
            str(path.relative_to(REPO_ROOT)): path.read_text() for path in _scan_paths()
        }
    if not file_texts:
        reasons.append("no start/serve files found to scan")
        return reasons
    for label, text in file_texts.items():
        if WILDCARD in text:
            reasons.append(f"{label} contains {WILDCARD} (must bind 127.0.0.1 only)")
        elif NON_LOOPBACK_RE.search(text):
            reasons.append(
                f"{label} names a non-loopback bind host (must bind 127.0.0.1 only)"
            )
    start = file_texts.get(".cursor/start.sh")
    if start is not None:
        if "127.0.0.1" not in start:
            reasons.append(".cursor/start.sh must bind 127.0.0.1")
        if "whisper.serve" not in start:
            reasons.append(".cursor/start.sh must invoke whisper.serve")
    return reasons


def main() -> int:
    reasons = reasons_wildcard_bind()
    if reasons:
        print("FAIL: start/serve path is not 127.0.0.1-only:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print("OK: start/serve paths bind 127.0.0.1 only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
