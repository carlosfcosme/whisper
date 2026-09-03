#!/usr/bin/env python3
"""CI assertion: tests must not reference remote (WAN) asset URLs.

Local fixtures only. Loopback health checks are allowed. Stdlib only.
No network. No weights. No keys.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Full remote URL literal with a public host. Split strings ("https://" + host)
# used only as refused-Hub examples are not matches.
REMOTE_URL = re.compile(
    r"""https?://(?!127\.0\.0\.1\b)(?!localhost\b)(?!\[::1\])[A-Za-z0-9.-]+\.[A-Za-z]{2,}""",
    re.IGNORECASE,
)

SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}


def iter_test_files(root: Path = ROOT) -> Iterable[Path]:
    tests = root / "tests"
    if not tests.is_dir():
        return
    for path in tests.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".yml", ".yaml", ".sh", ".json"}:
            continue
        yield path


def find_remote_asset_urls(root: Path = ROOT) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    for path in iter_test_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in REMOTE_URL.finditer(text):
            hits.append((str(path.relative_to(root)), match.group(0)))
    return hits


def check_tiny_fixture(root: Path = ROOT) -> None:
    tiny = root / "tests" / "fixtures" / "tone.wav"
    if not tiny.is_file():
        print(f"FAIL: missing tiny local fixture {tiny}", file=sys.stderr)
        raise SystemExit(1)
    size = tiny.stat().st_size
    if size > 64 * 1024:
        print(f"FAIL: tiny fixture is not tiny: {size} bytes", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    hits = find_remote_asset_urls()
    if hits:
        print("ERROR: tests must not contain remote asset URLs:", file=sys.stderr)
        for rel, url in hits:
            print(f"  {rel}: {url}", file=sys.stderr)
        return 1
    check_tiny_fixture()
    print("no-wan-fixtures: ok (local fixtures only; tiny tone.wav present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
