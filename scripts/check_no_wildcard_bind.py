#!/usr/bin/env python3
"""Fail CI if production code/config binds to a wildcard host.

Scans application and config paths for forbidden listen patterns
(unspecified IPv4 and ``--host`` wildcard). Tests may mention those
patterns when asserting they are rejected. This script does not fetch
Hub artifacts or model weights.

``--probe-negative`` plants a wildcard host in a temp tree and loads the
loopback policy; the job fails if either guard would miss the violation.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Patterns that mean "listen on all interfaces".
FORBIDDEN_SUBSTRINGS = (
    "0.0.0.0",
    "--host 0.0.0.0",
    "--host=0.0.0.0",
)

SCAN_ROOTS = (
    "whisper",
    ".github",
    ".cursor",
    "scripts",
    "pyproject.toml",
)

SKIP_NAMES = frozenset(
    {
        "scripts/check_no_wildcard_bind.py",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def _under_scan_root(relpath: str) -> bool:
    posix = relpath.replace("\\", "/")
    if posix in SKIP_NAMES:
        return False
    for prefix in SCAN_ROOTS:
        if posix == prefix or posix.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        if not _under_scan_root(relpath):
            continue
        path = root / relpath
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_SUBSTRINGS:
            if pattern in text:
                violations.append((relpath, pattern))
    return violations


def _load_localhost(root: Path):
    path = root / "whisper" / "localhost.py"
    spec = importlib.util.spec_from_file_location("whisper_localhost_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def probe_negative(root: Optional[Path] = None) -> int:
    """Return 0 if planted wildcards and non-loopback hosts are refused."""
    wildcard = FORBIDDEN_SUBSTRINGS[0]
    host_flag = FORBIDDEN_SUBSTRINGS[1]
    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp)
        (planted / "whisper").mkdir()
        (planted / "scripts").mkdir()
        (planted / "whisper" / "bad.py").write_text('host="{}"\n'.format(wildcard))
        (planted / "scripts" / "cli.py").write_text(host_flag + "\n")
        hits = find_violations(
            planted, relative_paths=["whisper/bad.py", "scripts/cli.py"]
        )
        patterns = {pattern for _path, pattern in hits}
        missing = [token for token in (wildcard, host_flag) if token not in patterns]
        if missing:
            sys.stderr.write(
                "ERROR: bind checker missed planted wildcard token(s): {}\n".format(
                    ", ".join(missing)
                )
            )
            return 1

    policy_root = root if root is not None else repo_root()
    loc = _load_localhost(policy_root)
    if loc.serve_bind_host() != "127.0.0.1":
        sys.stderr.write("ERROR: default bind host is not 127.0.0.1\n")
        return 1
    if loc.serve_bind_host("127.0.0.1") != "127.0.0.1":
        sys.stderr.write("ERROR: 127.0.0.1 was not accepted\n")
        return 1
    refused = (wildcard, "::", "", "   ", "192.168.1.1", "example.com", "*")
    for host in refused:
        try:
            loc.serve_bind_host(host)
        except ValueError:
            continue
        sys.stderr.write("ERROR: bind policy accepted non-loopback {!r}\n".format(host))
        return 1
    sys.stdout.write(
        "OK: negative probes flagged planted wildcard bind and non-loopback hosts\n"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if production code binds to a wildcard host"
    )
    parser.add_argument(
        "--probe-negative",
        action="store_true",
        help="plant a wildcard bind and assert the guards fail it",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.probe_negative:
        return probe_negative()
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write(
            "ERROR: forbidden wildcard bind pattern in production code/config:\n"
        )
        for relpath, pattern in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, pattern))
        sys.stderr.write("Listen only on 127.0.0.1 (never wildcard / empty host).\n")
        return 1
    sys.stdout.write("OK: no wildcard bind patterns in production code/config\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
