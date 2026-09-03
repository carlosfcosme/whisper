#!/usr/bin/env python3
"""CI guard: bind/listen hosts must be 127.0.0.1.

Fails if application sources contain all-interface / empty-bind literals,
or if whisper.bind.require_loopback_host accepts a non-loopback host.

Loads whisper/bind.py by path (stdlib only). No torch, no Hub, no weights,
no keys, no Field-Brain.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import List, Tuple

APP_TREES = ("whisper", ".cursor")
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
}

# Script may name these tokens; application trees must not contain them.
FORBIDDEN_SUBSTRINGS = (
    "0.0.0.0",
    "INADDR_ANY",
    "inaddr_any",
)

EMPTY_BIND = re.compile(
    r"(?:HTTPServer|ThreadingHTTPServer|TCPServer|UDPServer|bind|listen)"
    r"""\s*\(\s*\(\s*['\"]['\"]"""
)

# Hosts that must raise BindError. The all-interfaces IPv4 token is built
# without a second copy of the grep needle in whisper/.
NON_LOOPBACK_HOSTS = (
    ".".join(("0",) * 4),
    "::",
    "[::]",
    "*",
    "",
    "0",
    "192.168.1.1",
    "10.0.0.1",
    "172.16.0.1",
    "8.8.8.8",
    "255.255.255.255",
    "example.com",
    "::1",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_bind(root: Path):
    path = root / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_app_files(root: Path):
    for rel in APP_TREES:
        tree = root / rel
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            yield path


def scan_app_sources(root: Path) -> List[Tuple[str, str]]:
    hits = []
    for path in iter_app_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(root))
        for token in FORBIDDEN_SUBSTRINGS:
            if token in text:
                hits.append((rel, token))
        if EMPTY_BIND.search(text):
            hits.append((rel, "empty-host bind()"))
    return hits


def check_policy(root: Path) -> List[str]:
    errors = []
    bind = load_bind(root)
    if bind.require_loopback_host() != "127.0.0.1":
        errors.append("default bind host is not 127.0.0.1")
    if bind.require_loopback_host("127.0.0.1") != "127.0.0.1":
        errors.append("127.0.0.1 was not accepted")
    if bind.LOOPBACK_HOST != "127.0.0.1":
        errors.append("LOOPBACK_HOST is not 127.0.0.1")
    for host in NON_LOOPBACK_HOSTS:
        try:
            bind.require_loopback_host(host)
        except bind.BindError:
            continue
        errors.append("accepted non-loopback bind host {!r}".format(host))
    return errors


def main() -> int:
    root = repo_root()
    hits = scan_app_sources(root)
    policy_errors = check_policy(root)
    if hits or policy_errors:
        sys.stderr.write("ERROR: bind host must be 127.0.0.1 (loopback only)\n")
        for rel, token in hits:
            sys.stderr.write("  {}: forbidden token {}\n".format(rel, token))
        for message in policy_errors:
            sys.stderr.write("  policy: {}\n".format(message))
        return 1
    sys.stdout.write("OK: loopback bind policy and source guards passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
