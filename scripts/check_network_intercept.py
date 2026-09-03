#!/usr/bin/env python3
"""Fail if tests do not intercept sockets (Python-level, not BPF)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
CONFTEST = REPO / "tests" / "conftest.py"
BPF_MODULES = frozenset({"bcc", "libbpf", "bpf", "ebpf", "pybpf"})

REQUIRED_SNIPPETS = (
    "socket.socket.connect",
    "socket.create_connection",
    "socket.getaddrinfo",
    "socket.socket.bind",
    "is_hub_host",
    "is_loopback_host",
    "HF_HUB_OFFLINE",
)


def _imported_roots(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


def bpf_imports() -> List[str]:
    hits: List[str] = []
    roots = (REPO / "whisper", REPO / "tests", REPO / "scripts")
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            imported = set(_imported_roots(path)) & BPF_MODULES
            if imported:
                hits.append("{}: {}".format(path.relative_to(REPO), sorted(imported)))
    return hits


def main() -> int:
    text = CONFTEST.read_text(encoding="utf-8")
    missing = [snip for snip in REQUIRED_SNIPPETS if snip not in text]
    bpf_hits = bpf_imports()
    if missing or bpf_hits:
        if missing:
            sys.stderr.write(
                "ERROR: tests/conftest.py must intercept sockets; missing:\n"
            )
            for snip in missing:
                sys.stderr.write("  {}\n".format(snip))
        if bpf_hits:
            sys.stderr.write("ERROR: network intercept must not import BPF:\n")
            for line in bpf_hits:
                sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: tests intercept sockets in Python (no BPF)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
