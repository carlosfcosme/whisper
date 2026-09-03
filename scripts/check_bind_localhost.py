#!/usr/bin/env python3
"""Fail CI if serve/bind uses 0.0.0.0 instead of 127.0.0.1.

Loads whisper/bind.py as a standalone file (no torch). Does not download
weights and does not start a wildcard listener.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
WILDCARD = "0.0.0.0"
BIND_MODULE = ROOT / "whisper" / "bind.py"
SKIP_SCAN = {Path(__file__).name}


def _load_bind():
    spec = importlib.util.spec_from_file_location("whisper_bind_ci", BIND_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _func_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_wildcard_host_tuple(node: ast.AST) -> bool:
    if not isinstance(node, ast.Tuple) or not node.elts:
        return False
    first = node.elts[0]
    return isinstance(first, ast.Constant) and first.value == WILDCARD


def find_wildcard_binds(root: Path) -> List[str]:
    """Return violations for HTTPServer/bind((0.0.0.0, ...)) in package code."""
    violations: List[str] = []
    scan_roots = [root / "whisper", root / "scripts"]
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*.py"):
            if path.name in SKIP_SCAN:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _func_name(node)
                if name not in {
                    "HTTPServer",
                    "ThreadingHTTPServer",
                    "TCPServer",
                    "bind",
                }:
                    continue
                if node.args and _is_wildcard_host_tuple(node.args[0]):
                    violations.append(
                        '{0}:{1}: {2}(("{3}", ...))'.format(
                            rel, node.lineno, name, WILDCARD
                        )
                    )
    return violations


def find_default_host_errors(root: Path) -> List[str]:
    errors: List[str] = []
    bind_src = (root / "whisper" / "bind.py").read_text(encoding="utf-8")
    serve_src = (root / "whisper" / "serve.py").read_text(encoding="utf-8")
    defaults_src = (root / "whisper" / "defaults.py").read_text(encoding="utf-8")
    if 'BIND_HOST = "127.0.0.1"' not in bind_src:
        errors.append("whisper/bind.py BIND_HOST is not 127.0.0.1")
    if "DEFAULT_HOST = BIND_HOST" not in serve_src and (
        'DEFAULT_HOST = "127.0.0.1"' not in serve_src
    ):
        errors.append("whisper/serve.py DEFAULT_HOST is not 127.0.0.1")
    if 'BIND_HOST = "127.0.0.1"' not in defaults_src:
        errors.append("whisper/defaults.py BIND_HOST is not 127.0.0.1")
    if 'DEFAULT_DEVICE = "cpu"' not in defaults_src:
        errors.append("whisper/defaults.py DEFAULT_DEVICE is not cpu")
    return errors


def main() -> int:
    bind = _load_bind()
    errors: List[str] = []
    if getattr(bind, "BIND_HOST", None) != LOOPBACK:
        errors.append("bind.BIND_HOST is not 127.0.0.1")
    try:
        bind.require_bind_host(WILDCARD)
        errors.append("require_bind_host accepted 0.0.0.0")
    except bind.BindError:
        pass
    try:
        accepted = bind.require_bind_host(LOOPBACK)
        if accepted != LOOPBACK:
            errors.append("require_bind_host did not return 127.0.0.1")
    except bind.BindError as exc:
        errors.append("require_bind_host rejected 127.0.0.1: {0}".format(exc))
    errors.extend(find_default_host_errors(ROOT))
    errors.extend(find_wildcard_binds(ROOT))
    if errors:
        sys.stderr.write("error: bind/CI policy failed:\n")
        for item in errors:
            sys.stderr.write("  {0}\n".format(item))
        return 1
    sys.stdout.write("OK: bind is 127.0.0.1; 0.0.0.0 is refused\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
