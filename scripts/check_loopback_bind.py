#!/usr/bin/env python3
"""CI guard: listeners must bind 127.0.0.1 only.

Loads whisper.serve / whisper.runtime by path (stdlib only). No torch,
no Hub, no weight download.
"""

import importlib.util
import re
import sys
import types
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

# Built without a literal all-interfaces token in this file's app-scan trees.
ALL_INTERFACES = ".".join(("0",) * 4)
INADDR_ANY = "INADDR_ANY"

BIND_CALL = re.compile(
    r"(?:HTTPServer|ThreadingHTTPServer|TCPServer|UDPServer|bind|listen)"
    r"""\s*\(\s*\(\s*['\"]""" + re.escape(ALL_INTERFACES)
)

EMPTY_BIND = re.compile(
    r"(?:HTTPServer|ThreadingHTTPServer|TCPServer|UDPServer|bind|listen)"
    r"""\s*\(\s*\(\s*['\"]['\"]"""
)

NON_LOOPBACK_HOSTS = (
    ALL_INTERFACES,
    "::",
    "[::]",
    "*",
    "",
    "192.168.1.10",
    "10.0.0.1",
    "172.16.0.1",
    "8.8.8.8",
    "example.com",
    "::1",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_serve(root: Path):
    """Load serve.py without importing whisper/__init__.py (avoids torch)."""
    whisper_dir = root / "whisper"
    pkg_name = "whisper_offline_ci"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(whisper_dir)]
    sys.modules[pkg_name] = pkg

    runtime_spec = importlib.util.spec_from_file_location(
        pkg_name + ".runtime", whisper_dir / "runtime.py"
    )
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[pkg_name + ".runtime"] = runtime
    assert runtime_spec.loader is not None
    runtime_spec.loader.exec_module(runtime)

    serve_spec = importlib.util.spec_from_file_location(
        pkg_name + ".serve", whisper_dir / "serve.py"
    )
    serve = importlib.util.module_from_spec(serve_spec)
    sys.modules[pkg_name + ".serve"] = serve
    assert serve_spec.loader is not None
    serve_spec.loader.exec_module(serve)
    return serve


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
    hits: List[Tuple[str, str]] = []
    for path in iter_app_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(root)).replace("\\", "/")
        if ALL_INTERFACES in text:
            hits.append((rel, ALL_INTERFACES))
        if INADDR_ANY in text or INADDR_ANY.lower() in text:
            hits.append((rel, INADDR_ANY))
        if BIND_CALL.search(text):
            hits.append((rel, "all-interfaces bind()"))
        if EMPTY_BIND.search(text):
            hits.append((rel, "empty-host bind()"))
    return hits


def check_policy(serve) -> List[str]:
    errors: List[str] = []
    if serve.DEFAULT_HOST != "127.0.0.1":
        errors.append("DEFAULT_HOST is not 127.0.0.1")
    if serve.normalize_bind_host("127.0.0.1") != "127.0.0.1":
        errors.append("127.0.0.1 was not accepted")
    if serve.normalize_bind_host("localhost") != "127.0.0.1":
        errors.append("localhost was not rewritten to 127.0.0.1")
    for host in NON_LOOPBACK_HOSTS:
        try:
            serve.normalize_bind_host(host)
        except serve.BindError:
            continue
        errors.append("accepted non-loopback bind host {!r}".format(host))
    return errors


def main() -> int:
    root = repo_root()
    hits = scan_app_sources(root)
    serve = load_serve(root)
    errors = check_policy(serve)
    if hits or errors:
        sys.stderr.write("ERROR: bind policy must be 127.0.0.1 only:\n")
        for rel, token in hits:
            sys.stderr.write("  {}: {}\n".format(rel, token))
        for err in errors:
            sys.stderr.write("  {}\n".format(err))
        return 1
    sys.stdout.write("OK: bind policy is 127.0.0.1 only\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
