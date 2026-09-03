#!/usr/bin/env python3
"""CI guard: fail if weights are tracked or install scripts fetch them.

Localhost-only: also fail if start/install scripts bind 0.0.0.0.
Stdlib only — no torch, no Hub, no weight download.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".safetensors", ".ckpt", ".gguf", ".onnx"})
CACHE_MARKERS = (
    ".cache/whisper",
    "cache/whisper/",
    ".cache/huggingface",
    "weights/",
)
ALL_INTERFACES = "0.0.0.0"

INSTALL_NAMES = frozenset(
    {"install.sh", "setup.sh", "bootstrap.sh", "environment.json"}
)
START_NAMES = frozenset({"start.sh"})

WEIGHT_FETCH_RE = re.compile(r"""(?ix)
    (
        \b(curl|wget)\b
        .{0,240}
        (
            \.pt\b
            | \.pth\b
            | \.safetensors\b
            | \.gguf\b
            | azureedge\.net
            | huggingface\.co
            | hf\.co
            | whisper/models
        )
    )
    |
    (\b(load_model|whisper\.load_model)\s*\()
    """)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_ls_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def is_weight_or_cache_path(relpath: str) -> bool:
    posix = relpath.replace("\\", "/")
    suffix = Path(posix).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return True
    lowered = posix.lower()
    return any(marker in lowered for marker in CACHE_MARKERS)


def tracked_weight_hits(root: Path, paths: Optional[Sequence[str]] = None) -> List[str]:
    if paths is None:
        paths = git_ls_files(root)
    return [path for path in paths if is_weight_or_cache_path(path)]


def _uncommented_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = re.sub(r"\s+#.*$", "", stripped)
        if stripped:
            lines.append(stripped)
    return lines


def discover_install_scripts(root: Path) -> List[Path]:
    found = []
    cursor_install = root / ".cursor" / "install.sh"
    if cursor_install.is_file():
        found.append(cursor_install)
    env = root / ".cursor" / "environment.json"
    if env.is_file():
        found.append(env)
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name in INSTALL_NAMES or (
            path.name.startswith("install-") and path.name.endswith(".sh")
        ):
            found.append(path)
    return sorted(set(found))


def discover_start_scripts(root: Path) -> List[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name in START_NAMES or (
            path.name.startswith("start-") and path.name.endswith(".sh")
        ):
            found.append(path)
    env = root / ".cursor" / "environment.json"
    if env.is_file():
        found.append(env)
    return sorted(set(found))


def _script_text(path: Path) -> str:
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8")
        parts = [str(data.get("install", "")), str(data.get("start", ""))]
        return "\n".join(parts)
    return path.read_text(encoding="utf-8")


def install_weight_fetch_hits(paths: Iterable[Path]) -> List[str]:
    hits = []
    for path in paths:
        text = _script_text(path)
        body = "\n".join(_uncommented_lines(text))
        if WEIGHT_FETCH_RE.search(body):
            hits.append(str(path))
    return hits


def all_interface_hits(paths: Iterable[Path]) -> List[str]:
    hits = []
    for path in paths:
        if ALL_INTERFACES in _script_text(path):
            hits.append(str(path))
    return hits


def run_checks(root: Path) -> List[str]:
    errors: List[str] = []
    tracked = tracked_weight_hits(root)
    if tracked:
        errors.append("model/cache weight paths are tracked by git: {}".format(tracked))
    install_hits = install_weight_fetch_hits(discover_install_scripts(root))
    if install_hits:
        errors.append(
            "install scripts fetch weights by default: {}".format(install_hits)
        )
    bind_hits = all_interface_hits(
        discover_start_scripts(root) + discover_install_scripts(root)
    )
    if bind_hits:
        errors.append(
            "start/install scripts must bind 127.0.0.1 only (found {}): {}".format(
                ALL_INTERFACES, bind_hits
            )
        )
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root else repo_root()
    errors = run_checks(root)
    if errors:
        sys.stderr.write("ERROR: weight/localhost CI guard failed:\n")
        for err in errors:
            sys.stderr.write("  - {}\n".format(err))
        return 1
    sys.stdout.write(
        "OK: no tracked weights, install does not fetch weights, localhost-only\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
