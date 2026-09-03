#!/usr/bin/env python3
"""Fail CI if the install or test path would download weights or bind off-loopback.

Scans Cloud Agent install, GitHub Actions workflows, tests, and serve paths.
Library code may still download weights at user runtime; install and test must not.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Patterns that would pull model weights during install or CI/test.
# Constructed so this file does not contain raw download-host literals.
_AZURE = "openaipublic." + "azureedge.net"
_HF_LFS = "cdn-lfs." + "huggingface.co"
_HF_HOST = "huggingface." + "co/"
_HF_CLI = "huggingface-cli" + " download"

DOWNLOAD_PATTERNS: Sequence[Tuple[str, str]] = (
    (_AZURE, "Whisper Azure weight host"),
    (_HF_LFS, "Hugging Face LFS weight host"),
    (r"hf_hub_download\s*\(", "hf_hub_download()"),
    (r"snapshot_download\s*\(", "snapshot_download()"),
    (r"from_pretrained\s*\(", "from_pretrained()"),
    (re.escape(_HF_CLI), "huggingface-cli download"),
    (
        r"\bwget\b[^\n]*\.(?:pt|pth|bin|ckpt|safetensors|onnx|gguf)\b",
        "wget of a weight file",
    ),
    (
        r"\bcurl\b[^\n]*\.(?:pt|pth|bin|ckpt|safetensors|onnx|gguf)\b",
        "curl of a weight file",
    ),
)

INSTALL_ONLY_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"\bload_model\s*\(", "load_model() during install/CI"),
    (re.escape(_HF_HOST), "huggingface.co URL during install/CI"),
)

ALLOW_WEIGHT_URL_MENTIONS = frozenset(
    {
        "tests/test_safety_constraints.py",
        "tests/conftest.py",
    }
)

ALLOW_ZERO_ADDR_MENTIONS = frozenset(
    {
        "tests/test_safety_constraints.py",
        "tests/test_localhost_bind.py",
    }
)

SCAN_FILES = (
    Path(".cursor/install.sh"),
    Path(".cursor/start.sh"),
)

SCAN_DIRS = (
    Path(".github/workflows"),
    Path("tests"),
)

ZERO_ADDR = "0.0.0.0"
BIND_RX = re.compile(
    r"""(?:--host\s+{addr}|host\s*=\s*['\"]?{addr}|server_name\s*=\s*['\"]{addr}"""
    r"""|\(\s*['\"]{addr}['\"]|['\"]{addr}['\"])""".format(addr=re.escape(ZERO_ADDR))
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_scan_paths(root: Path) -> Iterable[Path]:
    for rel in SCAN_FILES:
        path = root / rel
        if path.is_file():
            yield path
    for rel in SCAN_DIRS:
        directory = root / rel
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yml", ".yaml", ".sh"}:
                yield path


def _is_install_or_workflow(relpath: str) -> bool:
    return relpath.startswith(".cursor/") or relpath.startswith(".github/workflows/")


def find_download_violations(root: Path) -> List[str]:
    compiled = [(re.compile(pat), label) for pat, label in DOWNLOAD_PATTERNS]
    install_compiled = [
        (re.compile(pat), label) for pat, label in INSTALL_ONLY_PATTERNS
    ]
    violations: List[str] = []
    for path in iter_scan_paths(root):
        relpath = _posix(path, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        patterns = list(compiled)
        if _is_install_or_workflow(relpath):
            patterns.extend(install_compiled)
        elif relpath in ALLOW_WEIGHT_URL_MENTIONS:
            patterns = [
                (rx, label)
                for rx, label in patterns
                if label
                not in {
                    "Whisper Azure weight host",
                    "Hugging Face LFS weight host",
                }
            ]
        for lineno, line in enumerate(text.splitlines(), 1):
            for rx, label in patterns:
                if rx.search(line):
                    violations.append(
                        "{}:{}: {} ({})".format(relpath, lineno, label, line.strip())
                    )
    return violations


def find_bind_violations(root: Path) -> List[str]:
    violations: List[str] = []
    for path in iter_scan_paths(root):
        relpath = _posix(path, root)
        if relpath in ALLOW_ZERO_ADDR_MENTIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if BIND_RX.search(line):
                violations.append(
                    "{}:{}: non-loopback bind ({})".format(
                        relpath, lineno, line.strip()
                    )
                )
    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    root = repo_root()
    downloads = find_download_violations(root)
    binds = find_bind_violations(root)
    failed = False
    if downloads:
        failed = True
        sys.stderr.write("ERROR: install/test path must not download model weights:\n")
        for item in downloads:
            sys.stderr.write("  {}\n".format(item))
    if binds:
        failed = True
        sys.stderr.write("ERROR: install/test/serve must bind to 127.0.0.1 only:\n")
        for item in binds:
            sys.stderr.write("  {}\n".format(item))
    if failed:
        return 1
    sys.stdout.write(
        "OK: install/test path does not download weights; binds stay on 127.0.0.1\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
