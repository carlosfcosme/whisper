#!/usr/bin/env python3
"""CI: .gitignore must ignore model weight suffixes."""

import sys
from pathlib import Path

REQUIRED_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.gguf",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def missing_patterns(root: Path):
    path = root / ".gitignore"
    if not path.is_file():
        return list(REQUIRED_PATTERNS)
    text = path.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    return [pattern for pattern in REQUIRED_PATTERNS if pattern not in lines]


def main() -> int:
    root = repo_root()
    missing = missing_patterns(root)
    if missing:
        sys.stderr.write("ERROR: .gitignore must ignore weight suffixes:\n")
        for pattern in missing:
            sys.stderr.write("  missing {}\n".format(pattern))
        return 1
    sys.stdout.write("OK: .gitignore covers model weight suffixes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
