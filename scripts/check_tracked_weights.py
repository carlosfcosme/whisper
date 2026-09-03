#!/usr/bin/env python3
"""Fail if git ls-files lists weight blobs or cache/weight directories.

Used by CI and pre-commit. Does not download models, contact the Hub, or
read secrets. Dummy bytes in a matching path are enough to fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".ggml",
    ".bin",
}

WEIGHT_DIR_NAMES = {
    ".cache",
    "cache",
    "weights",
    "checkpoints",
    ".huggingface",
    ".torch",
}

REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    ".huggingface/",
    ".torch/",
    "weights/",
    "checkpoints/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
    "*.ggml",
    "*.bin",
)

EXAMPLE_IGNORED_PATHS = (
    "tiny.pt",
    "models/base.pth",
    "archive.pth.tar",
    ".cache/whisper/tiny.pt",
    "cache/whisper/base.pt",
    "weights/model.safetensors",
    "checkpoints/epoch.ckpt",
    ".huggingface/hub/models--x/snapshots/y/pytorch_model.bin",
    ".torch/hub/checkpoints/x.pth",
    "export/model.onnx",
    "export/model.gguf",
    "export/model.ggml",
)


def is_weight_artifact(rel: str) -> bool:
    """True if *rel* is a weight blob or lives under a cache/weight directory."""
    path = Path(rel)
    name = path.name.lower()
    if name.endswith(".pth.tar"):
        return True
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return True
    return any(part in WEIGHT_DIR_NAMES for part in path.parts)


def git_ls_files(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"],
        text=True,
    )
    return [item for item in output.split("\0") if item]


def tracked_weight_artifacts(root: Path) -> list[str]:
    return [rel for rel in git_ls_files(root) if is_weight_artifact(rel)]


def gitignore_text(root: Path) -> str:
    return (root / ".gitignore").read_text()


def missing_gitignore_rules(root: Path) -> list[str]:
    text = gitignore_text(root)
    return [rule for rule in REQUIRED_GITIGNORE if rule not in text]


def path_is_ignored(root: Path, rel: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--", rel],
        check=False,
    )
    return result.returncode == 0


def unignored_examples(root: Path) -> list[str]:
    return [rel for rel in EXAMPLE_IGNORED_PATHS if not path_is_ignored(root, rel)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: this script's parent)",
    )
    args = parser.parse_args(argv)
    root = (
        Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    )

    errors: list[str] = []
    missing = missing_gitignore_rules(root)
    if missing:
        errors.append("gitignore missing rules: " + ", ".join(missing))

    leaked = unignored_examples(root)
    if leaked:
        errors.append("gitignore does not ignore: " + ", ".join(leaked))

    tracked = tracked_weight_artifacts(root)
    if tracked:
        errors.append("git ls-files listed weight/cache artifacts:")
        errors.extend(f"  - {rel}" for rel in tracked)

    if errors:
        print("check_tracked_weights failed:")
        for line in errors:
            print(line)
        return 1

    print(
        "check_tracked_weights passed: ignore rules present, "
        "examples ignored, git ls-files has no weight blobs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
