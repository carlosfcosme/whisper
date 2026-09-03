#!/usr/bin/env python3
"""CI tracked-file guard: cache/weight dirs and blobs must stay untracked.

Does not download checkpoints, contact the Hugging Face Hub, or read secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".ckpt",
        ".ggml",
        ".gguf",
        ".h5",
        ".hdf5",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".weights",
        ".bin",
    }
)

WEIGHT_DIR_PREFIXES = (".cache/", "cache/", "weights/", "checkpoints/")

REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    "weights/",
    "checkpoints/",
    "*.pt",
    "*.pth",
)

TRACKED_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.ggml",
    "*.gguf",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "checkpoints/tiny.pt",
    "tiny.pt",
    "model.pth",
)

# Official Whisper checkpoints start around 75 MiB. Existing fixtures stay
# under this limit (largest tracked file is ~5.7 MiB).
MAX_FILE_BYTES = 10 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def _posix(relpath: str) -> str:
    return relpath.replace("\\", "/").lstrip("./")


def classify(relpath: str, size: int) -> Optional[str]:
    """Return a violation reason, or None if the file is allowed."""
    posix = _posix(relpath)
    for prefix in WEIGHT_DIR_PREFIXES:
        name = prefix.rstrip("/")
        if posix == name or posix.startswith(prefix):
            return "cache/weight path ({})".format(prefix)
    suffix = Path(posix).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    if size > MAX_FILE_BYTES:
        return "large file ({} bytes > {})".format(size, MAX_FILE_BYTES)
    return None


def find_violations(
    root: Path, relative_paths: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    paths: Iterable[str] = (
        relative_paths if relative_paths is not None else tracked_files(root)
    )
    violations: List[Tuple[str, str]] = []
    for relpath in paths:
        path = root / relpath
        if not path.is_file():
            continue
        reason = classify(relpath, path.stat().st_size)
        if reason:
            violations.append((relpath, reason))
    return violations


def missing_gitignore_patterns(root: Path) -> List[str]:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return list(REQUIRED_GITIGNORE)
    lines = {line.strip() for line in gitignore.read_text().splitlines()}
    return [pat for pat in REQUIRED_GITIGNORE if pat not in lines]


def tracked_weight_paths(root: Path) -> List[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *TRACKED_PATHSPECS],
        cwd=root,
    )
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def unignored_examples(root: Path) -> List[str]:
    bad: List[str] = []
    for path in IGNORE_EXAMPLES:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=root,
        )
        if proc.returncode != 0:
            bad.append(path)
    return bad


def main() -> int:
    root = repo_root()
    errors: List[str] = []

    missing = missing_gitignore_patterns(root)
    if missing:
        errors.append(
            "ERROR: .gitignore missing cache/weight patterns: {}".format(
                ", ".join(missing)
            )
        )

    unignored = unignored_examples(root)
    if unignored:
        errors.append(
            "ERROR: .gitignore does not ignore: {}".format(", ".join(unignored))
        )

    tracked = tracked_weight_paths(root)
    if tracked:
        errors.append("ERROR: cache/weight paths are tracked:")
        errors.extend("  {}".format(path) for path in tracked)

    violations = find_violations(root)
    if violations:
        errors.append("ERROR: model weights must not be committed:")
        errors.extend("  {}: {}".format(path, reason) for path, reason in violations)

    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        sys.stderr.write("Do not add checkpoints or cache/weight dirs to git.\n")
        return 1
    sys.stdout.write("OK: no model weights or cache/weight dirs committed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
