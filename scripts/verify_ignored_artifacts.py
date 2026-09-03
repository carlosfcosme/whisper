#!/usr/bin/env python3
"""Verify cache/weight artifacts stay gitignored and untracked.

Default mode only checks pathspecs (no files written). ``--plant`` writes dummy
ignored artifacts, asserts ``git status --ignored`` reports them, then deletes
them. Never downloads weights.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

ARTIFACT_SAMPLES = (
    "tiny.pt",
    "model.pth",
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "checkpoints/model.safetensors",
    ".huggingface/hub/models--openai--whisper-tiny/pytorch_model.bin",
)

LS_FILES_PATHSPECS = (
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
    "*.safetensors",
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )


def is_ignored(root: Path, relpath: str) -> bool:
    return _git(root, "check-ignore", "-q", "--", relpath).returncode == 0


def tracked_artifact_paths(root: Path, pathspecs: Sequence[str] = LS_FILES_PATHSPECS):
    listed = _git(root, "ls-files", "-z", "--", *pathspecs)
    if listed.returncode != 0:
        raise RuntimeError(listed.stderr)
    return [path for path in listed.stdout.split("\0") if path]


def missing_ignored(root: Path, samples: Sequence[str] = ARTIFACT_SAMPLES):
    return [path for path in samples if not is_ignored(root, path)]


def _ensure_parents(path: Path, root: Path) -> List[Path]:
    created = []
    missing = []
    current = path
    while current != root and not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        created.append(directory)
    return created


def plant_artifacts(
    root: Path, samples: Sequence[str] = ARTIFACT_SAMPLES
) -> Tuple[List[Path], List[Path]]:
    files = []
    dirs = []
    for relpath in samples:
        path = root / relpath
        dirs.extend(_ensure_parents(path.parent, root))
        path.write_bytes(b"ignored-artifact-not-a-checkpoint")
        files.append(path)
    return files, dirs


def remove_planted(files: Iterable[Path], dirs: Iterable[Path]) -> None:
    for path in files:
        if path.is_file():
            path.unlink()
    for directory in reversed(list(dirs)):
        try:
            directory.rmdir()
        except OSError:
            pass


def ignored_status_lines(root: Path, samples: Sequence[str] = ARTIFACT_SAMPLES):
    result = _git(
        root,
        "status",
        "--ignored",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *samples,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return [line for line in result.stdout.splitlines() if line.strip()]


def verify_planted_ignored(root: Path, samples: Sequence[str] = ARTIFACT_SAMPLES):
    errors = []
    lines = ignored_status_lines(root, samples)
    by_path = {}
    for line in lines:
        tag, path = line[:2], line[3:]
        by_path[path] = tag
    for relpath in samples:
        tag = by_path.get(relpath)
        if tag is None:
            parent = str(Path(relpath).parts[0]) + "/"
            if by_path.get(parent) == "!!" or by_path.get(parent.rstrip("/")) == "!!":
                continue
            errors.append("{} missing from git status --ignored".format(relpath))
            continue
        if tag != "!!":
            errors.append("{} reported as {!r}, expected ignored".format(relpath, tag))
        if not is_ignored(root, relpath):
            errors.append("{} is not gitignored".format(relpath))
    return errors


def check_static(root: Path) -> List[str]:
    errors = []
    missing = missing_ignored(root)
    if missing:
        errors.append("not gitignored: {}".format(", ".join(missing)))
    tracked = tracked_artifact_paths(root)
    if tracked:
        errors.append("tracked cache/weight artifacts: {}".format(", ".join(tracked)))
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plant",
        action="store_true",
        help="write dummy ignored artifacts, verify, then delete them",
    )
    args = parser.parse_args(argv)
    errors = check_static(REPO)
    planted_files = []
    planted_dirs = []
    try:
        if args.plant:
            planted_files, planted_dirs = plant_artifacts(REPO)
            errors.extend(verify_planted_ignored(REPO))
            errors.extend(check_static(REPO))
    finally:
        remove_planted(planted_files, planted_dirs)

    if errors:
        sys.stderr.write("ERROR: ignored artifact verification failed:\n")
        for line in errors:
            sys.stderr.write("  {}\n".format(line))
        return 1
    sys.stdout.write("OK: cache/weight artifacts are gitignored and untracked\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
