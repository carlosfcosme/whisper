#!/usr/bin/env python3
"""Fail CI if Hugging Face Hub contact or weight pulls can sneak back in.

Stdlib only. No torch, no Hub, no weight downloads, no keys.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Optional

FORBIDDEN_IMPORTS = ("huggingface_hub", "transformers", "datasets")
REQUIRED_WORKFLOW_TOKENS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "WHISPER_OFFLINE",
    "WHISPER_NO_STORE",
    "-k 'not test_transcribe'",
    "assert_no_weight_download.py",
    "check_cache_weights.sh",
)
FORBIDDEN_WORKFLOW_TOKENS = (
    "test_transcribe[tiny]",
    "test_transcribe[tiny.en]",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workflow_path(root: Path) -> Path:
    return root / ".github" / "workflows" / "test.yml"


def _python_files(root: Path) -> List[Path]:
    package = root / "whisper"
    if not package.exists():
        return []
    return sorted(
        path
        for path in package.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


def _imported_names(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def check_package_imports(root: Path) -> List[str]:
    errors = []
    for path in _python_files(root):
        imported = _imported_names(path) & set(FORBIDDEN_IMPORTS)
        if imported:
            rel = path.relative_to(root)
            errors.append("{} imports {}".format(rel, sorted(imported)))
    return errors


def check_workflow(root: Path) -> List[str]:
    path = workflow_path(root)
    if not path.is_file():
        return ["missing {}".format(path)]
    text = path.read_text(encoding="utf-8")
    errors = []
    for token in REQUIRED_WORKFLOW_TOKENS:
        if token not in text:
            errors.append("workflow missing {}".format(token))
    for token in FORBIDDEN_WORKFLOW_TOKENS:
        if token in text:
            errors.append("workflow must not run {}".format(token))
    return errors


def find_violations(root: Path) -> List[str]:
    return check_workflow(root) + check_package_imports(root)


def main(root: Optional[Path] = None) -> int:
    if root is None:
        root = repo_root()
    errors = find_violations(root)
    if errors:
        sys.stderr.write("ERROR: no-hub / no-weight-download CI is broken:\n")
        for message in errors:
            sys.stderr.write("  {}\n".format(message))
        return 1
    sys.stdout.write("OK: no-hub CI hard; tests cannot fetch weights\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
