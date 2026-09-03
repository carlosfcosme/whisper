#!/usr/bin/env python3
"""Fail CI if serve or start scripts bind 0.0.0.0."""

import importlib.util
import sys
from pathlib import Path

ALL_INTERFACES = "0.0.0.0"
LOOPBACK = "127.0.0.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_serve(root: Path):
    path = root / "whisper" / "serve.py"
    spec = importlib.util.spec_from_file_location("whisper_serve_ci", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_script_paths(root: Path):
    found = []
    cursor = root / ".cursor"
    if cursor.is_dir():
        for path in cursor.iterdir():
            if path.is_file() and (
                path.name == "start.sh"
                or (path.name.startswith("start-") and path.name.endswith(".sh"))
            ):
                found.append(path)
        env = cursor / "environment.json"
        if env.is_file():
            found.append(env)
    return sorted(found)


def main() -> int:
    root = repo_root()
    serve = load_serve(root)
    errors = []

    if getattr(serve, "LOOPBACK_BIND", None) != LOOPBACK:
        errors.append("whisper.serve.LOOPBACK_BIND must be {}".format(LOOPBACK))
    if getattr(serve, "ALL_INTERFACES", None) != ALL_INTERFACES:
        errors.append("whisper.serve.ALL_INTERFACES must be {}".format(ALL_INTERFACES))

    try:
        resolved = serve.require_loopback_bind()
    except Exception as exc:  # pragma: no cover - import/runtime failure
        errors.append("require_loopback_bind() raised: {}".format(exc))
        resolved = None
    if resolved != LOOPBACK:
        errors.append("default bind is {!r}, expected {}".format(resolved, LOOPBACK))

    try:
        serve.require_loopback_bind(ALL_INTERFACES)
        errors.append("require_loopback_bind({!r}) must raise".format(ALL_INTERFACES))
    except serve.BindError:
        pass
    except Exception as exc:
        errors.append(
            "require_loopback_bind({!r}) raised {}: {}".format(
                ALL_INTERFACES, type(exc).__name__, exc
            )
        )

    scripts = start_script_paths(root)
    start = root / ".cursor" / "start.sh"
    if not start.is_file():
        errors.append("missing {}".format(start))
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        if ALL_INTERFACES in text:
            errors.append("{} binds or mentions {}".format(path, ALL_INTERFACES))
        if path.name == "start.sh" and LOOPBACK not in text:
            errors.append("{} must bind {}".format(path, LOOPBACK))

    if errors:
        sys.stderr.write("ERROR: localhost-only bind policy violated:\n")
        for item in errors:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: serve and start scripts bind {} only\n".format(LOOPBACK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
