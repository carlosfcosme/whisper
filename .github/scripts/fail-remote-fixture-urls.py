#!/usr/bin/env python3
"""Fail CI when tests assign remote audio fixture URLs.

Negative-test modules may mention http(s) URLs to assert they are refused.
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
ALLOW_URL_LITERALS = {
    "test_local_fixtures.py",
    "test_runtime_guards.py",
}
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg", ".m4a")
ASSIGN_NAMES = {
    "audio_path",
    "sample_audio_path",
    "tiny_wav_path",
    "fixture_path",
    "SAMPLE_AUDIO_PATH",
    "IN_REPO_SAMPLE_AUDIO",
}


def _is_remote_audio(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered.startswith(("http://", "https://")):
        return False
    path = lowered.split("?", 1)[0]
    return any(path.endswith(suffix) for suffix in AUDIO_SUFFIXES)


def _string_constants(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    for child in ast.iter_child_nodes(node):
        yield from _string_constants(child)


def main() -> int:
    sample = TESTS / "jfk.flac"
    if not sample.is_file():
        print("in-repo fixture missing: tests/jfk.flac", file=sys.stderr)
        return 1

    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        allow = path.name in ALLOW_URL_LITERALS
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
                if not any(name in ASSIGN_NAMES for name in names):
                    continue
                for value in _string_constants(node.value):
                    if _is_remote_audio(value):
                        offenders.append("%s: %s" % (path, value))
            if isinstance(node, ast.Call):
                func = node.func
                called = ""
                if isinstance(func, ast.Name):
                    called = func.id
                elif isinstance(func, ast.Attribute):
                    called = func.attr
                if called == "load_audio" and not allow:
                    for value in _string_constants(node):
                        if _is_remote_audio(value):
                            offenders.append("%s: load_audio(%s)" % (path, value))

    if offenders:
        print("Remote fixture URLs are forbidden:", file=sys.stderr)
        for item in offenders:
            print(item, file=sys.stderr)
        return 1

    print("ok: fixture paths are local (tests/jfk.flac present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
