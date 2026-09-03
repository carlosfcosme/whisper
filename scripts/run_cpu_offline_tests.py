#!/usr/bin/env python3
"""CI entrypoint for CPU-only offline tests.

Enforces: no CUDA tests, no weight-download tests, network disabled,
loopback-only binds. Install dependencies before calling this script.
"""

import os
import subprocess
import sys

OFFLINE_MARKEXPR = "not requires_cuda and not requires_weights"
FORBIDDEN_SUBSTRINGS = (
    "test_transcribe[",
    "test_dtw_cuda",
    "test_median_filter_equivalence",
)


def _set_offline_env():
    os.environ["WHISPER_OFFLINE"] = "1"
    os.environ["WHISPER_NO_STORE"] = "1"
    os.environ["WHISPER_DEVICE"] = "cpu"
    os.environ["WHISPER_TEST_DISABLE_NETWORK"] = "1"


def collected_nodeids(markexpr):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            markexpr,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if proc.returncode not in (0,):
        sys.stderr.write(proc.stdout)
        raise SystemExit("pytest --collect-only failed")
    return [line.strip() for line in proc.stdout.splitlines() if "::" in line]


def enforce_collection():
    selected = collected_nodeids(OFFLINE_MARKEXPR)
    leaked = [
        node
        for node in selected
        if any(token in node for token in FORBIDDEN_SUBSTRINGS)
    ]
    if leaked:
        sys.stderr.write("ERROR: offline suite selected weight/GPU tests:\n")
        for node in leaked:
            sys.stderr.write("  {0}\n".format(node))
        raise SystemExit(2)
    if not selected:
        raise SystemExit("ERROR: offline suite collected zero tests")
    print("offline collection OK ({0} tests)".format(len(selected)), flush=True)
    return selected


def run_tests():
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--durations=0",
        "-vv",
        "-m",
        OFFLINE_MARKEXPR,
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv=None):
    _set_offline_env()
    enforce_collection()
    if argv and "--collect-only" in argv:
        return 0
    return run_tests()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
