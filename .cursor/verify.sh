#!/usr/bin/env bash
# Localhost-only self-check for the whisper precache/verify path.
#
# This script must not be pointed at a remote or WAN host. It exports
# WHISPER_LOCALHOST_ONLY=1 so model-weight pulls to anything other than
# loopback (localhost / 127.0.0.0/8 / ::1) are refused — including the
# official CDN. A cache hit is not a pull.
#
# See .cursor/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_LOCALHOST_ONLY=1

echo "== Localhost-only policy =="
echo "  WHISPER_LOCALHOST_ONLY=${WHISPER_LOCALHOST_ONLY}"
echo "  allowed hosts: localhost, 127.0.0.0/8, ::1"
echo "  refused: remote/WAN pulls (CDN, public DNS, LAN, public IPs)"

echo "== Guard tests (no network) =="
python3 -m pytest -q tests/test_localhost_only.py

echo "== Cache-miss against the official CDN must be refused =="
python3 - <<'PY'
import tempfile

import whisper
from whisper.localhost import RemotePullError

url = whisper._MODELS["tiny.en"]
with tempfile.TemporaryDirectory() as tmp:
    try:
        whisper._download(url, tmp, in_memory=False)
    except RemotePullError as exc:
        print(f"  refused (expected): {exc}")
    else:
        raise SystemExit("FAIL: WAN pull to the official CDN was not refused")
PY

echo "== VERIFY OK: precache/verify path is localhost-only =="
