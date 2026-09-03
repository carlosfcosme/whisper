#!/usr/bin/env bash
# CI entry point: localhost-only verify.
#
# No model-weight download. No WAN pulls. Does not run transcription tests.
# GitHub Actions job localhost-only-verify invokes this script.
#
# See .cursor/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_LOCALHOST_ONLY=1

VERIFY_CACHE="$(mktemp -d)"
cleanup() { rm -rf "$VERIFY_CACHE"; }
trap cleanup EXIT
export XDG_CACHE_HOME="$VERIFY_CACHE"

echo "== Localhost-only CI verify =="
echo "  WHISPER_LOCALHOST_ONLY=${WHISPER_LOCALHOST_ONLY}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME} (disposable; must stay empty of weights)"
echo "  allowed hosts: localhost, 127.0.0.0/8, ::1"
echo "  refused: remote/WAN pulls (CDN, public DNS, LAN, public IPs)"
echo "  serve/bind: 127.0.0.1 only"
echo "  pytest: -m localhost_only (no transcription tests, no weight download)"

echo "== Guard tests (no network, no weights) =="
python3 -m pytest -q -m 'localhost_only and not requires_cuda'

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

echo "== No weight files written =="
written="$(find "$VERIFY_CACHE" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -print)"
if [ -n "$written" ]; then
  echo "FAIL: model weight file written during verify:"
  echo "$written"
  exit 1
fi
echo "  ok: no .pt/.pth/.bin under XDG_CACHE_HOME"

echo "== VERIFY OK: CI localhost-only verify (no weight download) =="
