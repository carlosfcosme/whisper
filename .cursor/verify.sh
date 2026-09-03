#!/usr/bin/env bash
# CI / Cloud Agent verify: force 127.0.0.1 and no default weight fetch.
# Must not download Whisper checkpoints. Exits non-zero on any failure.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source .cursor/env.sh

if [[ "${WHISPER_BIND_HOST}" != "127.0.0.1" ]]; then
  echo "FAIL: WHISPER_BIND_HOST must be 127.0.0.1" >&2
  exit 1
fi
if [[ "${WHISPER_ALLOW_WEIGHT_FETCH}" != "0" ]]; then
  echo "FAIL: WHISPER_ALLOW_WEIGHT_FETCH must be 0 (no default weight fetch)" >&2
  exit 1
fi

xdg_root="$(mktemp -d)"
export XDG_CACHE_HOME="${xdg_root}"
trap 'rm -rf "${xdg_root}"' EXIT

echo "== Policy =="
echo "  WHISPER_BIND_HOST=${WHISPER_BIND_HOST}"
echo "  WHISPER_ALLOW_WEIGHT_FETCH=${WHISPER_ALLOW_WEIGHT_FETCH}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME}"

echo "== Static: install does not pull weights or bind 0.0.0.0 =="
python3 - <<'PY'
import json
from pathlib import Path

root = Path(".")
install = (root / ".cursor/install.sh").read_text()
env_sh = (root / ".cursor/env.sh").read_text()
start = (root / ".cursor/start.sh").read_text()
verify = (root / ".cursor/verify.sh").read_text()
doc = (root / "ENVIRONMENT.md").read_text()
env = json.loads((root / ".cursor/environment.json").read_text())

wildcard = ".".join(["0"] * 4)
assert "load_model(" not in install, "install.sh must not call load_model"
assert "WHISPER_PRECACHE" not in install, "install.sh must not precache weights"
assert wildcard not in install
assert wildcard not in start
assert wildcard not in env_sh
assert "--host 127.0.0.1" in start
assert "WHISPER_BIND_HOST=127.0.0.1" in env_sh
assert "WHISPER_ALLOW_WEIGHT_FETCH=0" in env_sh
assert "WHISPER_DEVICE=cpu" in env_sh
assert "ports" not in env, env
assert env.get("install") == "bash .cursor/install.sh"
assert "## Install" in doc and "## No-weight-pull" in doc and "## Localhost-only" in doc
print("  static checks ok")
PY

echo "== Executable environment tests (bind, no fetch, gitignore) =="
python3 -m pytest -q -m environment

echo "== Guard tests (no WAN, no checkpoints) =="
python3 -m pytest -q tests/test_env_policy.py

echo "== Default CDN cache-miss is refused =="
python3 - <<'PY'
import tempfile

import whisper
from whisper.env_policy import WeightFetchError

url = whisper._MODELS["turbo"]
with tempfile.TemporaryDirectory() as tmp:
    try:
        whisper._download(url, tmp, in_memory=False)
    except WeightFetchError as exc:
        print(f"  refused (expected): {exc}")
    else:
        raise SystemExit("FAIL: default turbo weight fetch was not refused")
PY

echo "== No-weight-pull unit tests =="
python3 -m pytest -q -k "not test_transcribe" -m "not requires_cuda"

echo "== Isolated cache has no checkpoints =="
if find "${xdg_root}" -type f \( -name "*.pt" -o -name "*.pth" \) | grep -q .; then
  echo "FAIL: verify wrote weight files under XDG_CACHE_HOME" >&2
  find "${xdg_root}" -type f \( -name "*.pt" -o -name "*.pth" \) >&2
  exit 1
fi
echo "  no .pt/.pth under ${xdg_root}"

echo "== VERIFY OK: 127.0.0.1 forced; no default weight fetch =="
