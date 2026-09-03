#!/usr/bin/env bash
# Sovereign Cloud Agent self-check.
# No WAN, no weight pull, localhost bind only. Exits non-zero on failure.
set -euo pipefail

cd "$(dirname "$0")/.."

# Isolated cache: a stray load_model must not write ~/.cache/whisper.
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(mktemp -d)}"
export XDG_CACHE_HOME

# Blackhole proxy on localhost: any WAN attempt fails immediately.
export http_proxy="${http_proxy:-http://127.0.0.1:9}"
export https_proxy="${https_proxy:-http://127.0.0.1:9}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

echo "== Sovereign: localhost bind =="
python3 - <<'PY'
import json
from pathlib import Path

env = json.loads(Path(".cursor/environment.json").read_text())
assert "ports" not in env, env
assert "start" not in env, env
print("  environment publishes no ports; install binds nothing")
PY

echo "== Sovereign: no WAN install path =="
bash .cursor/install.sh

echo "== Sovereign: import only (no load_model) =="
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__)"

echo "== Sovereign: weight-free tests =="
pytest -q \
  -k 'not test_transcribe' \
  -m 'not requires_cuda'

echo "== Sovereign: no weight files in isolated cache =="
if find "$XDG_CACHE_HOME" -type f \( -name '*.pt' -o -name '*.pth' \) | grep -q .; then
  echo "FAIL: checkpoint written under XDG_CACHE_HOME=$XDG_CACHE_HOME" >&2
  find "$XDG_CACHE_HOME" -type f >&2
  exit 1
fi
echo "  isolated cache has no .pt / .pth"

echo "== SOVEREIGN OK =="
