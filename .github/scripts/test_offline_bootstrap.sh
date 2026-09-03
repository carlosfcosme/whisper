#!/usr/bin/env bash
# Offline bootstrap test target.
# Blocks WAN/model fetch (pytest --offline-bootstrap + dead loopback proxy)
# and requires 127.0.0.1 binds. No secrets, no kernel modules, no BPF.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

# If a fetch slips past the urllib/socket patches, it hits 127.0.0.1:9
# (discard), not the WAN. Loopback itself is exempt via NO_PROXY.
export http_proxy="${http_proxy:-http://127.0.0.1:9}"
export https_proxy="${https_proxy:-http://127.0.0.1:9}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:9}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:9}"
export NO_PROXY="${NO_PROXY:-127.0.0.1}"
export no_proxy="${no_proxy:-127.0.0.1}"

"$PYTHON" -m pytest \
  -m "not requires_network and not requires_cuda" \
  --offline-bootstrap \
  --durations=0 \
  -vv \
  "$@"

"$PYTHON" .github/scripts/check_loopback_listen.py
