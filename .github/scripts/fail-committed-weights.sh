#!/usr/bin/env bash
# Fail the build if model weight files are tracked in git.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

tracked="$(git ls-files -- '*.pt' '*.pth' '*.ckpt' '*.safetensors' || true)"
if [[ -n "${tracked}" ]]; then
  echo "Committed model weights are not allowed:" >&2
  echo "${tracked}" >&2
  exit 1
fi

echo "OK: no committed model weights"
