#!/usr/bin/env bash
# Fail CI when model weight files are committed to git.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
weights=$(git ls-files -- '*.pt' '*.pth' '*.ckpt' '*.safetensors')
if [[ -n "${weights}" ]]; then
  echo "error: committed model weights are forbidden:" >&2
  echo "${weights}" >&2
  exit 1
fi
