#!/usr/bin/env bash
# Fail if git tracks model-weight files. Weights must not be committed.
set -euo pipefail

cd "$(dirname "$0")/.."

pattern='\.(pt|pth|bin|safetensors|ckpt|onnx|h5|tflite)$'
hits="$(git ls-files | grep -E "$pattern" || true)"

if [ -n "$hits" ]; then
  echo "FAIL: committed weight files are not allowed:"
  echo "$hits"
  exit 1
fi

echo "OK: no committed weight files"
