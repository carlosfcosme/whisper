#!/usr/bin/env bash
# Fail if git ls-files lists weight blobs or cache/weight directories.
# No Hub fetch, no keys, dummy bytes in a matching path are enough to fail.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

cd "$ROOT"

fail=0
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  base="${rel##*/}"
  case "$base" in
    *.pt|*.pth|*.pth.tar|*.safetensors|*.ckpt|*.onnx|*.gguf|*.ggml|*.bin)
      echo "tracked weight blob: $rel"
      fail=1
      ;;
  esac
  case "$rel" in
    .cache/*|cache/*|weights/*|checkpoints/*|.huggingface/*|.torch/*|\
    */.cache/*|*/cache/*|*/weights/*|*/checkpoints/*|*/.huggingface/*|*/.torch/*)
      echo "tracked cache/weight path: $rel"
      fail=1
      ;;
  esac
done < <(git ls-files)

if [[ "$fail" -ne 0 ]]; then
  echo "check_tracked_weights.sh failed: git ls-files listed weight/cache artifacts"
  exit 1
fi

echo "check_tracked_weights.sh passed: git ls-files has no weight blobs"
exit 0
