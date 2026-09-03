#!/usr/bin/env bash
# Fail CI if model weights or large binaries are committed.
# Existing repo assets (sample audio, diagrams, tokenizer tables, notebooks)
# stay under the size limits; checkpoints must never be tracked.
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ANY_BYTES=$((8 * 1024 * 1024))
MAX_BINARY_BYTES=$((2 * 1024 * 1024))

# Weight / checkpoint extensions. whisper/assets/mel_filters.npz is a small
# filterbank, not a model, and is allowlisted below.
WEIGHT_GLOB='*.pt *.pth *.onnx *.safetensors *.ckpt *.ggml *.gguf *.h5 *.tflite *.pb *.mlmodel *.weights'

is_allowlisted() {
  # Upstream sample/docs assets only. Weight extensions never match these.
  case "$1" in
    whisper/assets/*) return 0 ;;
    notebooks/*.ipynb) return 0 ;;
    tests/jfk.flac) return 0 ;;
    approach.png) return 0 ;;
    language-breakdown.svg) return 0 ;;
    *) return 1 ;;
  esac
}

is_weight_name() {
  local path="$1" base
  base="$(basename "$path")"
  case "$base" in
    *.pt|*.pth|*.onnx|*.safetensors|*.ckpt|*.ggml|*.gguf|*.h5|*.tflite|*.pb|*.mlmodel|*.weights)
      return 0
      ;;
  esac
  return 1
}

is_binary() {
  # NUL byte => treat as binary (git's usual heuristic).
  grep -q $'\0' "$1" 2>/dev/null
}

fail=0
while IFS= read -r -d '' path; do
  if is_allowlisted "$path"; then
    continue
  fi
  if is_weight_name "$path"; then
    echo "FAIL: model weight/checkpoint is committed: ${path}"
    fail=1
    continue
  fi
  if [[ ! -f "$path" ]]; then
    continue
  fi
  size="$(wc -c <"$path")"
  if (( size > MAX_ANY_BYTES )); then
    echo "FAIL: file exceeds 8 MiB: ${path} (${size} bytes)"
    fail=1
    continue
  fi
  if (( size > MAX_BINARY_BYTES )) && is_binary "$path"; then
    echo "FAIL: large binary is committed: ${path} (${size} bytes)"
    fail=1
  fi
done < <(git ls-files -z)

if (( fail )); then
  echo "Committed model weights or large binaries are not allowed."
  exit 1
fi

echo "OK: no committed model weights or large binaries"
# Keep the weight glob visible so the policy is greppable.
: "${WEIGHT_GLOB}"
