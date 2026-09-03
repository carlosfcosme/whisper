#!/usr/bin/env bash
# Fail CI if tracked files match weight/cache patterns (git ls-files | grep -E).
# Also require .gitignore rules that keep Whisper cache/weight dumps out of git.
# Does not download weights and does not contact the Hugging Face Hub.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

GITIGNORE=".gitignore"

# Exact lines that must appear in .gitignore (repo-relative cache/weight dumps).
REQUIRED_PATTERNS=(
  '.cache/'
  '.cache/whisper/'
  'cache/'
  'models/'
  'weights/'
  '*.pt'
  '*.bin'
  '*.onnx'
)

# Ticket 3: weight blobs and cache/weight directories.
WEIGHT_CACHE_GREP='(\.pt|\.pth|\.bin|\.ckpt|\.safetensors|\.onnx|\.gguf|\.ggml|\.tflite|\.pb|\.weights)$|(^|/)\.cache/|(^|/)cache/|(^|/)\.huggingface/|(^|/)huggingface/|(^|/)weights/|(^|/)models/|(^|/)checkpoints/'

# Example paths git check-ignore must match (files need not exist).
IGNORE_EXAMPLES=(
  '.cache/whisper/tiny.pt'
  '.cache/whisper/base.en.pt'
  '.cache/huggingface/hub/models--openai--whisper/model.safetensors'
  'cache/whisper/small.bin'
  'models/tiny.pt'
  'models/large.onnx'
  'weights/base.bin'
  'checkpoints/decoder.pt'
  'tiny.pt'
  'export.onnx'
  'model.bin'
)

if [[ ! -f "$GITIGNORE" ]]; then
  echo "error: ${GITIGNORE} is missing" >&2
  exit 1
fi

missing=0
for pat in "${REQUIRED_PATTERNS[@]}"; do
  if ! grep -qxF -- "$pat" "$GITIGNORE"; then
    echo "error: ${GITIGNORE} is missing required pattern: ${pat}" >&2
    missing=1
  fi
done
if (( missing )); then
  echo "error: cache/weight directories must stay gitignored" >&2
  exit 1
fi

hits="$(git ls-files | grep -E "$WEIGHT_CACHE_GREP" || true)"
if [[ -n "$hits" ]]; then
  echo "error: tracked weight/cache artifacts (CI must fail):" >&2
  printf '%s\n' "$hits" >&2
  echo "Remove them with: git rm --cached -- <path>" >&2
  exit 1
fi

not_ignored=0
for path in "${IGNORE_EXAMPLES[@]}"; do
  if ! git check-ignore -q -- "$path"; then
    echo "error: expected gitignore to match: ${path}" >&2
    not_ignored=1
  fi
done
if (( not_ignored )); then
  echo "error: cache/weight example paths must stay gitignored" >&2
  exit 1
fi

echo "OK: no tracked weight/cache artifacts; ignore rules cover known paths"
