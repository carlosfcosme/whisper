#!/usr/bin/env bash
# Fail if a test or install step downloaded model weight files.
set -euo pipefail
roots=()
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  roots+=("${XDG_CACHE_HOME}")
fi
roots+=("${HOME}/.cache/whisper")

found=0
for root in "${roots[@]}"; do
  if [[ ! -d "${root}" ]]; then
    continue
  fi
  while IFS= read -r -d '' file; do
    echo "error: weight download detected: ${file}" >&2
    found=1
  done < <(
    find "${root}" -type f \
      \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) \
      -print0 2>/dev/null
  )
done
exit "${found}"
