#!/usr/bin/env bash
# Fail if cache/weight directories or checkpoint files are tracked in git.
# Invoked by CI (.github/workflows/test.yml) and the local pre-commit hook.
# Does not download model weights and does not read secrets.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

GITIGNORE=".gitignore"

REQUIRED_PATTERNS=(
  '.cache/'
  'cache/'
  'weights/'
  '*.pt'
  '*.pth'
)

TRACKED_PATHSPECS=(
  '.cache'
  '.cache/**'
  'cache'
  'cache/**'
  'weights'
  'weights/**'
  '*.pt'
  '*.pth'
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

tracked="$(git ls-files -- "${TRACKED_PATHSPECS[@]}")"
if [[ -n "$tracked" ]]; then
  echo "error: cache or weight files are tracked (CI must fail):" >&2
  printf '%s\n' "$tracked" >&2
  echo "Remove them with: git rm --cached -- <path>" >&2
  exit 1
fi

echo "OK: no tracked cache dirs or weight files"
