#!/usr/bin/env bash
# Fail if cache/weight directories or checkpoint files are tracked, or if
# .gitignore no longer covers those paths.
# Invoked by CI (.github/workflows/test.yml) and the local pre-commit hook.
# Does not download model weights and does not contact the Hugging Face Hub.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

GITIGNORE=".gitignore"

# Required ignore rules. Patterns must appear as exact lines in .gitignore.
REQUIRED_PATTERNS=(
  '.cache/'
  'cache/'
  'weights/'
  '*.pt'
  '*.pth'
)

# Pathspecs that must not appear in the git index / tree.
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

# Example paths that git check-ignore must match (files need not exist).
IGNORE_EXAMPLES=(
  '.cache/whisper/tiny.pt'
  'cache/whisper/tiny.pt'
  'weights/tiny.pt'
  'tiny.pt'
  'model.pth'
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

echo "OK: cache/weight paths untracked and gitignored"
