#!/usr/bin/env bash
# Fail if weight/cache paths are not gitignored, or if git add tracks a .pt.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

tmp=".offline-ignored-weight.pt"
cleanup() {
  rm -f "${tmp}"
  git rm -f --cached -- "${tmp}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf 'not-a-checkpoint' > "${tmp}"

if ! git check-ignore -q "${tmp}"; then
  echo "error: ${tmp} is not gitignored" >&2
  exit 1
fi
if ! git check-ignore -q ".cache/whisper/tiny.pt"; then
  echo "error: .cache/whisper/tiny.pt is not gitignored" >&2
  exit 1
fi
if ! git check-ignore -q "cache/whisper/tiny.pt"; then
  echo "error: cache/whisper/tiny.pt is not gitignored" >&2
  exit 1
fi

git add -- "${tmp}" 2>/dev/null || true
if [[ -n "$(git ls-files -- "${tmp}")" ]]; then
  echo "error: ignored weight was tracked by git add" >&2
  exit 1
fi
