#!/usr/bin/env bash
# Fail if production bind/listen host is not loopback (127.0.0.1).
# Grep rejects forbidden 0.0.0.0 in whisper/ sources.
set -euo pipefail
ROOT="${BIND_GUARD_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${ROOT}"

defaults="${ROOT}/whisper/defaults.py"
if [[ ! -f "${defaults}" ]]; then
  echo "error: missing whisper/defaults.py" >&2
  exit 1
fi

if ! grep -qE '^DEFAULT_HOST = "127\.0\.0\.1"$' "${defaults}"; then
  echo "error: DEFAULT_HOST must be 127.0.0.1 (loopback)" >&2
  grep -n 'DEFAULT_HOST' "${defaults}" >&2 || true
  exit 1
fi

# Forbidden all-interfaces bind/listen assignments in the package.
violations="$(
  grep -RnE \
    --include='*.py' \
    '(DEFAULT_HOST|host)[[:space:]]*=[[:space:]]*"0\.0\.0\.0"|HTTPServer\(\([[:space:]]*"0\.0\.0\.0"|\.bind\(\([[:space:]]*"0\.0\.0\.0"|listen\(\([[:space:]]*"0\.0\.0\.0"' \
    "${ROOT}/whisper" || true
)"

if [[ -n "${violations}" ]]; then
  echo "error: forbidden 0.0.0.0 bind/listen:" >&2
  echo "${violations}" >&2
  exit 1
fi
