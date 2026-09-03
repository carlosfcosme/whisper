#!/usr/bin/env bash
# Fail if the all-interfaces bind token appears under whisper/ or .cursor/.
# Tests may mention the token when asserting it is rejected.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if git grep -nF '0.0.0.0' -- whisper .cursor; then
  echo "ERROR: 0.0.0.0 is forbidden under whisper/ and .cursor/; bind 127.0.0.1" >&2
  exit 1
fi
echo "OK: no 0.0.0.0 under whisper/ or .cursor/"
