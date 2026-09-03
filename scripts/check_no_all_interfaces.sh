#!/usr/bin/env bash
# Fail if all-interface / non-loopback bind tokens appear under whisper/ or
# .cursor/. Tests may mention those tokens when asserting they are rejected.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
failed=0
# --untracked so a newly added 0.0.0.0 file fails before it is committed.
if git grep -nE --untracked '0\.0\.0\.0|INADDR_ANY|inaddr_any' -- whisper .cursor; then
  echo "ERROR: all-interface bind token is forbidden under whisper/ and .cursor/; bind 127.0.0.1" >&2
  failed=1
fi
if git grep -nE --untracked '(HTTPServer|ThreadingHTTPServer|TCPServer|UDPServer|bind|listen)[[:space:]]*\([[:space:]]*\([[:space:]]*['\''"]['\''"]' -- whisper .cursor; then
  echo "ERROR: empty-host bind() is forbidden under whisper/ and .cursor/; bind 127.0.0.1" >&2
  failed=1
fi
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "OK: no 0.0.0.0 / INADDR_ANY / empty-host bind under whisper/ or .cursor/"
