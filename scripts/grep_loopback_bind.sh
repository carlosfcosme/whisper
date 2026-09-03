#!/usr/bin/env bash
# CI grep guard: start/serve sources must not name a non-loopback bind host.
# Needles live in this script; tests/ and this file are not in the default scan.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
  files=("$@")
else
  files=(
    .cursor/start.sh
    .cursor/environment.json
    whisper/serve.py
    whisper/bind.py
  )
  shopt -s nullglob
  extra=(start*.sh serve*.sh)
  files+=("${extra[@]}")
fi

for f in "${files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "FAIL: missing scan target $f"
    exit 1
  fi
done

if grep -nE '0\.0\.0\.0|INADDR_ANY|IN6ADDR_ANY' "${files[@]}"; then
  echo "FAIL: grep found a wildcard / non-loopback bind host"
  exit 1
fi

host_hits="$(grep -nE -- '--host[=[:space:]]+[^[:space:]]+' "${files[@]}" || true)"
if [ -n "$host_hits" ]; then
  bad="$(printf '%s\n' "$host_hits" | grep -Ev -- '--host[=[:space:]]+127\.0\.0\.1' || true)"
  if [ -n "$bad" ]; then
    printf '%s\n' "$bad"
    echo "FAIL: grep found --host that is not 127.0.0.1"
    exit 1
  fi
fi

echo "OK: grep guard — bind hosts are 127.0.0.1 only"
