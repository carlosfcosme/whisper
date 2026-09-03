#!/usr/bin/env bash
# Fail if any registered test fixture path is a remote / Hub / WAN URL.
# Stdlib only — does not install the package or download anything.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec python3 whisper/local_fixtures.py
