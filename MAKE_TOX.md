# Make and tox

Offline, no-download tests and localhost-only service binds are run through
[`Makefile`](Makefile) and [`tox.ini`](tox.ini). CI invokes those entry
points; they do not fetch Whisper checkpoints or Hugging Face Hub artifacts.

## Inventory

| Tool | Present? | Role |
|------|----------|------|
| [`Makefile`](Makefile) | Yes | `test-offline`, `tox-offline`, `ci-offline` |
| [`tox.ini`](tox.ini) | Yes | `tox -e offline` |
| CI | Yes | matrix job runs `make test-offline`; `tox-offline` job runs make and tox |

Both set `WHISPER_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`WHISPER_BIND_HOST=127.0.0.1`.

## Commands

Install ffmpeg, CPU PyTorch, and `pip install -e ".[dev]"` (includes `tox`).
Then:

```bash
make test-offline
tox -e offline
# or
make ci-offline
```

These run:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' \
  -m 'not requires_cuda and not requires_weights'
```

`test_transcribe` is marked `requires_weights` and is skipped. Do not commit
`.pt` files. `whisper._download` refuses WAN fetches while the offline flags
are set.

## Localhost-only binds

[`whisper/localhost.py`](whisper/localhost.py) refuses wildcard (`0.0.0.0`,
`::`) and WAN hosts. Helper sockets must use `bind_localhost()` so listeners
stay on `127.0.0.1` / `::1`. There is no `--live` / `0.0.0.0` mode.

## Secrets

No credentials are required. Do not set `HF_TOKEN` or other Hub tokens.
Release publishing reads a GitHub Actions secret; do not put tokens in the
tree.
