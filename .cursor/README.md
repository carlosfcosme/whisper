# Cloud Agent environment

## Localhost-only precache / verify

The precache verify path is **localhost-only**. It is easy to run against
the wrong host: on a cache miss, `whisper.load_model` / `_download` would
otherwise pull weights from `openaipublic.azureedge.net` (or any other
URL in `_MODELS`).

That is refused.

- **Allowed download hosts:** `localhost`, `127.0.0.0/8`, `::1`, and
  `file:` URLs.
- **Refused:** remote and WAN pulls — public DNS names, public IPs, LAN
  addresses, and the official Azure CDN. Redirects to those hosts are
  refused too. Hostnames are not resolved (no DNS rebinding).
- **Enforcement:** set `WHISPER_LOCALHOST_ONLY=1`. `.cursor/verify.sh`
  always does this. `whisper._download` then refuses the pull before any
  network call.
- **Cache hits are not pulls.** A SHA-256-valid file already in the
  download root is reused even if the catalog URL is remote.
- **Install vs verify.** Package install may contact the network for
  Python dependencies. Verify must never pull model weights from a
  remote/WAN host.

```bash
bash tests/run_localhost_only.sh
bash .cursor/verify.sh
```

`tests/run_localhost_only.sh` is the executable localhost-only runner
(`WHISPER_LOCALHOST_ONLY=1`, CPU, disposable cache). Pytest hooks block
non-loopback `urlopen` / `urlretrieve` for `@pytest.mark.localhost_only`
so network and model downloads cannot run. Generated artifacts
(`.pytest_cache`, coverage, transcript files under `tests/`) are
gitignored.

CI runs `.cursor/verify.sh`, which invokes that runner. The
`localhost-only-verify` job in `.github/workflows/test.yml` installs the
package (PyTorch only) and must not download Whisper checkpoints. Tests
are selected with `-m localhost_only`. A disposable `XDG_CACHE_HOME` is
asserted empty of `.pt` / `.pth` / `.bin` files when verify exits.

Override: unset `WHISPER_LOCALHOST_ONLY` (or set it to `0`) only when you
intentionally want a WAN fetch. Do not do that in verify or CI.

## Serve / bind: 127.0.0.1 only

Any serve or start path must bind **127.0.0.1** only. Wildcard, LAN, and
WAN listen addresses are rejected before the socket is opened.

- **Allowed bind:** `127.0.0.1`
- **Rejected:** wildcard (`0.0.0.0`, `::`), `localhost`, `::1`, LAN, WAN
- **Enforcement:** `whisper.bind.require_bind_127_0_0_1` — used by
  `python -m whisper.serve` and `.cursor/start.sh`
- **No weights. No WAN.** The serve path is a local health endpoint. It
  does not call `load_model` / `_download` and does not fetch checkpoints.

```bash
bash .cursor/start.sh
```

`tests/test_bind_localhost.py` fails if `0.0.0.0` appears in start scripts
(`.cursor/*.sh`, `start*.sh`, `serve*.sh`, `.cursor/environment.json`).

`tests/test_bind_integration.py` starts the real CLI and `.cursor/start.sh`
as subprocesses: they must print `bound to 127.0.0.1` and serve a no-weight
health response. `python -m whisper.serve --host` with a wildcard exits 2
and never binds.

No secrets are stored in this repository.
