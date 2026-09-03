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
bash .cursor/verify.sh
```

Override: unset `WHISPER_LOCALHOST_ONLY` (or set it to `0`) only when you
intentionally want a WAN fetch. Do not do that in verify.

No secrets are stored in this repository.
