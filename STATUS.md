# Status

This environment is **offline by default**, **no Hub**, **CPU-only**, and
binds serve to **127.0.0.1**.

| Gate | Encoding |
| --- | --- |
| Offline default | `WHISPER_OFFLINE=1` (library default); `_download` does not fetch |
| No Hub | `WHISPER_NO_HUB=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| CPU-only | `whisper.DEFAULT_DEVICE=cpu`; CLI `--device` default `cpu` |
| Bind | `whisper.localhost.BIND_HOST=127.0.0.1`; `0.0.0.0` rejected |

See `.cursor/environment.json`, `.cursor/install.sh`, and `.cursor/whisper-policy.env`.
Remote checkpoint fetch requires `WHISPER_OFFLINE=0` and `WHISPER_NO_HUB=0`.
No secrets are stored in this repository.
