# CLI entrypoint

The [README Command-line usage](README.md#command-line-usage) samples invoke
`whisper` as if it were already on `PATH`. That command is a setuptools
console script, not a file in this repository. This file records the
entrypoint **from the tree** so the install environment is not implicit.

No model weights and no secrets belong in this file or in git.

## What `pip install` registers

`pyproject.toml` declares:

```toml
scripts.whisper = "whisper.transcribe:cli"
```

The generated egg-info repeats the same mapping:

```
[console_scripts]
whisper = whisper.transcribe:cli
```

After `pip install` (or the editable install in `.cursor/install.sh`), pip
writes a `whisper` executable into the active interpreter's scripts
directory. **`--break-system-packages` is not a user-site flag.** It only
allows modifying an externally-managed (PEP 668) install. `--user` is the
option that selects the user directory (typically `~/.local`). Without
`--user`, the destination follows the interpreter scheme and may be
`/usr/local/bin` or another prefix.

| Install | Scripts directory |
| --- | --- |
| This Cloud Agent VM (observed) | `~/.local/bin` (`site.USER_BASE`) |
| `pip install --user` | typically `~/.local/bin` |
| virtualenv | `$VIRTUAL_ENV/bin` |
| interpreter prefix (no `--user`) | `sysconfig.get_path("scripts")` (often `/usr/local/bin`) |

On this image the wrapper is `/home/ubuntu/.local/bin/whisper` after
`.cursor/install.sh`, even though that script does not pass `--user`.
Find the wrapper with `command -v whisper` rather than assuming a path.

The wrapper imported on this image is:

```python
from whisper.transcribe import cli
```

The shebang is `#!/usr/bin/python3`. The README samples assume that scripts
directory is on `PATH`. If it is not, `whisper: command not found` is an
environment problem, not a missing source file.

## Same function, two working invocations

These call `whisper.transcribe.cli()`:

| Invocation | How it is wired |
| --- | --- |
| `whisper …` | console script from `pyproject.toml` |
| `python3 -m whisper …` | [`whisper/__main__.py`](whisper/__main__.py) |

Prefer the console script or `python3 -m whisper`. The module form does not
require the scripts directory on `PATH`, only that the package is
importable. On this Cloud Agent image the interpreter is `python3` (there
is no `python` on `PATH`).

`whisper/transcribe.py` also has `if __name__ == "__main__": cli()`, but
running the file as a script fails:

```text
ImportError: attempted relative import with no known parent package
```

Do not use `python3 whisper/transcribe.py`. `python3 -m whisper.transcribe`
reaches `cli()` but emits a `RuntimeWarning` because
`whisper/__init__.py` binds `whisper.transcribe` to the `transcribe`
function, which shadows the submodule.

## Help without weights

`cli()` builds an `argparse` parser and exits on `--help` before
`load_model()`. These commands do not download checkpoints:

```bash
whisper --help
python3 -m whisper --help
```

On this image both exit 0. Isolated `XDG_CACHE_HOME` stays empty. No
`~/.cache/whisper` and no `.pt` files are created.

## Samples that pull weights

The README examples use `--model turbo` (the argparse default in
`whisper.transcribe.cli`). On a cache miss that fetches the turbo
checkpoint from `openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper`
or `~/.cache/whisper`. Do not run those samples to smoke-test a
weight-free install.

## Confirm on a Cloud Agent VM

```bash
command -v whisper          # ~/.local/bin/whisper
command -v python           # expected: not found
command -v python3          # /usr/bin/python3
whisper --help              # exit 0; default --model turbo
python3 -m whisper --help   # exit 0
```

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.
