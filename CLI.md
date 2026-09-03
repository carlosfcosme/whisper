# CLI entrypoint

The [README Command-line usage](README.md#command-line-usage) samples invoke
`whisper` as if it were already on `PATH`. That command is a setuptools
console script, not a file in this repository. This file records the
entrypoint **from the tree** so the sample environment is not implicit.

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
directory:

| Install | Scripts directory |
| --- | --- |
| Cloud Agent / `--break-system-packages` | `~/.local/bin` |
| virtualenv | `$VIRTUAL_ENV/bin` |

The wrapper imported on this image is:

```python
from whisper.transcribe import cli
```

The README samples assume that scripts directory is on `PATH`. If it is
not, `whisper: command not found` is an environment problem, not a missing
source file.

## Same function, two invocations

Both of these call `whisper.transcribe.cli()`:

| Invocation | How it is wired |
| --- | --- |
| `whisper …` | console script from `pyproject.toml` |
| `python -m whisper …` | [`whisper/__main__.py`](whisper/__main__.py) |

Prefer the console script or `python -m whisper`. The module form does not
require the scripts directory on `PATH`, only that the package is
importable. On this Cloud Agent image the interpreter is `python3` (there
is no `python` on `PATH`), so the module form is `python3 -m whisper`.

[`whisper/transcribe.py`](whisper/transcribe.py) defines `cli()` and has
`if __name__ == "__main__"`, but `python whisper/transcribe.py` is not a
working entrypoint: relative imports fail
(`ImportError: attempted relative import with no known parent package`).

## Help without weights

`cli()` builds an `argparse` parser and exits on `--help` before
`load_model()`. These commands do not download checkpoints:

```bash
whisper --help
python3 -m whisper --help
```

## Samples that pull weights

The README examples use `--model turbo` (the argparse default in
`whisper.transcribe.cli`). On a cache miss that fetches the turbo
checkpoint from `openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper`
or `~/.cache/whisper`. Do not run those samples to smoke-test a
weight-free install.

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.
