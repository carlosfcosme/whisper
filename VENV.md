# Virtualenv

This checkout does not use a Python virtualenv. `.cursor/install.sh`
installs with system `pip` and `--break-system-packages` (user site).

No model weights and no secrets belong in this file or in git.

## What is present

On a Cloud Agent image the interpreter is system `/usr/bin/python3`.
`VIRTUAL_ENV` is unset. `sys.prefix == sys.base_prefix` (`/usr`), so
the process is not inside a venv.

Wheels from the install script land in the user site
(`~/.local/lib/python3.12/site-packages` on CPython 3.12). Entry points
(`whisper`, `pytest`, linters) land on `~/.local/bin`.

## What is absent

| Signal | In this tree / image? |
| --- | --- |
| `venv/`, `.venv/`, or `pyvenv.cfg` | No |
| `python3 -m venv` in [`.cursor/install.sh`](.cursor/install.sh) | No |
| `Pipfile`, Conda `environment.yml` | No |
| `VIRTUAL_ENV` | Unset |
| `python3-venv` / `python3.12-venv` (apt) | Not installed |
| `ensurepip` | Missing (Debian split) |

GitHub Actions creates an ephemeral Conda env named `test` in
[`.github/workflows/test.yml`](.github/workflows/test.yml). That env is
not checked in and is not a `venv`.

## Why `python3 -m venv` fails here

The stdlib `venv` module imports, but creating an environment fails
because `ensurepip` is not available:

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv
```

Do not add a venv step to `.cursor/install.sh` without also installing
`python3-venv`. The current install path is intentional: system `pip`
plus `--break-system-packages`.

## Confirm (no weights)

```bash
echo "${VIRTUAL_ENV-}"   # expected: empty
python3 -c "import sys; assert sys.prefix == sys.base_prefix, (sys.prefix, sys.base_prefix)"
test ! -e pyvenv.cfg && test ! -d venv && test ! -d .venv
python3 -c "import whisper, torch; print(whisper.__version__, torch.__version__)"
```

The last line only imports. It does not call `whisper.load_model` and
does not download checkpoints.

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.
