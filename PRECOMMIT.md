# Pre-commit

Status: **present** (not none).

This repository ships [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
GitHub Actions runs it as the `pre-commit` job in
[`.github/workflows/test.yml`](.github/workflows/test.yml). That was implicit
in the Cloud Agent environment: `.cursor/install.sh` installs the hook CLIs
via `.[dev]` (`black`, `isort`, `flake8`) but does **not** install the
`pre-commit` runner. `pre-commit` is also absent from
`[project.optional-dependencies] dev` in [`pyproject.toml`](pyproject.toml).

## Inventory

| Item | Present? |
|------|----------|
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Yes |
| CI `pre-commit` job | Yes ([`test.yml`](.github/workflows/test.yml)) |
| `pre-commit` package in `.[dev]` | No — install separately |
| `.cursor/install.sh` installs `pre-commit` | No |

## Hooks

Pinned in [`.pre-commit-config.yaml`](.pre-commit-config.yaml):

| Repo | Rev | Hook |
|------|-----|------|
| `pre-commit/pre-commit-hooks` | v5.0.0 | `check-json` |
| | | `end-of-file-fixer` (python files) |
| | | `trailing-whitespace` (python files) |
| | | `mixed-line-ending` |
| | | `check-added-large-files` (`--maxkb=4096`) |
| `psf/black` | 25.1.0 | `black` |
| `pycqa/isort` | 6.0.0 | `isort` (`--profile black -l 88 --trailing-comma --multi-line 3`) |
| `pycqa/flake8` | 7.1.1 | `flake8` (`--max-line-length 88 --ignore E203,E501,W503,W504`) |

## Run (same as CI)

```bash
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

On the Cloud Agent system Python, add `--break-system-packages` to the
`pip install` if site-packages is not writable. `pre-commit install` is
refused when `core.hooksPath` is set (this VM); `pre-commit run --all-files`
still installs hook environments and runs the same checks.

After Cloud Agent install, the same formatters can be invoked without the
runner (`black`, `isort`, `flake8`). That is not a substitute for the CI job.

## Weights and secrets

`pre-commit` does not download Whisper checkpoints and does not need API
keys or credentials. Do not commit `.pt` files;
`check-added-large-files` rejects additions over 4096 KB.
