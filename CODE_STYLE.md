# Code style: Black is present, Ruff is not

This checkout formats Python with **Black**. **Ruff is not part of the
project** and is not installed in the Cloud Agent environment.

There is no `ruff.toml`, `.ruff.toml`, or `[tool.ruff]` table. `ruff` is
not listed under `[project.optional-dependencies] dev` in `pyproject.toml`,
and `.pre-commit-config.yaml` has no Ruff hook. Do not run `ruff format`
or `ruff check` unless those files change.

## What the environment actually installs

`.cursor/install.sh` runs `pip install -e ".[dev]"`. The `dev` extra is:

`black`, `flake8`, `isort`, `pytest`, `scipy`

That is why `black`, `isort`, and `flake8` are on `PATH` after setup, and
why `ruff` is not.

## Format

```bash
black .
isort .
```

`[tool.black]` in `pyproject.toml` is empty, so Black uses its defaults
(line length 88). isort reads `[tool.isort]` (`profile = "black"`,
`line_length = 88`, trailing commas, `multi_line_output = 3`).

## Check

```bash
black --check .
isort --check-only .
flake8 --max-line-length 88 --ignore E203,E501,W503,W504
```

The flake8 flags match `.pre-commit-config.yaml`. `.flake8` only adds
`F401` ignores for `*/__init__.py`.

GitHub Actions runs the same stack through pre-commit (not installed by
`.cursor/install.sh`; CI installs it in the `pre-commit` job):

```bash
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

## Config inventory

| Tool | Present? | Config |
|------|----------|--------|
| Black | Yes | `pyproject.toml` `[tool.black]`; pre-commit hook `psf/black` |
| isort | Yes | `pyproject.toml` `[tool.isort]`; pre-commit hook |
| flake8 | Yes | `.flake8` plus pre-commit `--max-line-length 88` |
| Ruff | No | none |

## Weights and secrets

These commands do not download Whisper model weights and do not need
credentials. Do not commit `.pt` files or secrets.
