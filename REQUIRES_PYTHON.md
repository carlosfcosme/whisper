# `requires-python`

The supported Python floor is **not** the README Setup sentence that says the
codebase is “expected to be compatible with Python 3.8-3.11”. That line is
historical and does not match packaging or CI.

The machine-readable requirement is the PEP 621 field in
[`pyproject.toml`](pyproject.toml):

```toml
requires-python = ">=3.8"
```

`pip` / build backends refuse installers on Python older than 3.8. There is
**no upper bound** in this specifier. Classifiers and CI list the series that
are declared and tested today; they do not change the floor.

No model weights and no secrets belong in this file or in git.

## What the tree actually records

| Role | Value | Source |
| --- | --- | --- |
| Packaging floor | `>=3.8` | `requires-python` in [`pyproject.toml`](pyproject.toml) |
| Built metadata | `Requires-Python: >=3.8` | `openai_whisper.egg-info/PKG-INFO` (generated) |
| Declared CPython series | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 | Trove classifiers in [`pyproject.toml`](pyproject.toml) |
| Tested series | 3.8–3.13 | [`.github/workflows/test.yml`](.github/workflows/test.yml) matrix |
| Pre-commit job | 3.9 | [`.github/workflows/test.yml`](.github/workflows/test.yml) |
| Publish job | 3.8 | [`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml) |
| Train/test mention in README | 3.9.9 + PyTorch 1.10.1 | [`README.md`](README.md) Setup (historical) |

Python 3.7 is not supported. Changelog: “drop python 3.7 support” and
“update setup.py to specify python >= 3.8 requirement”
([`CHANGELOG.md`](CHANGELOG.md), v20230306 / v20230307). The `setup.py`
`python_requires` moved to `requires-python` in the PEP 621 migration.

## How to read it

`tomllib` is stdlib only on Python 3.11+. The commands below use the 3.8+
stdlib so they work on every version this field allows. They do not need
`tomli`.

From the checkout (`pyproject.toml`):

```bash
python3 -c "import re; print(re.search(r'(?m)^requires-python = \"([^\"]+)\"', open('pyproject.toml', encoding='utf-8').read()).group(1))"
```

From an installed `openai-whisper` (`importlib.metadata` is stdlib on 3.8+):

```bash
python3 -c "from importlib.metadata import metadata; print(metadata('openai-whisper')['Requires-Python'])"
```

Both must print `>=3.8`. Do not treat the README “3.8-3.11” range, the
classifiers, or a Cloud Agent interpreter as a replacement for this field.

## Tests (no weights)

Skip `test_transcribe` so the suite does not download checkpoints:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

Do not commit `.pt` files. No credentials are required to read
`requires-python` or run the weight-free tests.
