# Releasing duq

This is the maintainer checklist for cutting a release. The repository ships with
`version = "1.0.0.dev0"`; nothing here is automated, and **no CI job publishes** —
uploading to PyPI is a deliberate, manual step.

All commands run through [pixi](https://pixi.sh); the `build` environment carries
`python-build` and `twine`.

## 1. Pre-flight: all gates green

```sh
pixi run -e lint lint
pixi run -e lint fmt-check
pixi run -e type typecheck
pixi run -e test test-cov            # coverage gate ≥ 95%
pixi run -e test-jax test-jax        # duq/jax coverage gate ≥ 90%
pixi run -e test-np126 test          # NumPy 1.26
pixi run -e docs docs-build          # mkdocs build --strict
```

Re-execute the notebooks if the public API changed:

```sh
for nb in docs/notebooks/*.ipynb; do
  pixi run -e test-jax jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## 2. Set the release version

Edit `pyproject.toml`:

```toml
version = "1.0.0a1"      # from 1.0.0.dev0
```

(`hatchling` reads the version from `pyproject.toml`; `duq.__version__` resolves
it from the installed package metadata.)

## 3. Finalise the changelog

In `CHANGELOG.md`, rename the top heading to the concrete version and date, e.g.

```md
## [1.0.0a1] — 2026-07-15
```

and add/refresh the comparison link references at the bottom of the file.

## 4. Build and verify the artifacts

```sh
pixi run -e build build          # -> dist/duq-1.0.0a1.tar.gz and *.whl
pixi run -e build twine-check    # twine check dist/*   (must PASS)
```

Sanity-check the wheel contents before uploading — it must include `duq/py.typed`
and the `duq/data/*.toml` catalogs, and must **not** contain tests or `.pyc`:

```sh
python -m zipfile -l dist/duq-1.0.0a1-py3-none-any.whl
```

Optionally smoke-test the wheel in a throwaway environment:

```sh
pixi exec --spec python=3.11 -- bash -c '
  python -m venv /tmp/duq-smoke && . /tmp/duq-smoke/bin/activate &&
  pip install dist/duq-1.0.0a1-py3-none-any.whl &&
  python -c "import duq; print(duq.__version__, duq.Quantity(1.0, \"kJ/mol\"))"'
```

## 5. Tag

```sh
git commit -am "release: v1.0.0a1"
git tag -a v1.0.0a1 -m "duq 1.0.0a1"
git push origin main --tags
```

## 6. Publish

Upload to TestPyPI first, verify a clean install, then upload to PyPI:

```sh
pixi run -e build twine upload --repository testpypi dist/*
pixi run -e build twine upload dist/*
```

(Use a PyPI API token; configure it in `~/.pypirc` or the `TWINE_*` env vars.)

## 7. Post-release: bump back to a dev version

Set the next development version so `main` never advertises a released version:

```toml
version = "1.0.0.dev1"     # or the next planned pre-release, e.g. 1.0.0a2.dev0
```

Add a fresh `## [Unreleased]` section to `CHANGELOG.md`, commit, and push.

## Versioning

duq follows [Semantic Versioning](https://semver.org/). Pre-1.0-final releases
use PEP 440 pre-release suffixes (`a`/`b`/`rc`); the public API may still change
between alphas. Breaking changes are called out in the changelog's **Removed** /
**Changed** sections.
