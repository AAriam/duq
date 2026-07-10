# Contributing to duq

Thanks for your interest in duq! This is a short guide to the dev setup, the
quality gates, and how changes land.

## Development setup

duq uses [pixi](https://pixi.sh) to manage every dev/test/CI environment from the
single `pyproject.toml`. After [installing pixi](https://pixi.sh/latest/#installation):

```sh
git clone https://github.com/AAriam/duq.git
cd duq
pixi run -e test test          # run the test suite
```

Each task runs in its own reproducible, locked environment — you never manage a
virtualenv by hand.

## Quality gates

Every one of these must be green before a change merges (CI enforces them):

| Command | Gate |
|---|---|
| `pixi run -e lint lint` | Ruff lint |
| `pixi run -e lint fmt-check` | Ruff format check (`pixi run -e lint fmt` to apply) |
| `pixi run -e type typecheck` | mypy `--strict` |
| `pixi run -e test test-cov` | pytest + coverage ≥ 95% |
| `pixi run -e test-jax test-jax` | full suite with JAX; `src/duq/jax` coverage ≥ 90% |
| `pixi run -e test-np126 test` | tests against NumPy 1.26 |
| `pixi run -e docs docs-build` | `mkdocs build --strict` |

Notebooks under `docs/notebooks/` are executed manually (not in CI) in the
`test-jax` environment; if you touch the public API, re-run them so their outputs
stay in sync:

```sh
for nb in docs/notebooks/*.ipynb; do
  pixi run -e test-jax jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## Coding conventions

- **Fail loud.** duq never silently drops a unit or guesses a conversion; an
  unsupported operation raises a `DuqError` subclass naming the operation.
- **Typed and documented.** Code is type-annotated (mypy `--strict`, `py.typed`)
  and public functions carry numpydoc-style docstrings; **every docstring example
  must be executed and verified** — no aspirational code.
- **Pure core.** `import duq` and scalar arithmetic must not import NumPy or JAX;
  array back-ends are imported lazily. CI asserts this.
- The JAX unit rules are registered per `lax` primitive and documented in the
  test-enforced `docs/dev/jax_coverage.md` (regenerate it with
  `pixi run -e test-jax python docs/dev/generate_jax_coverage.py`).

## Pull requests

- Branch from the appropriate base and open a PR; large work lands as **stacked**
  PRs (e.g. `feat/core` → `feat/numpy` → `feat/jax` → `docs/overhaul`).
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit and
  PR titles (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, …).
- Update `CHANGELOG.md` (the `Unreleased` section) for any user-visible change.
- Keep PRs focused; all gates above must pass in CI.

## Where the specs live

The design rationale and technical due-diligence for the rewrite live in
[`docs/dev/design.md`](docs/dev/design.md) and
[`docs/dev/research_numpy_jax_units.md`](docs/dev/research_numpy_jax_units.md).
Release steps are in [`RELEASING.md`](RELEASING.md).
