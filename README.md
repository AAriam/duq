# duq — dimensions, units, quantities

[![CI](https://github.com/AAriam/duq/actions/workflows/ci.yml/badge.svg)](https://github.com/AAriam/duq/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AAriam/duq/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/AAriam/duq/blob/main/pyproject.toml)

<!-- --8<-- [start:overview] -->
**duq** is a lean, rigorously-typed library for physical **d**imensions, **u**nits, and
**q**uantities, built for scientific computing. A pure-Python dimensional core carries
units through **unit-carrying NumPy arrays** *and* **jit/grad/vmap-safe JAX arrays** — the
same `Quantity` semantics on both back-ends. Units compose **as entered** (`kJ/mol` stays
`kJ/mol`, never silently collapsed to base units), exponents are **exact** `Fraction`s
(`L^3/2` is exact, never `1.4999…`), conversions are **fail-loud** (a dimensional mismatch
raises, units are never silently dropped), it ships **CODATA-2022-versioned** constants, and
it offers first-class **dimensional analysis** and opt-in **molar equivalence** (`kJ/mol ↔ J`
via *N*<sub>A</sub>) — a comp-chem staple.

`import duq` imports **no** array library, so it stays fast and never requires NumPy or JAX at
import time; the JAX back-end lives behind the optional `duq[jax]` extra.

## Install

```sh
pip install duq            # core + NumPy support
pip install "duq[jax]"     # adds the JAX back-end (jax, quax, equinox)
```

Requires Python ≥ 3.11 and `numpy ≥ 1.26`. JAX is strictly optional.

## Quickstart

**Scalars** — dimension-checked arithmetic, as-entered units, exact conversion:

```python
import duq

bond = duq.Quantity(1.0, "kJ/mol")
print(bond.to("eV/mol"))            # 6.241509074460763e+21 eV·mol⁻¹
print(bond.to("J", equivalence="molar"))   # 1.66…e-21 J   (opt-in, via N_A)

r = duq.Quantity(2.0, "m")
t = duq.Quantity(4.0, "s")
print(r / t)                        # 0.5 m·s⁻¹
duq.Quantity(1.0, "m") + duq.Quantity(1.0, "s")  # raises DimensionalityError
```

**NumPy** — units ride through ufuncs, reductions and broadcasting (NEP 13 / NEP 18):

```python
import numpy as np
import duq

d = np.array([1.0, 2.0, 3.0]) * duq.units.km
t = np.array([0.5, 1.0, 1.5]) * duq.units.h
v = (d / t).to("m/s")
print(v.mean())                     # 0.5555555555555556 m·s⁻¹
print(np.sqrt(d * d).unit)          # km
```

**JAX** — the same quantity, safe under `jit`, `grad` and `vmap`:

```python
import duq.jax

def kinetic(v):                     # v is a velocity quantity
    return 0.5 * duq.jax.Quantity(2.0, "kg") * v * v

g = duq.jax.grad(kinetic)(duq.jax.Quantity(3.0, "m/s"))
print(g.value, g.unit)              # 6.0  kg·m·s⁻¹   (= d(energy)/d(velocity))
print(g.unit == duq.unit("J") / duq.unit("m/s"))   # True
```

## Why duq?

duq is **not** a general-purpose pint competitor. It is deliberately small and opinionated,
and trades catalog breadth for JAX support, exactness and fail-loud safety. If you need a
huge unit catalog, imperial/US-customary units, or pandas integration today, **pint**,
**unyt** or **astropy.units** are excellent and far more mature choices.

| Feature | duq | pint | unyt | astropy.units |
|---|---|---|---|---|
| NumPy arrays | ✅ NEP 13/18 wrapper | ✅ | ✅ ndarray subclass | ✅ ndarray subclass |
| JAX `jit`/`grad`/`vmap` | ✅ native pytree | ❌ | ❌ | ❌ |
| Fail-loud (never silently drops units) | ✅ everywhere | ⚠️ partial | ⚠️ partial | ⚠️ partial |
| Exact exponents | ✅ `Fraction` | ❌ float | ❌ float | ❌ float |
| As-entered composition | ✅ | ✅ | ➖ base-collapsed | ✅ |
| Molar equivalence (`kJ/mol ↔ J`) | ✅ opt-in, built-in | ⚠️ via contexts | ❌ | ⚠️ via equivalencies |
| Versioned constants (CODATA) | ✅ 2022 | ➖ | ➖ | ✅ |
| Catalog size | 🟡 small, curated | 🟢 very large | 🟢 large | 🟢 large |
<!-- --8<-- [end:overview] -->

## Documentation

The full documentation site is built with MkDocs from the `docs/` tree:

- **Learn** — quickstart, units & dimensions, quantities, NumPy arrays, JAX, dimensional
  analysis, constants, and a generated unit-catalog reference.
- **API reference** — generated from the numpydoc-style docstrings.
- **Design & research** — [`docs/dev/design.md`](docs/dev/design.md),
  [`docs/dev/research_numpy_jax_units.md`](docs/dev/research_numpy_jax_units.md), and the
  test-enforced [`docs/dev/jax_coverage.md`](docs/dev/jax_coverage.md).

Runnable notebooks live in [`docs/notebooks/`](docs/notebooks/).

## Development

duq uses [pixi](https://pixi.sh) to manage every dev/test/CI environment from the single
`pyproject.toml`. After [installing pixi](https://pixi.sh/latest/#installation):

```sh
git clone https://github.com/AAriam/duq.git
cd duq
pixi run -e test test          # run the test suite
```

Common tasks (each runs in a dedicated, reproducible environment):

| Command | What it does |
|---|---|
| `pixi run -e lint lint` | Ruff lint |
| `pixi run -e lint fmt-check` | Ruff format check (`fmt` to apply) |
| `pixi run -e type typecheck` | mypy `--strict` |
| `pixi run -e test test-cov` | pytest with coverage (gate ≥ 95%) |
| `pixi run -e test-jax test-jax` | full suite with JAX (duq/jax gate ≥ 90%) |
| `pixi run -e test-np126 test` | tests against NumPy 1.26 |
| `pixi run -e docs docs-build` | build the docs site (`--strict`) |
| `pixi run -e docs docs-serve` | live-preview the docs |

**Contributing.** See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and
[`RELEASING.md`](RELEASING.md) for the release checklist. Design rationale lives in
[`docs/dev/design.md`](docs/dev/design.md). Changes land through stacked, conventional-commit
PRs; all gates above must be green in CI.

## License

[MIT](LICENSE) © Armin Ariamajd
