# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Complete rewrite of the core as a pure-Python, array-free layer:
  - `Dimension`: an immutable, hashable vector of exact `Fraction` exponents
    over the seven SI base dimensions, with `*`/`/`/`**` algebra and a
    string parser accepting base symbols/names and named derived dimensions.
  - `Unit` and `UnitRegistry`: an as-entered unit model (`kJ/mol` stays
    `kJ/mol`), TOML-backed catalogs, lazy longest-match SI-prefix resolution
    (`km` is never pre-generated), an enforced affine-unit policy for `°C`/`°F`
    with explicit delta units, and refusal to combine units across registries.
  - `Quantity`: an immutable scalar (int/float/complex/`Fraction`/`Decimal`)
    with dimension-checked arithmetic, exact equality after conversion, a
    hash/`==` contract, `to`/`to_si`/`value_in`/`compact`/`allclose`, and
    opt-in molar equivalence (`q.to("J", equivalence="molar")`).
  - One parser understands both dot (`kg.m^2.s^-2`) and star
    (`kg*m**2/s**2`, `kJ/mol`) syntaxes, exact fractional exponents and
    Unicode superscripts; formatters render unicode/plain/latex.
  - A declarative `_rules` engine holds the dimensional rules shared by the
    forthcoming NumPy and JAX layers.
  - `duq.constants`: CODATA-versioned constants as `Quantity` objects
    (`duq.constants.k_B`, `duq.constants.codata(2022)`), every value verified
    against the NIST CODATA 2022 recommended values.
  - `duq.analysis`: `name_of`, `decompose`, `compositions` and
    `shortest_composition` (which returns an exact result or raises, never a
    silently-wrong one); NumPy is imported lazily only for the search.
  - TOML catalogs (`prefixes`, `dimensions`, `units`, `constants`) plus a
    `DuqError` hierarchy, `duq.units`/`duq.dims` namespaces, and free
    functions `duq.uconvert`/`duq.ustrip`; `import duq` never imports NumPy.

### Removed

- The pre-1.0 prototype API (`duq.dimension`, `duq.unit`, `duq.quantity`,
  `duq.helpers`, the `duq.data.*` Python modules and their `predefined`
  containers) is removed outright, without a deprecation cycle, as the package
  was an unreleased prototype.

### Changed

- Relicensed the project from AGPL-3.0 to the MIT License.
- Restructured the repository to a `src/` layout: the package now lives in
  `src/duq/` and the test suite in `tests/`.
- Modernized the development toolchain:
  - a single root `pyproject.toml` that serves as both the packaging manifest
    (PEP 621/639, [hatchling](https://hatch.pypa.io) build backend) and the
    [pixi](https://pixi.sh) workspace manifest;
  - [Ruff](https://docs.astral.sh/ruff/) for linting and formatting;
  - [mypy](https://mypy-lang.org/) for static type checking;
  - a consolidated GitHub Actions CI workflow driven by pixi;
  - a `.pre-commit-config.yaml` with Ruff and standard hygiene hooks.
