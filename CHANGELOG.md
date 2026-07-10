# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **JAX integration — jit/grad/vmap-safe unit-carrying arrays (`duq[jax]`,
  quax primitive interception):**
  - New `duq.jax.Quantity`: a `quax.ArrayValue` (equinox pytree) whose
    magnitude is the traced leaf and whose unit is a **static** field, so the
    unit is checked and propagated at trace time, participates in `jit`'s
    cache key (same unit → cached; new unit → retrace), and costs nothing at
    runtime. Construction from anything `jnp.asarray` accepts, plus
    `Quantity.from_core(q)`/`q.to_core()` to cross between the NumPy core and
    JAX; the core `duq.Quantity` now rejects JAX arrays/tracers with a pointer
    to `duq.jax.Quantity` (without importing JAX).
  - 93 `lax` primitives registered through the shared `duq._rules` engine —
    arithmetic with the core affine (`°C`) policy, unit powers
    (`integer_pow`/`sqrt`/`rsqrt`/`cbrt`/`square`), dimensionless/angle rules
    (`%` and `deg` rescaled at trace time), converted comparisons
    (`==`/`!=` on incompatible dimensions → all-False/all-True, matching the
    NumPy layer), n-ary structure (`concatenate`/`stack`/`select_n`/`sort`/
    `split`/`pad`/`gather`/`scatter*`), reductions (`reduce_prod` →
    `unit ** n` via static shapes) and `dot_general` — see
    `docs/dev/jax_coverage.md` (generated, test-enforced). Control flow
    (`scan`/`while`/`cond`) keeps units in carries and branches via quax.
  - Fail-loud everywhere: `materialise()` refuses, and any uncovered
    primitive raises `UnsupportedOperationError` **naming the primitive**;
    `np.asarray(q)`/`float(q)`-style coercions follow the core policy
    (dimensionless converts, dimensional raises with `ustrip` guidance).
  - `duq.jax.numpy`: a lazily quaxified mirror of `jax.numpy` (including
    proxied submodules such as `linalg` and curated unit-aware
    `deg2rad`/`rad2deg` overrides), plus full Python operator support and a
    unit-aware `.at[...]` indexed-update helper on the quantity itself.
  - Unit-aware transformation wrappers: `duq.jax.grad`/`jacfwd`/`jacrev`/
    `hessian` label derivatives with the exact `unit_out / unit_in` unit
    (Hessians with `unit_out / unit_in²`); `duq.jax.jit`/`vmap` document the
    pytree pathway (plain `jax.jit`/`jax.vmap` work directly on quantities).
  - `duq.uconvert`/`duq.ustrip` now dispatch structurally on any
    quantity flavour via the new `duq.QuantityLike` protocol.
  - Packaging: the `jax` extra pins `quax>=0.3.6,<0.5` (single-maintainer
    risk) and `equinox>=0.11`; the pixi `test-jax` environment gets quax from
    PyPI (not on conda-forge) and CI enforces a dedicated ≥90% coverage gate
    on `src/duq/jax` (the global 95% gate covers everything else).
    `import duq` still never imports JAX, and `import duq.jax` without the
    extra raises a helpful `ImportError` (both asserted by tests).
- **NumPy integration — unit-carrying arrays (NEP 13 / NEP 18):**
  - The existing `Quantity` now also wraps a NumPy array (or NumPy scalar) as its
    magnitude (`Quantity(np.array(...), unit)`, or `Quantity.from_array([...],
    unit)` for lists); the array is stored as-is, with no copy.
  - `Quantity.__array_ufunc__` and `__array_function__` carry units through a
    curated ufunc table (~70 ufuncs) and function table (~130 functions), driven
    by the shared `duq._rules` engine: `multiply`/`matmul`→product,
    `divide`/`floor_divide`→quotient, `add`/`subtract`/`hypot`/`maximum`/… →
    same-dimension (right operand converted to the left unit), `power`/`sqrt`/
    `cbrt`/`square`/`reciprocal`→unit powers, `exp`/`log`/trig→dimensionless
    (angle and `%` rescaled), inverse trig→radian-labelled, comparisons→plain
    bool arrays, `var`→unit², `gradient`/`trapezoid`→unit division/multiplication,
    and much more.
  - Fail-loud everywhere: any uncovered ufunc/function, any `out=` argument, and
    `multiply.reduce` raise `UnsupportedOperationError` naming the operation;
    units are never silently dropped. `np.array(q)`/`np.asarray(q)` raise with
    guidance to use `duq.ustrip`, closing the classic silent-strip hole.
    `float(q)`/`int(q)`/`complex(q)`/`bool(q)` coerce **dimensionless**
    quantities by applying the unit scale (`float(Quantity(50.0, "%")) == 0.5`;
    radian-labelled values pass through) and raise for anything dimensional —
    pint-consistent behaviour.
  - `Unit` opts out of NumPy's ufunc machinery (`__array_ufunc__ = None`) and
    scales into a `Quantity` when combined with a number or array, so
    `np.linspace(0, 1, 5) * duq.units.m`, `5 * duq.units.m`, `duq.units.m * 5`
    and `array / unit` all yield a `Quantity` (`1 / unit` still inverts the unit).
  - Array `Quantity` gains `.shape`/`.ndim`/`.size`/`.dtype`/`.T`, `len()`,
    iteration and indexing (yielding element quantities), `.item()`, and thin
    `.sum()/.mean()/.std()/.var()/.min()/.max()/.reshape()/.ravel()/.astype()`
    delegates that route through the same unit rules.
  - New `duq._numpy` back-end package, imported lazily only from the array hooks,
    so `import duq` and all scalar arithmetic remain NumPy-free (asserted by CI).
  - A pixi `test-np126` environment (Python 3.11, NumPy 1.26) plus an Ubuntu CI
    job guard the NumPy 1.26 ↔ 2.x differences (including the `trapz`/`trapezoid`
    rename).
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
