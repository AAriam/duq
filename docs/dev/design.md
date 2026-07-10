# duq v1.0 — Design Document

Status: final (v1)
Date: 2026-07-10
Companion: `research_numpy_jax_units.md` (technical due-diligence underpinning §3.5)

## 1. Goals and positioning

duq is a **lean, rigorously-typed library for physical dimensions, units, and quantities**,
built for scientific computing. It is *not* a general-purpose pint competitor. Its
differentiators:

1. **Unit-carrying arrays for both NumPy and JAX** — units propagate through ufuncs,
   reductions, and broadcasting; the JAX quantity is safe under `jit`/`grad`/`vmap`.
2. **As-entered composition** — `kJ/mol` stays `kJ/mol` in display and arithmetic; it is
   never silently collapsed to base units (canonical forms are available on demand).
3. **Dimensional analysis as a first-class feature** — naming a dimension, decomposing it,
   and searching equivalent compositions.
4. **Molar equivalence** — explicit, opt-in conversion between per-amount and absolute
   quantities via N_A (kJ/mol ↔ J), a comp-chem staple pint doesn't ship.

Non-goals: imperial/US-customary catalogs, currency, arbitrary runtime unit systems,
pandas integration (all possible later, none in v1).

## 2. Constraints (client-approved)

- License: MIT. Python ≥ 3.11. `numpy ≥ 1.26` required; `jax` strictly optional (`duq[jax]`).
- Delivery: staged PRs — 1 scaffolding, 2 core, 3 numpy, 4 jax, 5 docs/polish. No PyPI publish.

## 3. Architecture: four layers

```
duq.core        pure Python, no numpy import — Dimension, Unit, UnitRegistry, scalar Quantity
duq.analysis    dimensional-analysis extras (naming, composition search) over duq.core
duq.numpy       unit-carrying numpy arrays (wrapper, __array_ufunc__/__array_function__)
duq.jax         unit-carrying jax arrays (pytree; mechanism per research report)
duq/data/*.toml declarative catalogs loaded via stdlib tomllib
```

The top-level `duq` namespace re-exports the public API; `duq.numpy`/`duq.jax` import
their backends lazily so `import duq` stays fast and never requires jax.

### 3.1 Dimension (core)

Immutable, hashable mapping of the 7 SI base dimensions to `fractions.Fraction`
exponents (exact arithmetic; `L^3/2` is exact, never 1.4999…).

- Constructors: `Dimension.parse("M.L^2.T^-2")`, from mapping, and module-level
  singletons for the base dimensions.
- Algebra: `*`, `/`, `**` (int/Fraction powers); `==`/`hash`; `.is_dimensionless`.
- Follows Python protocols correctly: foreign types get `NotImplemented`, never a raise
  from dunders; errors are `TypeError`/`ValueError` subclasses, not `NotImplementedError`.

The prototype's mixing of "vector over all *named* dimensions" into `Dimension` is
dropped: `Dimension` is canonical base-7 only. Named-dimension features move to
`duq.analysis` (see 3.4). As-entered preservation lives in `Unit`, where it belongs
(that is where users actually enter compositions).

### 3.2 Unit and UnitRegistry (core)

Two-level model:

- **Unit atom** (registry entry): name, symbol, aliases, `Dimension`, scale-to-coherent-SI,
  optional affine offset, prefixable flag.
- **`Unit`**: immutable composition `{(prefix, atom): Fraction}` preserving as-entered
  form. Cached derived data: `dimension`, `scale`, `offset`. Canonicalizers:
  `.to_coherent_si()`, plus display in as-entered form by default.

Key policies:

- **Prefixes are data, not units.** `km` is resolved at parse time (longest-match against
  prefixable atoms); the catalog never eagerly generates prefix×unit combinations. This
  fixes the prototype's arbitrary catalog gaps (cm existed, km didn't).
- **Affine units (°C, °F)** are legal only as a bare atom with exponent 1 and no
  companions; anything else raises `AffineUnitError`. Delta atoms (`Δ°C`) are provided
  for differences. Conversion math keeps shift and factor as separate fields — the
  prototype's conflation of both into one `conv_factor` column (kelvin: `0`!) is gone.
- **Registry pattern**: `UnitRegistry` builds from the TOML catalogs; a module-level
  default registry backs `duq.unit("kJ/mol")` and predefined attribute access
  (`duq.units.kJ`, generated with correct laziness — the prototype's predefined
  containers were broken and shared mutable state). `registry.define(...)` allows user
  extension. Units from different registries don't mix (pint's hard-learned lesson).
- **Parsing**: accepts the prototype's dot syntax (`kg.m^2.s^-2`) *and* the common
  Python-ish syntax (`kg*m**2/s**2`, `kJ/mol`). Small hand-written tokenizer, exact
  Fraction exponents, clear `UnitParseError` messages.

### 3.3 Quantity (core, scalar)

Immutable `Quantity(value, unit)` for Python scalars (int/float/complex/Fraction/Decimal).

- `.value`, `.unit`, `.dimension`; `.to(unit)`, `.to_si()`, `.value_in(unit)`.
- Arithmetic with dimension checking; addition auto-converts compatible units to the
  left operand's unit; incompatible dimensions raise `DimensionalityError`.
- Comparisons convert; equality is **exact** (the prototype's `np.isclose`-based `==`
  violated transitivity and the hash contract); an `math.isclose`-style helper
  `Quantity.allclose` handles tolerance explicitly.
- Molar equivalence is **opt-in**: `q.to("J", equivalence="molar")`. The prototype
  silently treated every dimension pair differing by N^k as convertible — dangerous.
- The prototype's `normalization` feature (auto SI-prefix scaling of the value) is
  replaced by an explicit, working `q.compact()` (returns new Quantity; nothing mutates).

Everything in core is immutable: no `inplace=` parameters, no `__imul__` mutation, and
all objects are hashable.

### 3.4 duq.analysis

Home of the prototype's genuinely novel features, rebuilt on the clean core:

- `name_of(dim)` → "energy" (catalog lookup).
- `decompose(dim)` → base decomposition views.
- `compositions(dim, max_terms=, max_exp=)` → equivalent compositions from named
  dimensions (the linear-algebra search; numpy allowed here, imported lazily).
- `shortest_composition(dim)` → deterministic exact search (the prototype's greedy loop
  could silently return wrong results at its iteration cap; the new one either solves it
  exactly or raises).

### 3.5 duq.numpy and duq.jax (finalized per research report)

There is no single mechanism that units-enables both backends — the array API standard
explicitly excludes subclassing and cross-library dispatch from its scope. So: one pure
core, two thin interception front-ends. The **unit-rule logic** (what unit does `mul`
produce; add requires same dimension; `exp`/`log`/trig require dimensionless/angle)
lives once in the core as a declarative rule table keyed by abstract operation; both
front-ends call it.

**NumPy** — wrapper, never an ndarray subclass:

- `Quantity` implements `__array_ufunc__` (NEP 13) + `__array_function__` (NEP 18),
  pint-style. Subclassing is rejected: it pays the `__array_wrap__` 2.x churn,
  view-casting and `out=` taxes, silently strips units in uncovered code paths, and
  can never share a conceptual Quantity with a JAX magnitude.
- Fail-loud: any ufunc/function without a registered unit rule raises
  `UnsupportedOperationError`; units are never silently dropped.
- One `Quantity` class serves scalars and numpy arrays: Python-scalar arithmetic runs a
  pure-Python path (no numpy import); array magnitudes route through the ufunc
  machinery. CI asserts `import duq` + scalar arithmetic never imports numpy.
- `out=`, `__matmul__`, reductions, `np.concatenate`/`stack`/`where` get explicit
  entries in the function table (the historic subclass trouble spots).

**JAX** (`duq[jax]` extra) — quax primitive interception, the unxt-proven route:

- `jax.numpy` never consults NEP 13/18; dispatch must happen post-tracing on `lax`
  primitives. `duq.jax.Quantity` is a `quax.ArrayValue` (equinox Module → pytree):
  magnitude is the traced leaf, unit is a **static** field (aux data).
- Unit rules are registered per `lax` primitive (`mul_p`, `add_p`, `dot_general_p`,
  `concatenate_p`, `select_n_p`, `reduce_sum_p`, `integer_pow_p`, control-flow
  `scan_p`/`while_p`/`cond_p`, …) — a few dozen primitives cover most of `jnp`,
  versus enumerating hundreds of functions jpu-style. `materialise()` raises, so
  unhandled primitives fail loudly.
- `jit`: unit is part of the treedef → distinct units retrace. Standard, accepted
  practice (unxt, jpu). Documented guidance: `ustrip` to a canonical unit before hot
  kernels.
- `grad`/`vmap`: gradient units are `unit_out / unit_in` and emerge naturally from the
  unit-carrying forward pass; tests assert this explicitly. Known sharp edges (quax
  `custom_vjp` gap, grad scalar-check) are documented with `ustrip`-based escape
  hatches.
- Dependency policy: quax is single-maintainer, 0.3.x — **pin it**, isolate it entirely
  behind `duq/jax/`, and keep a curated-namespace fallback (jpu-style) as the
  documented plan B. A `duq.jax.numpy` convenience namespace (quaxified) spares users
  manual `quaxify` calls.

**Shared semantics:** free functions `duq.uconvert(q, unit)` and `duq.ustrip(q, unit)`
work on every Quantity flavor (JAX-idiomatic, copied from unxt). One conformance test
suite runs parametrically against scalar/numpy/jax backends asserting identical
magnitudes and units.

## 4. Data catalogs (TOML)

`src/duq/data/*.toml`, loaded once at first registry construction with stdlib `tomllib`
(no new dependency; human-editable with comments — the reason TOML beats JSON here, and
it needs no third-party parser, which is why it beats YAML).

- `prefixes.toml` — all SI prefixes incl. 2022 additions (quetta…quecto).
- `dimensions.toml` — base dimensions + named derived dimensions (for analysis/naming).
- `units.toml` — curated catalog: SI base + all 22 named derived units; time
  (s…, min, h, day, year-julian); length (m, Å, bohr, au); mass (kg…, Da, m_e, t);
  energy (J, eV, hartree, cal, kcal, erg); pressure (Pa, bar, atm, torr, mmHg);
  volume (L); angle (rad, deg, arcmin, arcsec, sr); temperature (K, °C, Δ°C);
  charge (e); magnetic field (T, G); molarity (M). Scales stored as exact decimal
  strings where the definition is exact.
- `constants.toml` — CODATA-**versioned**: each constant carries value, unit,
  uncertainty, exactness flag, and source year; default set CODATA 2022, selectable
  (scipy/astropy both learned this lesson). Exposed as `duq.constants.k_B` → Quantity.
- `%` and `ppm` ship as scaled dimensionless units; radian is
  dimensionless-but-labelled with an explicit drop/attach rule.

A schema-validation test asserts every entry is well-formed and every scale/offset
round-trips.

## 5. Errors

`DuqError` root; `UnitParseError(DuqError, ValueError)`,
`DimensionalityError(DuqError, ValueError)`, `AffineUnitError(DimensionalityError)`,
`UndefinedUnitError(DuqError, KeyError)`.

## 6. Quality gates

- ruff (lint+format, line 99, numpy docstring convention), mypy `--strict`, `py.typed`.
- pytest + hypothesis: property-based tests for the algebraic laws (Dimension/Unit form
  abelian groups; conversion round-trips; parse/format round-trips), a parametrized
  ufunc-conformance table shared by numpy/jax backends, and jit/grad/vmap equivalence
  tests against numpy references.
- Coverage gate ≥ 95% on `src/duq`.
- CI: lint, typecheck, test matrix (3 OS × Python 3.11/newest), jax job on ubuntu.
