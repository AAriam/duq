# Technical Due-Diligence: NumPy + JAX Integration for a Units/Quantities Library

**Prepared for:** the `duq` rewrite (physical dimensions / units / quantities)
**Date:** 2026-07-10
**Scope:** How to build numpy AND jax arrays that carry units and propagate them through ufuncs, reductions, linalg, indexing, broadcasting — and, for jax, survive `jit`/`grad`/`vmap`.

---

## 0. Executive orientation

There are, in practice, only three families of mechanism for attaching units to an array, and each maps onto a different "layer" of the array libraries:

1. **Be an array** — subclass `numpy.ndarray` (astropy `Quantity`, `unyt_array`).
2. **Wrap an array** — a composition type that implements NumPy's dispatch protocols `__array_ufunc__` (NEP 13) and `__array_function__` (NEP 18) (pint `Quantity`).
3. **Be a dtype** — push units into the element type via the new DType API (NEP 41/42/43). Prototype-only as of 2026.

JAX supports **none** of these three directly — `jax.numpy` does not consult `__array_ufunc__`/`__array_function__`, cannot be subclassed usefully, and its dispatch happens on **primitives** (`jax.lax.*_p`), not Python-level functions. So JAX needs a **fourth** mechanism: intercept the traced jaxpr and dispatch per-primitive. That is exactly what **quax** (Patrick Kidger) does, and what **unxt** builds a real units library on top of. The older **jpu** takes the pragmatic route of shipping a hand-curated `jpu.numpy` namespace.

The single most important architectural fact for `duq`: **there is no unified abstraction that serves numpy and jax with one integration implementation.** The Python array API standard explicitly disclaims this. You will write two integration layers. Design the pure dimension/unit core so that both can sit on top of it cheaply.

---

## 1. NumPy integration mechanisms

### 1.1 `ndarray` subclassing (astropy `Quantity`, `unyt_array`)

**How it works.** You subclass `numpy.ndarray`, override `__new__` (not `__init__`), and use `__array_finalize__` to propagate the unit attribute across the many internal code paths that create new arrays (slicing, `.view()`, ufunc outputs, templating). Ufunc behaviour is customised through `__array_ufunc__`; a handful of non-ufunc paths (e.g. `.squeeze()`, `.reshape` templating, scalar boxing) still go through `__array_wrap__`.
- NumPy subclassing docs: https://numpy.org/doc/stable/user/basics.subclassing.html
- astropy `Quantity` (subclass): https://docs.astropy.org/en/stable/units/quantity.html
- `unyt_array` (subclass): https://unyt.readthedocs.io/en/stable/modules/unyt.array.html and https://github.com/yt-project/unyt

**Pros.**
- Instances *are* `ndarray`, so a huge amount of third-party code that does `np.asarray(x)` or isinstance checks "just works," and every C-level buffer consumer accepts them.
- Indexing, broadcasting, reshaping, memory layout, and most reductions come for free.
- Best-in-class ecosystem penetration: astropy `Quantity` is the de-facto scientific standard.

**Cons / known pitfalls (the important part).**
- **Unit-stripping functions.** Any NumPy function that internally builds a fresh base-class array and does *not* route through `__array_wrap__`/`__array_function__` silently drops the unit or returns a bare `ndarray`. `np.concatenate`, `np.stack`, `np.where`, `np.dot`/`@`, and many `np.linalg` entry points have historically been problem spots because they were implemented before/around the protocol machinery. `__array_function__` (NEP 18) was the fix, but coverage is function-by-function.
- **`__array_wrap__` changed in NumPy 2.x.** The signature is now `__array_wrap__(self, arr, context=None, return_scalar=False)`; implementations that don't accept all three are deprecated. Worse, "in certain code paths `__array_wrap__` will now be passed a **base class**, rather than a subclass array" — a real behavioural change subclasses must defend against. astropy's own `__array_wrap__` now *raises* if called *with* a ufunc `context`, forcing all ufunc handling through `__array_ufunc__`/`__array_function__`, and only uses `__array_wrap__` for the residual non-ufunc paths (`squeeze`, etc.).
  - NumPy 2.0 release notes: https://numpy.org/doc/2.0/release/2.0.0-notes.html
  - NumPy 2.0 migration guide: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
  - astropy `__array_wrap__` behaviour: https://docs.astropy.org/en/latest/_modules/astropy/units/quantity.html
- **`out=` on ufuncs.** Writing into a pre-existing array with `out=` must re-validate/assign units on the output object; getting this wrong produces silent unit corruption. Every subclass has to special-case `out` inside `__array_ufunc__`.
- **View casting hazards.** `arr.view(MyQuantity)` produces an instance with **no unit set**; `__array_finalize__` must invent a sensible default (usually dimensionless) or you get `AttributeError`s deep in library code. `a.view(np.ndarray)` is the standard "escape hatch" but it also becomes the standard way units get lost.
- **Scalar boxing.** 0-d results and numpy scalar types (`np.float64`) are a recurring edge (the `return_scalar` parameter above exists precisely to manage whether a subclass instance or a plain scalar is returned).
- **The `unyt` maintainers themselves questioned the choice.** The long-standing "Why does `unyt_array` subclass `numpy.ndarray`?" issue frames the tension as composition-over-inheritance vs ecosystem compatibility. They keep the subclass for compatibility but acknowledge the maintenance cost. https://github.com/yt-project/unyt/issues/15
- astropy's own "Known Issues" page continues to document subclass-related surprises across NumPy versions: https://docs.astropy.org/en/latest/known_issues.html

**Verdict for a NEW library:** subclassing gives you the most "free" behaviour and the widest passive compatibility, at the cost of a permanent tax to NumPy's internals and its version churn. It also does **not** help you at all for JAX. If you want one conceptual `Quantity` shared between backends, subclassing `ndarray` is a poor foundation because a JAX-backed instance can't be an `ndarray`.

### 1.2 `__array_ufunc__` (NEP 13) + `__array_function__` (NEP 18) wrapper (pint `Quantity`)

**How it works.** The `Quantity` is a plain Python object holding `(magnitude, units)` where `magnitude` is any duck-array. NumPy hands *all* ufunc calls to `__array_ufunc__` and (almost) all high-level function calls to `__array_function__`; pint intercepts, computes the output unit, delegates the numeric work to the wrapped array, and re-wraps.
- pint NumPy support (the canonical reference for what's covered): https://pint.readthedocs.io/en/stable/user/numpy.html
- NEP 18: https://numpy.org/neps/nep-0018-array-function-protocol.html
- NEP-18 compatibility PR (history of how pint got here): https://github.com/hgrecco/pint/pull/905

**Pros.**
- Clean composition. The wrapped magnitude can itself be a Dask array, CuPy array, sparse array, xarray, etc. — pint deliberately treats these as **upcast types** (pint-pandas `PintArray`, pandas `Series`, xarray `DataArray`/`Dataset`/`Variable` get priority) and "assumes it can wrap any other duck array" that provides `__array_function__`, `shape`, `ndim`, `dtype`. This is how `pint-xarray` and `pint-pandas` exist.
- **Fail-loud semantics.** Functions pint does *not* explicitly handle raise, rather than silently dropping units — the opposite of the subclass failure mode. This is a genuine safety advantage.
- You never fight `ndarray`'s internals, view casting, or `__array_finalize__`.

**Cons.**
- **Coverage is a treadmill.** Every ufunc/function/method must be enumerated and given a unit rule. pint's docs literally list "~25 ufuncs, ~14 trig, comparison + floating ufuncs, ~130+ `np.*` functions, ~25 ndarray methods" as *the supported set*, and close with "Pull requests are welcome for any NumPy function, ufunc, or method that is not currently supported." That is the maintenance burden in one sentence. https://pint.readthedocs.io/en/stable/user/numpy.html
- Historic gaps have included `@`/`matmul`, and partial-only support for masked arrays, Dask, and CuPy ("full integration planned").
- **Wrapping-vs-being-wrapped conflicts.** When *another* library also implements NEP 18 and tries to wrap pint (rather than the reverse), you get protocol ping-pong / precedence ambiguity — a documented class of bugs. https://github.com/hgrecco/pint/issues/878
- Small constant-factor Python overhead per operation (one interception + re-wrap per call). Usually negligible next to the array work; occasionally visible in tight scalar loops.

**Verdict:** This is the **right default for the NumPy side of a new library.** Fail-loud, composes with the duck-array ecosystem, no `ndarray` entanglement. The cost is that you own a per-function unit-rule table forever — but that table is small, declarative, and exactly the domain logic you want to own anyway.

### 1.3 `np.lib.mixins.NDArrayOperatorsMixin`

A tiny but real convenience: mix it into your wrapper and it defines all the Python arithmetic dunders (`__add__`, `__mul__`, `__gt__`, `__matmul__`, in-place ops, reflected ops, …) **in terms of `__array_ufunc__`**. So you implement dispatch **once** in `__array_ufunc__` and get consistent operator behaviour for free, with no risk of operator/ufunc divergence. NumPy explicitly recommends this pattern for wrapper types: put all override logic in `__array_ufunc__`, don't hand-roll dunders.
- Reference: https://numpy.org/doc/stable/reference/arrays.classes.html

**Verdict:** if you go the wrapper route (1.2), **use this mixin.** It's free correctness.

### 1.4 Units as a custom dtype (NEP 41 / 42 / 43)

**The idea.** Instead of a wrapper object, make the *element type* carry the unit: `np.array([...], dtype=Meter)`. Ufunc loops would then compute output dtypes/units (`meter / second -> meter_per_second`), and a units dtype "should be able to fall back to NumPy's existing math implementations."
- NEP 41 (new dtype system, explicitly motivates units): https://numpy.org/neps/nep-0041-improved-dtype-support.html
- NEP 42 (extensible dtypes): https://numpy.org/neps/nep-0042-new-dtypes.html
- NEP 43 (extensible ufuncs — the piece that would compute output units): https://numpy.org/neps/nep-0043-extensible-ufuncs.html

**Current status (2026): prototype only; do not build on it.**
- There **is** a units dtype prototype — `unytdtype` — living in the official example repo `numpy/numpy-user-dtypes`, alongside `asciidtype`, `stringdtype`, `quaddtype`, `mpfdtype`, `metadatadtype`. https://github.com/numpy/numpy-user-dtypes
- That repo's README is blunt: **"These dtypes are not meant for real-world use yet. The dtype API is not finalized and the dtypes in this repository are still active works in progress."** Only `stringdtype` graduated into NumPy proper; `quaddtype` is the most active spin-out. The units dtype has **not** shipped anywhere as a product.
- The deep reason units-as-dtype is hard is called out in NEP 43 itself: **parametric dtypes** (a unit is a parameter, like string length) are not supported by NumPy's own ufuncs, precisely because "a datatype which embeds a physical unit must calculate the new unit information" and "the current casting rules cannot describe casting for such parametric datatypes implemented outside of NumPy." Dimensional analysis is *the* motivating hard case, and it is not solved. https://numpy.org/neps/nep-0043-extensible-ufuncs.html
- The dtype approach is also inherently **NumPy-only** — it does nothing for JAX, which has its own dtype system and no NEP 42/43 equivalent.

**Verdict:** conceptually the most elegant (units become invisible and zero-overhead), but **not viable for a from-scratch library shipping in 2026**. Revisit in a few years. Keep your architecture such that a future dtype backend *could* be added, but don't bet on it.

### 1.5 The Python array API standard as an abstraction layer

**The hope:** write your integration once against `data-apis` array-api and have it serve numpy + jax + torch + cupy.
- Standard: https://data-apis.org/array-api/latest/purpose_and_scope.html
- `array-api-compat` (the practical shim): https://data-apis.org/array-api-compat/ and https://github.com/data-apis/array-api-compat
- NumPy's adoption: https://numpy.org/neps/nep-0047-array-api-standard.html ; JAX is array-API-compatible since v0.4.32.

**Why it does NOT give you a single units integration.** The standard is a **common function namespace**, not a dispatch mechanism. Its explicit non-goals directly kill the dream:
- "**subclassing of an array class**" is out of scope;
- there is **no** `__array_ufunc__`/`__array_function__`-style protocol in the standard;
- "making it possible to **mix multiple array libraries** in function calls" is out of scope;
- "transparent backend switching" is rejected as infeasible.
(All four are stated in Purpose & Scope: https://data-apis.org/array-api/latest/purpose_and_scope.html)

`array-api-compat` reinforces this: it "does not use a separate Array object" — it just normalises calls onto each library's *own* array type. It gives you a **portable way to call functions on whichever backend** you already have, but it provides **nothing** for a unit-carrying wrapper to intercept those calls generically.

**Where it IS useful for `duq`.** Use the array API *internally* to write **backend-agnostic numeric kernels** once (e.g. your unit-conversion arithmetic, `decompose`, reductions), calling `xp = array_namespace(magnitude)` and then `xp.multiply(...)`. That deduplicates the *numeric* code between numpy and jax. But the **interception layer** (how a `Quantity` gets a chance to run at all) is still per-backend: NEP 13/18 on NumPy, quax-style primitive dispatch on JAX. The array API removes duplicated arithmetic; it does not remove the two integration front-ends.

---

## 2. JAX integration mechanisms (the hard part)

### 2.1 Why `__array_ufunc__` / `__array_function__` do not work with `jax.numpy`

Three independent reasons, all fatal:
1. **`jax.numpy` is not `numpy`.** `jnp.add`, `jnp.concatenate`, etc. are JAX functions that lower to `jax.lax` primitives. They do not implement NumPy's dispatch protocols and never call `__array_ufunc__`/`__array_function__`. There is no hook to intercept.
2. **JAX has no general custom-array dispatch.** As jpu's author puts it directly: "JAX does not (yet?) provide a general interface for dispatching of ufuncs on custom array classes." He tried the undocumented `__jax_array__` hook and found it "insufficiently flexible for Pytree objects." https://github.com/dfm/jpu/blob/main/README.md
3. **Dispatch happens after tracing, on primitives.** Under `jit`/`grad`/`vmap`, your Python object is replaced by a `Tracer` whose *abstract value* is a `ShapedArray`. By the time work happens, it's a jaxpr of `lax` primitives operating on abstract arrays — your Python-level `Quantity` and its `__mul__` are long gone. Any units mechanism must inject itself into **primitive evaluation**, not Python operators.

Consequence: you cannot reuse the NumPy integration for JAX. You need either (a) a curated shadow namespace (jpu) or (b) primitive-level interception (quax).

### 2.2 Pytree registration of a `Quantity` wrapper: leaves vs static/aux

A JAX-friendly `Quantity` must be a **pytree**: JAX flattens it into leaves (traced) + `aux_data` (static, hashable, part of the treedef).
- Custom pytree docs: https://docs.jax.dev/en/latest/custom_pytrees.html ; https://docs.jax.dev/en/latest/pytrees.html

The design decision is **which field is which**:
- **`magnitude` = leaf** (a traced array; differentiable, batchable, the thing `jit` compiles over).
- **`unit` = static `aux_data`** (part of the treedef; not traced).

This is exactly how the equinox/quax idiom expresses it — `units: ... = eqx.field(static=True)` in quax's own `Unitful` example (see 2.4). Consequences of "unit is static":
- **`jit`:** the treedef (including the unit) participates in the compilation cache key. Two calls with the *same* unit reuse the compiled program; a call with a *different* unit **retraces and recompiles**. See 2.7 — this is the central tradeoff and it is acceptable practice.
- **`grad`:** differentiation flows through the leaf (`magnitude`). The unit rides along on the pytree structure. If `f: x[unit_in] -> y[unit_out]`, then mathematically `d y / d x` has units `unit_out / unit_in`; a units-aware `grad` must *produce* that — i.e. wrap the raw gradient array with the divided unit. quax/unxt achieve this because the whole forward computation is done in unit-carrying values, so the VJP/JVP naturally carries divided units (quaxed re-exports `grad`/`jacfwd`/`jacobian`/`hessian`). Historic sharp edge: `jax.grad`'s `_check_scalar` calls `concrete_aval` which errored on a `Quantity` output — i.e. "Gradient only defined for scalar-output functions" false-positives — see https://github.com/patrick-kidger/quax/issues/5.
- **`vmap`:** batching maps over the leaf's mapped axis; the unit is structure and is shared across the batch (you cannot `vmap` over "different units per row" — units aren't a traced axis, by design).

**If instead you made `unit` a leaf:** it would have to be a traced array (a numeric encoding of the dimension vector). You'd lose static dimensional checking at trace time, `jit` couldn't specialise on it, and error messages would be about array values not units. Everyone who has built this (unxt, quax's example, jpu) keeps units **static**. Follow them.

### 2.3 Why a curated shadow namespace is the other option (and what it costs)

The alternative to primitive interception is to **not** try to make `jnp.*` work at all, and instead ship your own `duq.jax.numpy` with hand-written unit rules that call the real `jnp.*` under the hood. This is jpu (2.5). Pro: dead simple, transparent, no dependency on quax's tracing machinery. Con: users must remember to import *your* namespace, and you own the same coverage treadmill as pint — but now for `jax.numpy` too.

### 2.4 quax — primitive-level interception (Patrick Kidger)

**Repo/docs:** https://github.com/patrick-kidger/quax  · https://docs.kidger.site/quax/  · custom rules example: https://docs.kidger.site/quax/examples/custom_rules/  · API: https://docs.kidger.site/quax/api/quax/

**Mechanism (this is the crux of the whole JAX story).**
- You subclass **`quax.ArrayValue`** (which is a `quax.Value`, which is an equinox `Module`, hence a pytree). You implement two methods:
  - `aval(self) -> jax.core.ShapedArray` — the abstract value JAX should "see" for your object (shape + dtype).
  - `materialise(self)` — how to collapse to a plain array; a units type typically **raises** here ("refusing to materialise a Unitful array") so that unhandled primitives fail loudly instead of silently dropping units.
- You register per-primitive rules with **`@quax.register(jax.lax.mul_p)`** etc. Rules are **multiple-dispatch** functions (via `plum`) keyed on the argument types (`Unitful×Unitful`, `ArrayLike×Unitful`, …).
- **`quax.quaxify(f)`** is a custom JAX transform: it traces `f` to a jaxpr, then re-interprets the jaxpr **primitive by primitive**, looking up the registered rule for each primitive given the actual (possibly `Unitful`) operand types — "just like `jax.vmap` reinterprets each operation as its batched version." Registered → your unit rule runs; unregistered → it falls back to `materialise()` (which, for units, raises).

Quax's own docs ship a **`Unitful`** example that is essentially a miniature units library:
```python
class Unitful(quax.ArrayValue):
    array: ArrayLike
    units: dict[Dimension, int] = eqx.field(static=True)   # <-- units are STATIC aux data
    def aval(self):  return jax.core.ShapedArray(jnp.shape(self.array), jnp.result_type(self.array))
    def materialise(self): raise ValueError("Refusing to materialise Unitful array.")

@quax.register(jax.lax.mul_p)
def _(x: Unitful, y: Unitful):  # units add on multiply
    ...
@quax.register(jax.lax.add_p)
def _(x: Unitful, y: Unitful):  # units must match on add
    ...
```
(https://docs.kidger.site/quax/examples/custom_rules/)

**Coverage model.** You register the ~small set of `lax` primitives, not the ~large set of `jnp` functions. This is a real leverage win: `jnp.concatenate`, `jnp.where`, `jnp.stack`, reductions, etc. all *lower to* a handful of primitives (`concatenate_p`, `select_n_p`, `reduce_sum_p`, `broadcast_in_dim_p`, `convert_element_type_p`, `dot_general_p`, `integer_pow_p`, …). Cover the primitives and you cover a large swath of `jnp` for free. That is why quax/unxt can plausibly claim broad coverage where jpu enumerates functions.

**Transformation support / limitations (from the README):**
- Supported: `jax.custom_jvp`, `jax.lax.cond_p`, `jax.lax.while_p`, `jax.lax.scan_p` (so control flow and `scan`-carry work — important for ODE/loop code).
- **Not** supported: `jax.custom_vjp` ("should be fairly straightforward to add").
- Labelled **"Work in progress!"**

**Maturity / maintenance (2026).** ~140+ GitHub stars; the released line is **0.3.x** (the `0.3.6` tag is the current pin used across the ecosystem). Release cadence was heavy through 2024 and has slowed since — i.e. it is **stable-but-quiet**, not dead: it remains the foundation of unxt, which shipped **v1.11.4 in June 2026** (below), so quax is very much a live dependency in 2026, just not fast-moving. https://github.com/patrick-kidger/quax/releases  This "critical dependency, single maintainer, low velocity" profile is a real supply-chain risk to weigh (Risk #4).

### 2.5 jpu — JAX + pint, the curated-namespace approach (Dan Foreman-Mackey)

**Repo/README:** https://github.com/dfm/jpu  · https://github.com/dfm/jpu/blob/main/README.md  · PyPI: https://pypi.org/project/jpu/

- Design: wraps **pint** `Quantity` objects to be JAX-compatible; registers the `Quantity` as a pytree with the **magnitude as leaf** and units as aux; unit propagation happens **at trace time** so "jitted functions should see no runtime cost."
- Usage: `u = jpu.UnitRegistry(); x = jnp.array([...]) * u.cm`, then call **`jpu.numpy.*`** functions (`jpu.numpy.linspace`, etc.) instead of `jax.numpy.*`.
- **The defining limitation:** "users must use `jpu.numpy` functions ... instead of the `jax.numpy` interface. This is because JAX does not provide a general interface for dispatching of ufuncs on custom array classes." And "only a subset of the numpy/jax.numpy interface is implemented" — PRs welcome (same treadmill as pint).
- Status: **experimental** ("expect some sharp edges"); latest release **v0.0.5 (April 2025)**; ~54 stars, small. Reuses pint's mature unit *catalog* and registry, which is its main advantage over rolling your own.

**jpu vs unxt in one line:** jpu = "pint units + hand-written `jpu.numpy`"; unxt = "astropy units + quax primitive interception + `quaxed` shadow namespace." jpu is simpler and leans on pint; unxt is more ambitious and gets broader `jnp` coverage via quax.

### 2.6 unxt — the current state of the art (GalacticDynamics)

**Repo/docs/paper:** https://github.com/GalacticDynamics/unxt  · https://unxt.readthedocs.io/en/latest/  · API: https://unxt.readthedocs.io/en/latest/api/index.html  · JOSS/arXiv paper: https://arxiv.org/html/2603.08770v1 (JOSS DOI 10.21105/joss.07771) · PyPI: https://pypi.org/project/unxt/

- **Built on quax + equinox**, with a companion **`quaxed`** package that pre-wraps JAX functions in `quax.quaxify` so users write `from quaxed import grad, vmap; import quaxed.numpy as jnp` and get unit-aware versions without boilerplate. https://github.com/GalacticDynamics/quaxed
- **Units backend is astropy.units** — unxt does *not* reinvent the unit catalog; it reuses astropy's conversion framework and (per the paper) several unxt authors are astropy.units maintainers. There is an open design thread (issue #139) to **abstract the backend** so astropy / unyt / pint / Julia `Unitful` / a custom backend could be swapped in — i.e. they explicitly recognise the value of decoupling the catalog from the JAX integration. https://github.com/GalacticDynamics/unxt/issues/139
- **API surface (function-oriented front-end, astropy-flavoured object API):**
  - `Quantity(value, unit)` / alias `Q`; **dimension-parameterised** form `Q["length"](1, "m")` for runtime dimensional checking. `BareQuantity` skips the check for hot paths.
  - Free functions: `unit()`, `unit_of()`, `dimension()`, `dimension_of()`, `uconvert(q, unit)` (convert), **`ustrip(q, unit)`** (return raw magnitude in a given unit — the JAX-idiomatic "get me a plain array" call).
  - `unitsystem()` / `unitsystem_of()`; built-in SI, CGS, Galactic; **dynamical unit systems** where `G=1` and you supply only 2 of {length,time,mass}.
  - Specialised types: `Angle` (with `.wrap_to()`), `StaticQuantity` / `StaticValue` (numpy-backed, for values you want as `jit` static args).
  - Submodules: `unxt.dims`, `unxt.units`, `unxt.unitsystems`, `unxt.quantity`, `unxt.experimental`.
- **JIT-compatibility model:** units are **static metadata** (aux data), so `jit`/`vmap`/`grad`/`jacobian`/`hessian` all work; the price is that a compiled function is specialised per unit and dynamic in-graph unit changes are disallowed (paper is explicit about this). Maturity: actively developed, **v1.11.4 (June 2026)**, peer-reviewed (JOSS). This is the reference design to study.

### 2.7 The `jit`-static-unit / retrace-per-unit tradeoff — is it acceptable?

**Yes, and it is the standard practice.** Because the unit lives in the pytree `aux_data` / equinox static field, it becomes part of the treedef, and the treedef is part of `jit`'s cache key. Therefore:
- Calling a jitted function with `metres` vs `kilometres` are two different treedefs → **two compilations**. Within a program you typically call with a fixed unit per call-site, so this is amortised and invisible.
- The alternative (unit as a traced leaf) would avoid retracing but would (a) forfeit *compile-time* dimensional-consistency checking, (b) make error messages worse, and (c) require encoding dimensions as arrays. Nobody does this.
- **Guidance for `duq`:** keep units static; document that a hot loop should convert to a canonical unit *once* (via `ustrip`/`uconvert`) before entering a tight jitted kernel, so the kernel sees plain arrays and never retraces. This is also the fastest path (no per-op unit bookkeeping inside the kernel). Equinox's `filter_jit`/static-field semantics are the relevant machinery: https://docs.kidger.site/equinox/api/transformations/

### 2.8 How the libraries handle the awkward functions

| Operation | Curated (jpu) | Primitive-interception (quax/unxt) |
|---|---|---|
| `jnp.concatenate` | must be in `jpu.numpy`; unit-checks all inputs | lowers to `concatenate_p`; one registered rule checks all operand units equal |
| `jnp.where` / select | in `jpu.numpy` | `select_n_p` rule: branches must share a unit; predicate is dimensionless |
| reductions (`sum`,`mean`) | per-function | `reduce_sum_p` etc.: unit preserved; `mean` stays same unit |
| `dot`/`@`/linalg | per-function | `dot_general_p` rule multiplies units |
| `scan`/`while` carry | works if body uses `jpu.numpy` | quax supports `scan_p`/`while_p`/`cond_p` so carry keeps units |
| `grad` through unit conversion | pytree carries units; gradient unit = out/in | quaxed `grad`; gradient naturally carries `unit_out/unit_in` (modulo the `custom_vjp`/scalar-check sharp edges above) |

---

## 3. API design survey

### 3.1 pint — registry-centric
- **Shape:** `ureg = UnitRegistry()`; units are *attributes of the registry* (`ureg.meter`, `3 * ureg.kg`); `Quantity = ureg.Quantity`. There is also an **application registry** (`get_application_registry()` / `set_application_registry()`) — a process-global default used for bare `pint.Quantity(...)` construction and, crucially, for **unpickling** (a pickled quantity must be unpickled against a compatible registry). https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html · serialization: https://pint.readthedocs.io/en/stable/advanced/serialization.html
- **String parsing:** rich — `ureg("kg*m/s^2")`, `ureg.parse_expression`, `ureg.Unit("km/h")`. Powerful but is a parser you must maintain, and locale/`^` vs `**` conventions bite users.
- **Affine/offset units:** `degC`/`degF` are offset units; **arithmetic on them raises by default** (add/mul/pow are ambiguous). Two escape hatches: (1) `delta_degC` "delta units" for differences, safe in multiplicative contexts; (2) `UnitRegistry(autoconvert_offset_to_baseunit=True)` to auto-convert to Kelvin before multiplying. https://pint.readthedocs.io/en/stable/user/nonmult.html
- **Loved:** breadth of unit catalog, `pint-pandas`/`pint-xarray` integration, contexts/equivalencies, fail-loud NumPy behaviour.
- **Disliked (from issues):** cross-registry quantities can't interoperate ("not supposed to operate between quantities that belong to different registries"), the pickling/registry global is a footgun, string-parse edge cases, protocol ping-pong when *wrapped* by another NEP-18 type (https://github.com/hgrecco/pint/issues/878), and per-op overhead in scalar-heavy code.

### 3.2 unyt — module-level, ndarray subclass
- **Shape:** `from unyt import km, s`; `a = [1,2,3] * km`; `unyt_array` (arrays) and `unyt_quantity` (scalars). A `UnitRegistry` exists but is mostly implicit; units feel like importable module-level objects, which many users find more ergonomic than pint's registry indirection. https://unyt.readthedocs.io/en/stable/modules/unyt.array.html · https://github.com/yt-project/unyt
- **Mechanism:** `ndarray` subclass; a `_ufunc_registry` maps each numpy ufunc to a unit rule (`_preserve_units`, `_multiply_units`, `_divide_units`, `_power_unit`). Inherits all the subclass pitfalls in §1.1.
- **Loved:** simple imports, fast, good yt/astro integration. **Disliked:** subclass edge cases; smaller catalog than pint/astropy.

### 3.3 astropy.units — the scientific standard
- **Shape:** module-level unit singletons `u.m`, `u.kg`; `Quantity = value * u.m` (also an `ndarray` subclass). Conversion via `.to(u.km)`, numeric extraction via `.to_value(u.km)`, canonicalisation via `.decompose()`. https://docs.astropy.org/en/stable/units/quantity.html
- **Equivalencies:** the standout feature — context-scoped conversions between *different* dimensions when physics licenses it (spectral wavelength↔frequency↔energy, `dimensionless_angles()` for rad↔dimensionless, temperature energy, etc.). Passed as `q.to(target, equivalencies=[...])`. https://docs.astropy.org/en/stable/units/equivalencies.html
- **Constants:** `astropy.constants` are `Quantity`s with units and **versioned CODATA** (see §4).
- **Loved:** breadth, correctness, equivalencies, ecosystem gravity. **Disliked:** heavyweight import, `ndarray`-subclass surprises, no JAX (this is exactly the gap unxt fills by *reusing astropy's catalog* behind a JAX front-end).

### 3.4 unxt — function-oriented, dimension-parameterised (JAX-native)
- Covered in §2.6. The API novelty worth copying: **`ustrip`/`uconvert` as free functions** (functional, JAX-idiomatic, no method-call-on-tracer awkwardness) and **`Quantity["length"]` parametric types** for optional dimensional typing. `dimension()`/`unit()` as first-class constructors that parse strings (`"length/time"`, `"km/h"`).

### 3.5 Cross-cutting API decisions `duq` must make
- **Registry pattern vs module-level units.** Registry (pint) = explicit, customisable, but global-state/pickling pain and cross-registry incompatibility. Module-level (astropy/unyt) = ergonomic imports, one implicit global catalog. **Recommendation:** default module-level units backed by a single default registry, *plus* an explicit registry object for advanced/custom catalogs — but make **all** default units share one registry so quantities always interoperate (avoid pint's cross-registry trap).
- **String parsing `"kg*m/s^2"`.** Expected by users, but a maintenance liability and a security/ambiguity surface. Support it, but make constructed-from-objects (`u.kg * u.m / u.s**2`) the canonical internal path and treat strings as a thin parser front-end.
- **Affine/offset units (Celsius).** Adopt pint's proven policy: **offset units are convertible but not freely arithmetic**; provide explicit **delta** units for differences; raise (don't guess) on ambiguous multiply/add; offer an opt-in auto-convert-to-base flag. Never silently treat `degC` as multiplicative.
- **Dimensionless.** Must be a first-class unit (identity of the dimension group), so `q / q`, `sin(angle)`, ratios, and `%`/`ppm` all land somewhere well-defined. Radian should be dimensionless-but-labelled (astropy's `dimensionless_angles` equivalency shows why you want to track it yet allow dropping it).
- **`%` / `ppm`.** Model as dimensionless *scaled* units (percent = 1e-2, ppm = 1e-6), i.e. prefixes-on-dimensionless, so they convert cleanly to/from a bare ratio.
- **Array-of-quantity vs quantity-of-array.** Decisively choose **quantity-of-array** (one unit for the whole buffer; magnitude is the array) — this is what pint/unyt/astropy/unxt/jpu all do and what makes JAX pytree/`jit` work (one static unit, one traced leaf). "Array of per-element quantities" is the dtype approach (§1.4) and is not viable today.

---

## 4. Data formats for unit-definition catalogs

### 4.1 pint `default_en.txt` — a DSL text file
pint ships a plain-text DSL (`pint/default_en.txt`) loaded at registry creation. Syntax by category:
- **Base units define a dimension:** `meter = [length] = m = metre` (name = `[dimension]` = symbol = aliases).
- **Prefixes (declared once, combined lazily):** `kilo- = 1e3 = k-`, `milli- = 1e-3 = m-`, `micro- = 1e-6 = µ- = u-`. The trailing `-` marks a prefix; pint combines prefix×unit **on demand** at lookup (`kilometer`), not by pre-expanding the cross product.
- **Derived units by expression:** `joule = newton * meter = J`, `hertz = 1 / second = Hz`.
- **Derived dimensions:** `[area] = [length] ** 2`, `[velocity] = [length] / [time]`.
- **Groups:** `@group USCSLengthInternational … @end` (named collections).
- **Systems:** `@system SI … @end` (which units are canonical in a system).
- **Contexts/equivalencies:** `@context … @end` (rules for cross-dimension conversion).
Source: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

**Pros:** human-editable, users can supply their own file / append definitions, prefix combination is lazy (cheap import). **Cons:** it's a bespoke grammar with its own parser to maintain and validate.

### 4.2 astropy — units defined in Python code
astropy defines units imperatively in Python modules (`def_unit(...)` calls in `astropy/units/si.py`, `cgs.py`, `astrophys.py`, …), not a data file. Constants likewise are Python objects. **Pro:** full language power, no parser, IDE/type support. **Con:** catalog is code, so third-party extension means importing/patching modules; contributes to astropy's heavy import time.

### 4.3 unyt — Python dict/registry in code
unyt keeps a default unit lookup and a `_ufunc_registry` in Python; units are `sympy`-backed symbolic objects. Similar tradeoffs to astropy (code, not data), lighter than astropy overall.

### 4.4 CODATA constants — sourcing and versioning (do this deliberately)
Constants must be **versioned to a CODATA release** and pinned, because values change between releases and reproducibility depends on it:
- **CODATA 2022** was published 2024-05-24 and is now propagating through the stack. **astropy** added CODATA 2022 and made it the default (`astropyconst80` = CODATA 2022 + IAU 2015), with `ScienceState` switches to pin older sets. https://docs.astropy.org/en/stable/constants/index.html
- **scipy.constants** exposes versioned modules (`codata2010/2014/2018/2022`) with the `physical_constants[name] = (value, unit, uncertainty)` triple, and only updated to 2022 after a lag. https://docs.scipy.org/doc/scipy/reference/constants.html · https://github.com/scipy/scipy/issues/21596
- astropy long ago recognised the need for **"version control for constants"** as an explicit feature. https://github.com/astropy/astropy/issues/4958

**Lesson for `duq`:** treat the CODATA year as a **first-class, selectable version** (carry value + uncertainty + unit + source-year), default to the latest (2022), and make it switchable — do not hard-code magic numbers inline (cf. the historical Boltzmann-constant exponent typo already fixed in this repo's own history).

### 4.5 Recommendation for a declarative catalog
- **Format:** ship a **TOML** catalog (stdlib `tomllib` since 3.11 → zero-dependency, fast, comment-friendly, less whitespace-fragile than YAML, more human-friendly than JSON). Structure:
  - `[dimensions]` — base dimensions and derived-dimension expressions.
  - `[prefixes]` — name, factor, symbol, aliases (declared **once**).
  - `[units]` — name, definition (expression or base-dimension), symbol, aliases, `offset`/`affine` flag for temperatures, `prefixable = true/false`.
  - `[constants]` — name, value, unit, uncertainty, `codata_year`/source.
  - `[systems]` / `[groups]` and optional `[contexts]` for equivalencies.
- **Prefix handling:** **declare prefixes separately and combine lazily** (pint's model). Do **not** eagerly expand the prefix×unit cross product at import — that is thousands of objects for no reason. Resolve `kilometre` on first lookup and memoise. This is the single biggest import-time lever.
- **Import-time performance:** parse the TOML into lightweight **frozen dimension vectors** (tuple of small ints per base dimension) rather than sympy expressions; defer prefix combination and defer any array/numpy import until a `Quantity` is actually made. Target: importing `duq` (dimension/unit core) should not import numpy, and definitely not jax/astropy. Consider caching the parsed catalog (e.g. a pickled/precompiled table) if TOML parse ever shows up in import profiles.

---

## 5. Recommendation for the `duq` rewrite

**Targets:** Python ≥3.11, numpy ≥1.26 (incl. 2.x), jax optional.

### 5.1 Architecture: a pure core with two thin backends

```
duq/                      # PURE PYTHON. No numpy in the hot path, no jax, no astropy.
  dimensions.py           # Dimension = frozen int-vector over base dims; add/mul/pow algebra
  units.py                # Unit = (scale, offset, Dimension, symbol); parsing; prefixes (lazy)
  registry.py             # one default registry (module-level unit singletons) + custom registries
  catalog/*.toml          # declarative units/prefixes/constants (CODATA-versioned)
  constants.py            # versioned CODATA constants (value, unit, uncertainty, year)
  quantity.py             # backend-agnostic Quantity: (magnitude, Unit); arithmetic in terms of
                          #   an injected array namespace (array-API) — NO hard numpy dependency

duq/numpy/  (extra: [numpy])
  - Quantity_np wrapper: __array_ufunc__ (NEP 13) + __array_function__ (NEP 18)
  - mix in np.lib.mixins.NDArrayOperatorsMixin
  - a per-ufunc / per-function UNIT-RULE TABLE (the only thing you maintain)
  - fail-loud on unhandled functions

duq/jax/    (extra: [jax])
  - Quantity_jax as an equinox Module / pytree: magnitude=leaf, unit=STATIC aux
  - quax.ArrayValue subclass + @quax.register rules on lax primitives
  - a `duq.jax.numpy` (quaxed-style) convenience namespace pre-wrapped in quaxify
  - ustrip()/uconvert() free functions for entering hot jitted kernels
```

**Key layering principles:**
1. **The dimension/unit/registry/constants core has zero array-library dependency.** Dimensional analysis is pure integer arithmetic on exponent vectors; unit conversion is `scale`/`offset` scalars. This core imports instantly, is trivially testable, and is shared verbatim by both backends. (This is also what unxt's issue #139 is groping toward — decoupling the catalog from the JAX layer.)
2. **The magnitude arithmetic is written once against the array API** (`array_namespace(magnitude)`), so `Quantity.__mul__` etc. don't branch on backend. numpy and jax both satisfy the array API (numpy 1.26+ via `array-api-compat`, jax ≥0.4.32 natively). This deduplicates the *numeric* code.
3. **The interception front-ends are separate and backend-specific** because there is no shared mechanism (§1.5): NEP 13/18 for numpy, quax primitive rules for jax. The **unit-rule logic itself** (what unit does `mul` produce?) lives in the pure core and is *called by* both front-ends, so you write the physics once and the plumbing twice.
4. **quantity-of-array, unit static.** One unit per buffer; magnitude is the array; unit is aux/static everywhere. Provide `ustrip`/`uconvert` to drop to plain arrays for hot loops.
5. **Do NOT subclass `ndarray`.** Wrapper + NEP 13/18 gives fail-loud behaviour, avoids the `__array_wrap__`/view-casting/`out=`/2.x-migration tax (§1.1), and — decisively — is the only design that can share one conceptual `Quantity` with a jax-backed magnitude.
6. **Reuse a mature unit catalog conceptually, own it in data.** Ship your own TOML catalog (§4.5) rather than depending on astropy/pint at runtime — but seed it from CODATA + a known-good catalog and keep constants CODATA-versioned. (unxt chose to *depend* on astropy.units; for a clean-room rewrite that wants a light core and jax-first ergonomics, a self-contained TOML catalog is preferable and removes the astropy import weight.)

### 5.2 On quax vs a curated jax namespace
Prefer **quax primitive interception** (à la unxt) over a hand-curated `jax.numpy` clone (à la jpu): registering ~a few dozen `lax` primitives covers most of `jnp` for free, versus enumerating hundreds of functions forever. Wrap it in a `duq.jax.numpy` (quaxed-style) namespace so users don't type `quaxify` by hand. **But** vendor a fallback plan (5.3, Risk #4): keep the quax dependency isolated behind `duq/jax/` so that if quax stalls you can (a) pin it, (b) upstream fixes, or (c) fall back to a curated namespace without touching the core.

### 5.3 The 5 biggest technical risks

1. **JAX coverage & correctness of primitive rules.** `jnp` lowers to many `lax` primitives (`dot_general`, `gather`/`scatter`, `pad`, `reduce_*`, `select_n`, `convert_element_type`, `integer_pow`, `dynamic_slice`, `concatenate`, control-flow `scan/while/cond`). Each needs a correct unit rule; missing ones fail (fail-loud is good, but coverage gaps are user-visible). Indexing/scatter and `dot_general` (which unit multiplies) are the fiddly ones. **Mitigation:** property-based tests that run the same computation with and without units and assert magnitude-equality + expected-unit; treat the primitive-rule table as the crown-jewel test target.

2. **`grad`/autodiff unit correctness and sharp edges.** Gradients must carry `unit_out/unit_in`; `jacfwd`/`jacobian`/`hessian` compound this. quax historically tripped `jax.grad`'s scalar-output check on `Quantity` outputs (https://github.com/patrick-kidger/quax/issues/5), and `jax.custom_vjp` is unsupported in quax. **Mitigation:** test grad/jvp/vjp unit propagation explicitly; document `custom_vjp` gap; provide `ustrip`-then-grad-then-rewrap helpers as an escape hatch.

3. **NumPy 2.x + duck-array interoperability churn.** Even the wrapper route must track `__array_function__` coverage, upcast/downcast precedence vs pandas/xarray/dask/cupy, and the NEP-18 "who wraps whom" ambiguity (https://github.com/hgrecco/pint/issues/878). NumPy 2.x also shifted `__array_wrap__`/scalar semantics. **Mitigation:** CI matrix across numpy 1.26 / 2.x; explicit upcast-type registry like pint's; fail-loud defaults.

4. **quax single-maintainer / velocity risk.** quax is the linchpin of the jax story, is labelled "work in progress," sits at the 0.3.x line with slowed cadence, and is effectively one maintainer — yet everything (incl. unxt v1.11.4, June 2026) depends on it. https://github.com/patrick-kidger/quax **Mitigation:** pin quax; isolate it behind `duq/jax/`; keep a curated-namespace fallback design on the shelf; be prepared to upstream/patch.

5. **Import-time & catalog performance / correctness.** Eager prefix expansion, sympy-style symbolic units, or importing numpy/astropy at `import duq` will bloat startup; wrong/unversioned constants silently corrupt results (this repo already had a Boltzmann-exponent typo). **Mitigation:** pure-Python core with no array import, lazy prefix combination, frozen int-vector dimensions, CODATA-year-pinned constants with uncertainty, and an import-time budget test in CI.

### 5.4 Secondary risks to track
- **Offset/affine units** semantics (Celsius) are a perennial user-confusion and correctness source — adopt pint's raise-by-default + delta-units policy from day one.
- **String-parser** ambiguity/security — keep object construction canonical, parser thin.
- **Registry global state / pickling / serialization** — one default registry, avoid pint's cross-registry incompatibility; define a stable serialization format (magnitude + unit string) that round-trips across backends.
- **Dimensionless & angle handling** — decide radian/steradian policy and `%`/`ppm` scaling up front (they leak into every reduction and trig call).

---

## Appendix: primary sources (all verified July 2026)

**NumPy mechanisms**
- Subclassing ndarray: https://numpy.org/doc/stable/user/basics.subclassing.html
- Standard array subclasses / NDArrayOperatorsMixin: https://numpy.org/doc/stable/reference/arrays.classes.html
- NumPy 2.0 release notes (`__array_wrap__`/`return_scalar`): https://numpy.org/doc/2.0/release/2.0.0-notes.html
- NumPy 2.0 migration guide: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
- NEP 13 (array_ufunc) & NEP 18 (array_function): https://numpy.org/neps/nep-0018-array-function-protocol.html
- NEP 41/42/43 (dtype system, extensible ufuncs): https://numpy.org/neps/nep-0041-improved-dtype-support.html · https://numpy.org/neps/nep-0042-new-dtypes.html · https://numpy.org/neps/nep-0043-extensible-ufuncs.html
- numpy-user-dtypes (unytdtype prototype; "not for real-world use"): https://github.com/numpy/numpy-user-dtypes
- Array API purpose/scope (subclassing & dispatch out of scope): https://data-apis.org/array-api/latest/purpose_and_scope.html
- array-api-compat: https://data-apis.org/array-api-compat/ · https://github.com/data-apis/array-api-compat

**astropy / pint / unyt**
- astropy Quantity: https://docs.astropy.org/en/stable/units/quantity.html · known issues: https://docs.astropy.org/en/latest/known_issues.html
- astropy equivalencies: https://docs.astropy.org/en/stable/units/equivalencies.html · constants (CODATA): https://docs.astropy.org/en/stable/constants/index.html · constants versioning issue: https://github.com/astropy/astropy/issues/4958
- pint NumPy support (coverage list + PR-driven maintenance): https://pint.readthedocs.io/en/stable/user/numpy.html
- pint temperature/offset units: https://pint.readthedocs.io/en/stable/user/nonmult.html
- pint application registry / pickling: https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html · https://pint.readthedocs.io/en/stable/advanced/serialization.html
- pint NEP-18 wrapping conflict: https://github.com/hgrecco/pint/issues/878
- pint default_en.txt catalog DSL: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
- unyt array module: https://unyt.readthedocs.io/en/stable/modules/unyt.array.html · repo: https://github.com/yt-project/unyt · subclass-rationale issue: https://github.com/yt-project/unyt/issues/15
- scipy.constants versioning / CODATA 2022: https://docs.scipy.org/doc/scipy/reference/constants.html · https://github.com/scipy/scipy/issues/21596

**JAX mechanisms**
- JAX pytrees / custom pytree nodes: https://docs.jax.dev/en/latest/pytrees.html · https://docs.jax.dev/en/latest/custom_pytrees.html
- quax: https://github.com/patrick-kidger/quax · docs https://docs.kidger.site/quax/ · custom rules (Unitful example) https://docs.kidger.site/quax/examples/custom_rules/ · API https://docs.kidger.site/quax/api/quax/ · releases (0.3.x) https://github.com/patrick-kidger/quax/releases · grad sharp-edge issue https://github.com/patrick-kidger/quax/issues/5
- equinox static-field / filter_jit: https://docs.kidger.site/equinox/api/transformations/
- jpu (JAX+pint, curated jpu.numpy, experimental, v0.0.5 Apr 2025): https://github.com/dfm/jpu · https://github.com/dfm/jpu/blob/main/README.md · https://pypi.org/project/jpu/
- unxt (quax+astropy, v1.11.4 Jun 2026, JOSS): https://github.com/GalacticDynamics/unxt · docs https://unxt.readthedocs.io/en/latest/ · API https://unxt.readthedocs.io/en/latest/api/index.html · paper https://arxiv.org/html/2603.08770v1 · backend-abstraction issue https://github.com/GalacticDynamics/unxt/issues/139
- quaxed (pre-quaxified JAX namespace): https://github.com/GalacticDynamics/quaxed
