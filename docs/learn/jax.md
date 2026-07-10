# JAX

The `duq[jax]` extra adds [`duq.jax.Quantity`][duq.jax.Quantity]: a JAX-array
magnitude coupled with a **static** [`Unit`][duq.Unit]. Units are checked and
propagated at **trace time** through [quax](https://github.com/patrick-kidger/quax)
primitive interception, so a compiled kernel pays **zero runtime cost** for its
units. `import duq` never imports JAX; this back-end requires
`pip install "duq[jax]"`.

```python
import duq.jax

q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
print(q)                 # [1. 2. 3.] m
print(q.to("cm").value)  # [100. 200. 300.]
```

## Operators and `duq.jax.numpy`

Operators work eagerly and under every transformation. `duq.jax.numpy` mirrors
`jax.numpy` with unit awareness — quantity operands dispatch through the duq
rules, plain arrays behave exactly as in `jax.numpy`:

```python
import duq.jax
import duq.jax.numpy as djnp

q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
print((q + q).value)             # [2. 4. 6.]
print(str(djnp.sqrt(q * q).unit))  # m
```

An indexed-update helper mirrors `jax.numpy`'s `.at[...]`, converting the update
to the operand's unit:

```python
import duq.jax

q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
print(q.at[0].set(duq.jax.Quantity(500.0, "cm")).value)   # [5. 2. 3.]
```

## Transformations

A quantity is a **pytree** whose magnitude is the traced leaf and whose unit is
static metadata, so plain `jax.jit`, `jax.vmap` and `jax.lax.scan` accept
quantity arguments directly:

```python
import jax
import duq.jax

q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
print(jax.jit(lambda x: x * x)(q).value)   # [1. 4. 9.]
```

[`duq.jax.grad`][duq.jax.grad] (and `jacfwd`/`jacrev`/`hessian`) additionally
label derivative outputs with the exact `unit_out / unit_in` unit:

```python
import duq.jax

def energy(x):                       # a spring: E = ½ k x²
    k = duq.jax.Quantity(2.0, "N/m")
    return 0.5 * k * x * x

g = duq.jax.grad(energy)(duq.jax.Quantity(0.1, "m"))
h = duq.jax.hessian(energy)(duq.jax.Quantity(0.1, "m"))
print(g.value, g.unit)   # 0.2 N        (force = -dE/dx)
print(h.value, h.unit)   # 2.0 N·m⁻¹    (stiffness = d²E/dx²)
```

## Retracing and hot kernels

The unit is part of the pytree structure, hence part of `jit`'s cache key:
calling a jitted function with metres and then with kilometres compiles twice.
Call sites with a fixed unit compile once and are reused. For hot kernels, strip
to a canonical unit **before** entering the kernel:

```python
import duq, duq.jax

q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
mag = duq.ustrip("m", q)             # a plain jax.Array — no unit bookkeeping
out = duq.jax.Quantity(mag * 2.0, "m/s")
```

## Bare (unit-less) operands: a NumPy ↔ JAX difference

There is one deliberate, documented semantic difference between the two
back-ends, in **selection/update** operations that mix a quantity with a **bare
(unit-less)** operand — `where`, `pad`, `scatter`, `dynamic_update_slice`:

- The **NumPy** layer is strict: combining a bare array with a dimensional
  quantity raises [`DimensionalityError`][duq.DimensionalityError]. A bare
  number never silently acquires a unit.
- The **`duq.jax`** layer lets the bare operand **adopt** the quantity's unit.

```python
import numpy as np, jax.numpy as jnp
import duq, duq.jax
import duq.jax.numpy as djnp

# NumPy: strict — the bare [3., 4.] has no unit, so this raises.
np.where([True, False], duq.Quantity(np.array([1.0, 2.0]), "m"), np.array([3.0, 4.0]))
# duq.DimensionalityError: cannot combine a bare array/number with a quantity …

# duq.jax: the bare operand adopts metres.
djnp.where(jnp.array([True, False]),
           duq.jax.Quantity(jnp.array([1.0, 2.0]), "m"),
           jnp.array([3.0, 4.0]))
# Quantity([1., 4.], "m")   ← the bare [3., 4.] was taken as metres
```

This is a constraint of primitive-level interception, not a choice: after
tracing, a `jit`-internal literal (e.g. a constant JAX inserts) is
indistinguishable from a user-provided bare array, so the JAX layer cannot
reject bare operands without breaking legitimate compiled code. `unxt` behaves
the same way. When you need the strict check under JAX, wrap the bare operand in
a `duq.jax.Quantity` yourself. See
[`duq.jax` primitive coverage](../dev/jax_coverage.md) for the per-primitive
rules.

## Known limitations

- `jax.custom_vjp` is not supported by quax (`custom_jvp` is); strip units around
  a `custom_vjp` boundary with [`duq.ustrip`][duq.ustrip].
- Uncovered primitives fail loud with
  [`UnsupportedOperationError`][duq.UnsupportedOperationError] naming the
  primitive; see [the coverage table](../dev/jax_coverage.md) for the covered set
  and the known exclusions (`lax.linalg`, FFTs, convolutions, …).
