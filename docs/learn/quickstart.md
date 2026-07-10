# Quickstart

Install the core (with NumPy support) or add the optional JAX back-end:

```sh
pip install duq            # core + NumPy
pip install "duq[jax]"     # + JAX (jax, quax, equinox)
```

`import duq` imports no array library, so it stays fast and never requires NumPy
or JAX at import time.

## A quantity

A [`Quantity`][duq.Quantity] pairs a magnitude with a [`Unit`][duq.Unit]. Units
are kept **as entered** and arithmetic is **dimension-checked**:

```python
import duq

bond = duq.Quantity(1.0, "kJ/mol")
print(bond)                  # 1.0 kJ·mol⁻¹
print(bond.to("eV/mol"))     # 6.241509074460763e+21 eV·mol⁻¹

speed = duq.Quantity(2.0, "m") / duq.Quantity(4.0, "s")
print(speed)                 # 0.5 m·s⁻¹
```

Mixing incompatible dimensions fails loud instead of guessing:

```python
duq.Quantity(1.0, "m") + duq.Quantity(1.0, "s")
# raises duq.DimensionalityError
```

## Convenient namespaces

`duq.units` and `duq.dims` resolve units and dimensions by attribute:

```python
import duq

print(duq.units.kJ)              # kJ
print(duq.dims.energy)           # M·L²·T⁻²
print(duq.units.metre == duq.unit("m"))   # True
```

## Constants

CODATA-2022 constants are [`Quantity`][duq.Quantity] objects:

```python
import duq

print(duq.constants.k_B)         # 1.380649e-23 J·K⁻¹
print(duq.constants.N_A.value)   # 6.02214076e+23
```

## NumPy and JAX

The very same `Quantity` semantics extend to whole arrays. Multiply an array by
a unit and units ride through every ufunc and reduction:

```python
import numpy as np
import duq

d = np.array([1.0, 2.0, 3.0]) * duq.units.km
print(d.to("m").value)           # [1000. 2000. 3000.]
print(np.sqrt(d * d).unit)       # km
```

With the `duq[jax]` extra the same quantity is safe under `jit`, `grad` and
`vmap`:

```python
import duq.jax

def kinetic(v):
    return 0.5 * duq.jax.Quantity(2.0, "kg") * v * v

g = duq.jax.grad(kinetic)(duq.jax.Quantity(3.0, "m/s"))
print(g.value, g.unit)           # 6.0 kg·m·s⁻¹
```

Continue with [Units & dimensions](units-and-dimensions.md), or jump to
[NumPy arrays](numpy-arrays.md) or [JAX](jax.md).
