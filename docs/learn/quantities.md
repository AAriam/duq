# Quantities

A [`Quantity`][duq.Quantity] is an **immutable** pairing of a magnitude with a
[`Unit`][duq.Unit]. The core `Quantity` holds a Python scalar
(`int`/`float`/`complex`/`Fraction`/`Decimal`); the same class also wraps NumPy
arrays (see [NumPy arrays](numpy-arrays.md)), and [`duq.jax.Quantity`](jax.md)
mirrors it for JAX.

```python
import duq

q = duq.Quantity(1.5, "kJ/mol")
print(q.value)        # 1.5
print(q.unit)         # kJ·mol⁻¹
print(q.dimension)    # M·L²·T⁻²·N⁻¹
```

## Conversion

`to` returns a new quantity in another unit; `value_in` returns just the
magnitude; `to_si` and `compact` are convenience canonicalisers:

```python
import duq

d = duq.Quantity(1500.0, "m")
print(d.to("km"))                    # 1.5 km
print(d.value_in("cm"))              # 150000.0
print(duq.Quantity(1.0, "kJ/mol").to_si())   # 1000.0 kg·m²·s⁻²·mol⁻¹
print(d.compact())                   # 1.5 km  (nearest sensible SI prefix)
```

## Arithmetic is dimension-checked

Addition auto-converts a compatible right operand to the left operand's unit;
incompatible dimensions raise:

```python
import duq

print(duq.Quantity(1.0, "m") + duq.Quantity(50.0, "cm"))   # 1.5 m
duq.Quantity(1.0, "m") + duq.Quantity(1.0, "s")            # DimensionalityError
```

Adding a bare number to a dimensional quantity is also an error (never a silent
"assume same unit"):

```python
import duq

duq.Quantity(1.0, "m") + 2.0   # raises duq.DimensionalityError
```

## Equality is exact

Unlike the prototype's tolerance-based `==` (which broke transitivity and the
hash contract), `==` is **exact after conversion**, and quantities are hashable.
For tolerant comparison, use `allclose` explicitly:

```python
import duq

print(duq.Quantity(1.0, "m") == duq.Quantity(100.0, "cm"))       # True
print(duq.Quantity(1.0, "m").allclose(duq.Quantity(100.001, "cm"), rel_tol=1e-3))   # True
```

## Molar equivalence (opt-in)

Converting between a per-amount quantity and an absolute one goes through
Avogadro's number — but only when you **ask** for it, so it can never fire
silently:

```python
import duq

per_mol = duq.Quantity(10.0, "kJ/mol")
print(per_mol.to("J", equivalence="molar"))   # 1.66…e-20 J  (÷ N_A)
per_mol.to("J")                                # raises DimensionalityError
```

## Free functions

`duq.uconvert` and `duq.ustrip` are unit-first helpers (JAX-idiomatic) that work
on every quantity flavour:

```python
import duq

q = duq.Quantity(1.0, "m")
print(duq.uconvert("mm", q))   # 1000.0 mm
print(duq.ustrip("cm", q))     # 100.0
```
