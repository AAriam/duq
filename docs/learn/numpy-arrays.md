# NumPy arrays

The **same** [`Quantity`][duq.Quantity] class that holds a scalar also wraps a
NumPy array. duq implements NumPy's dispatch protocols (`__array_ufunc__`,
NEP 13, and `__array_function__`, NEP 18) as a **wrapper** — never an `ndarray`
subclass — so units ride through ufuncs, reductions and broadcasting, and are
**never silently dropped**.

`import duq` and scalar arithmetic stay NumPy-free; the array back-end is
imported lazily, only when an array magnitude is involved.

## Creating array quantities

Multiply an array (or list, via `from_array`) by a unit, or construct directly:

```python
import numpy as np
import duq

a = np.array([1.0, 2.0, 3.0]) * duq.units.eV       # ndarray * Unit → Quantity
b = duq.Quantity(np.array([1.0, 2.0]), "nm")        # wrap an existing array
c = duq.Quantity.from_array([1.0, 2.0], "nm")       # from a list
print(a)                                             # [1. 2. 3.] eV
print(a.shape, a.dtype)                              # (3,) float64
```

Array quantities expose `.shape`/`.ndim`/`.size`/`.dtype`/`.T`, support `len()`,
iteration and indexing (each element is itself a `Quantity`):

```python
import numpy as np
import duq

a = np.array([1.0, 2.0, 3.0]) * duq.units.eV
print(a[1])            # 2.0 eV
print(a[1].to("J"))    # 3.204353268e-19 J
```

## Units ride through ufuncs and reductions

```python
import numpy as np
import duq

d = np.array([1.0, 2.0]) * duq.units.m
t = np.array([1.0, 1.0]) * duq.units.s
print(d / t)                       # [1. 2.] m·s⁻¹
print(np.sqrt(d * d).unit)         # m
print((np.array([1.0, 2.0]) * duq.units.m) @ (np.array([3.0, 4.0]) * duq.units.m))   # 11.0 m²
print(d.sum(), d.mean())           # 3.0 m   1.5 m
print(d.var().unit)                # m²   (variance squares the unit)
```

Same-dimension operations convert the right operand to the left unit; joins and
selections do the same:

```python
import numpy as np
import duq

print(np.concatenate([np.array([1.0]) * duq.units.m,
                      np.array([100.0]) * duq.units.cm]))   # [1. 1.] m
print(np.where(np.array([True, False]),
               np.array([1.0, 2.0]) * duq.units.m,
               np.array([100.0, 200.0]) * duq.units.cm))    # [1. 2.] m
```

Comparisons return plain boolean arrays (after conversion):

```python
import numpy as np
import duq

print((np.array([1.0, 2.0]) * duq.units.m) > (np.array([50.0, 250.0]) * duq.units.cm))
# [ True False]
```

## Fail-loud, never silent

Any ufunc/function without a registered unit rule, any `out=` argument, and any
bare-array coercion raise [`UnsupportedOperationError`][duq.UnsupportedOperationError]
naming the operation — closing the classic silent-strip hole:

```python
import numpy as np
import duq

a = np.array([1.0, 2.0, 3.0]) * duq.units.eV
np.asarray(a)     # UnsupportedOperationError: ... use duq.ustrip(unit, q) ...
```

Coercion to a Python scalar follows the same policy as pint: **dimensionless**
quantities coerce (applying the unit scale), everything else raises:

```python
import duq

print(float(duq.Quantity(50.0, "%")))   # 0.5   (dimensionless: scale applied)
float(duq.Quantity(1.0, "m"))            # raises — use duq.ustrip("m", q)
```

!!! note "NumPy 1.26 and 2.x"
    duq supports NumPy ≥ 1.26. A dedicated `test-np126` CI job guards the
    NEP-13/18 behaviour differences (and the `trapz` → `trapezoid` rename)
    between NumPy 1.26 and 2.x.
