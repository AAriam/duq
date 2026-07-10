# Units & dimensions

duq separates two ideas that other libraries often conflate:

- a **[`Dimension`][duq.Dimension]** is *what kind* of physical quantity you
  have — an exact vector of `Fraction` exponents over the seven SI base
  dimensions (time `T`, length `L`, mass `M`, current `I`, temperature `Θ`,
  amount `N`, luminous intensity `J`);
- a **[`Unit`][duq.Unit]** is *how you measure* it — an as-entered composition of
  named atoms (with SI prefixes) that maps to a coherent-SI scale and dimension.

## Dimensions

Parse a dimension from base symbols, base names, or a named derived dimension:

```python
import duq

energy = duq.dimension("M.L^2.T^-2")
print(energy)                                  # M·L²·T⁻²
print(duq.dimension("force") == duq.dimension("M.L.T^-2"))   # True
print(energy.is_dimensionless)                 # False
```

Exponents are exact `Fraction`s, so `L^3/2` stays exact — never `1.4999…`:

```python
import duq
from fractions import Fraction

print(duq.dimension("L^3/2").exponents["L"])   # 3/2
print(duq.dimension("L") ** Fraction(1, 2))    # L^1/2
print(duq.dimension("volume") == duq.dimension("L^3"))   # True
```

`duq.dims` is the attribute-style shortcut (`duq.dims.energy` ==
`duq.dimension("energy")`).

## Units

Parse a unit expression against the default registry. Both the dot syntax
(`kg.m^2.s^-2`) and the Python-ish syntax (`kg*m**2/s**2`, `kJ/mol`) are
accepted:

```python
import duq

u = duq.unit("kJ/mol")
print(u)                       # kJ·mol⁻¹   (as entered)
print(u.dimension)             # M·L²·T⁻²·N⁻¹
print(u.scale)                 # 1000       (to coherent SI)
```

### As-entered composition

`kJ/mol` stays `kJ/mol`; it is never silently collapsed to base units. The
canonical form is available on demand:

```python
import duq

print(duq.unit("kJ/mol").to_coherent_si())     # kg·m²·s⁻²·mol⁻¹
```

### Prefixes are data, not units

Prefixes are resolved at parse time by longest match against prefixable atoms,
so the catalog never eagerly generates `km`, `MJ`, `ns`, … — they always work:

```python
import duq

print(duq.unit("km"))     # km
print(duq.unit("MJ"))     # MJ
print(duq.unit("ns"))     # ns
```

### Display styles

A unit renders in `unicode` (default), `plain` ASCII, or `latex`:

```python
import duq

u = duq.unit("kJ/mol")
print(u.format("unicode"))   # kJ·mol⁻¹
print(u.format("plain"))     # kJ.mol^-1
print(u.format("latex"))     # \mathrm{kJ}\,\mathrm{mol}^{-1}
```

### Affine units are handled honestly

Temperature units with an offset (`°C`, `°F`) are legal only as a bare atom;
duq keeps the scale and offset as separate fields and refuses ambiguous
compositions:

```python
import duq

print(duq.Quantity(0.0, "degC").to("K"))   # 273.15 K
duq.Quantity(0.0, "degC") + duq.Quantity(5.0, "degC")
# raises duq.AffineUnitError — add a delta unit (Δ°C) instead
```

See the [unit catalog](unit-catalog.md) for every bundled unit and the
[API reference](../api/core.md) for the full `Dimension` / `Unit` surface.
