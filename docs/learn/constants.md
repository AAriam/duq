# Constants

`duq.constants` provides fundamental physical constants as
[`Quantity`][duq.Quantity] objects, **versioned by CODATA release**. The default
set is CODATA 2022, reached by attribute access on the module or explicitly via
`codata`.

```python
import duq

print(duq.constants.k_B)          # 1.380649e-23 J·K⁻¹
print(duq.constants.N_A.value)    # 6.02214076e+23
print(duq.constants.h.unit.dimension == duq.dimension("action"))   # True
```

Because each constant is a real `Quantity`, it participates in dimensional
arithmetic and conversion:

```python
import duq

thermal = duq.constants.k_B * duq.Quantity(300.0, "K")
print(thermal.to("meV"))          # ~25.85 meV  (k_B·T at room temperature)
```

## Selecting a CODATA release

```python
import duq

s = duq.constants.codata(2022)
print(s.year)                     # 2022
print(s.c.value)                  # 299792458.0
```

## Metadata

`info` exposes the raw catalogue entry (value, unit, uncertainty, symbol, name):

```python
import duq

info = duq.constants.codata(2022).info("h")
print(info["name"])               # Planck constant
print(info["uncertainty"])        # exact
```

The bundled CODATA-2022 slugs include `c`, `h`, `hbar`, `e`, `k_B`, `N_A`, `R`,
`G`, `m_e`, `m_p`, `m_n`, `u`, `alpha`, `eps_0`, `mu_0`, `a_0`, `E_h`, `R_inf`,
`sigma_SB` and `g_n`; every value is verified against the NIST CODATA-2022
recommended values. See the [API reference](../api/constants.md) for the full
surface.
