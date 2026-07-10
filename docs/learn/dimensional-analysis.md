# Dimensional analysis

`duq.analysis` rebuilds the prototype's genuinely novel features on the clean
core: name a dimension, decompose it, and search for equivalent compositions
over the catalogue's named dimensions. NumPy is imported lazily, only inside the
composition search, so `import duq` stays array-free.

## Naming and decomposition

```python
import duq
from duq.analysis import name_of, decompose

print(name_of(duq.dimension("M.L^2.T^-2")))   # energy
print(name_of(duq.dimension("M^2.L")))        # None  (unnamed)
print(decompose(duq.dimension("energy")))
# {'T': Fraction(-2, 1), 'L': Fraction(2, 1), 'M': Fraction(1, 1)}
```

## Equivalent compositions

`compositions` returns integer-exponent rewrites of a dimension in terms of
named dimensions, sorted by term count then total exponent:

```python
import duq
from fractions import Fraction
from duq.analysis import compositions

comps = compositions(duq.dimension("energy"), max_terms=2)
print({"force": Fraction(1), "length": Fraction(1)} in comps)   # True
print(comps[0])   # {'time': Fraction(1, 1), 'power': Fraction(1, 1)}
```

## Shortest composition — exact or nothing

`shortest_composition` returns the fewest-term exact composition. Crucially, it
either solves the problem exactly or **raises** — it never returns a
silently-wrong answer (the prototype's greedy loop could):

```python
import duq
from duq.analysis import shortest_composition

print(shortest_composition(duq.dimension("energy")))     # {'energy': Fraction(1, 1)}
print(shortest_composition(
    duq.dimension("force") * duq.dimension("velocity")))  # {'power': Fraction(1, 1)}

# A dimension with fractional base exponents and no integer-named composition:
shortest_composition(duq.dimension("L^1/2"))   # raises duq.AnalysisError
```

See the [API reference](../api/analysis.md) for the full signatures.
