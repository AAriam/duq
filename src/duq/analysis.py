"""Dimensional-analysis helpers built on the pure core.

This is the one core module permitted to use NumPy, and only lazily inside the
composition search.  It provides naming, base decomposition and a search for
equivalent compositions over the catalogue's named dimensions.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING, Final

from ._dimension import BASE_NAMES, BASE_SYMBOLS, Dimension
from ._dimension import _derived_catalog as _derived
from ._errors import AnalysisError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ("compositions", "decompose", "name_of", "shortest_composition")

# Curated pool of named dimensions for the composition search.  Zero-vector
# dimensions (dimensionless/angle/solid_angle) are excluded because they add no
# information; the pool is kept small so the subset search stays fast.
_POOL_DERIVED: Final[tuple[str, ...]] = (
    "force",
    "energy",
    "power",
    "pressure",
    "velocity",
    "acceleration",
    "frequency",
    "electric_charge",
    "momentum",
    "area",
)


@lru_cache(maxsize=1)
def _name_by_vector() -> Mapping[tuple[Fraction, ...], str]:
    """Map each base-exponent vector to a canonical dimension name."""
    mapping: dict[tuple[Fraction, ...], str] = {}
    zero = tuple(Fraction(0) for _ in BASE_SYMBOLS)
    mapping[zero] = "dimensionless"
    for name, sym in zip(BASE_NAMES, BASE_SYMBOLS, strict=True):
        vec = tuple(Fraction(1) if s == sym else Fraction(0) for s in BASE_SYMBOLS)
        mapping.setdefault(vec, name)
    for name, vec in _derived().items():
        mapping.setdefault(vec, name)
    return mapping


@lru_cache(maxsize=1)
def _pool() -> tuple[tuple[str, tuple[Fraction, ...]], ...]:
    """Return the ``(name, vector)`` pool for the composition search."""
    derived = _derived()
    items: list[tuple[str, tuple[Fraction, ...]]] = []
    for name, sym in zip(BASE_NAMES, BASE_SYMBOLS, strict=True):
        vec = tuple(Fraction(1) if s == sym else Fraction(0) for s in BASE_SYMBOLS)
        items.append((name, vec))
    for name in _POOL_DERIVED:
        items.append((name, derived[name]))
    return tuple(items)


def name_of(dim: Dimension) -> str | None:
    """Return the catalogue name of a dimension, or ``None`` if unnamed.

    Parameters
    ----------
    dim : Dimension
        The dimension to name.

    Returns
    -------
    str or None
        The dimension's name, or ``None`` when no named dimension matches.

    Examples
    --------
    >>> import duq
    >>> from duq.analysis import name_of
    >>> name_of(duq.dimension("M.L^2.T^-2"))
    'energy'
    >>> name_of(duq.dimension("M^2.L")) is None
    True
    """
    vec = tuple(dim.exponents[s] for s in BASE_SYMBOLS)
    return _name_by_vector().get(vec)


def decompose(dim: Dimension) -> dict[str, Fraction]:
    """Return the non-zero base-dimension exponents of a dimension.

    Parameters
    ----------
    dim : Dimension
        The dimension to decompose.

    Returns
    -------
    dict of str to fractions.Fraction
        Base-symbol to exponent, for non-zero exponents only.

    Examples
    --------
    >>> import duq
    >>> from duq.analysis import decompose
    >>> decompose(duq.dimension("energy"))
    {'T': Fraction(-2, 1), 'L': Fraction(2, 1), 'M': Fraction(1, 1)}
    """
    return {s: dim.exponents[s] for s in BASE_SYMBOLS if dim.exponents[s] != 0}


def _combinations(n: int, r: int) -> list[tuple[int, ...]]:
    return list(combinations(range(n), r))


def compositions(
    dim: Dimension, max_terms: int = 5, max_exp: int = 3
) -> list[dict[str, Fraction]]:
    """Find equivalent compositions of a dimension over named dimensions.

    A deterministic linear-algebra search returns integer-exponent
    compositions built from a curated pool of named dimensions.

    Parameters
    ----------
    dim : Dimension
        The target dimension.
    max_terms : int, optional
        Maximum number of composing dimensions (default 5).
    max_exp : int, optional
        Maximum absolute exponent for any term (default 3).

    Returns
    -------
    list of dict of str to fractions.Fraction
        Compositions (name to exponent), sorted by term count then total
        absolute exponent.  The trivial single-name identity is excluded.

    Examples
    --------
    >>> import duq
    >>> from fractions import Fraction
    >>> from duq.analysis import compositions
    >>> comps = compositions(duq.dimension("energy"), max_terms=2)
    >>> {"force": Fraction(1), "length": Fraction(1)} in comps
    True
    """
    import numpy as np  # noqa: PLC0415 - lazy: keeps NumPy out of the core import

    pool = _pool()
    target = np.array([float(dim.exponents[s]) for s in BASE_SYMBOLS])
    vectors = np.array([[float(v) for v in vec] for _, vec in pool])
    names = [name for name, _ in pool]
    self_name = name_of(dim)

    seen: set[tuple[tuple[str, int], ...]] = set()
    results: list[dict[str, Fraction]] = []
    n = len(pool)
    for r in range(1, min(max_terms, 7) + 1):
        for subset in _combinations(n, r):
            matrix = vectors[list(subset)].T  # (7, r)
            solution, _residuals, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if rank < r:
                continue
            recon = matrix @ solution
            if not np.allclose(recon, target, atol=1e-9):
                continue
            rounded = np.round(solution).astype(int)
            if not np.allclose(solution, rounded, atol=1e-9):
                continue
            if np.any(np.abs(rounded) > max_exp) or np.all(rounded == 0):
                continue
            comp = {
                names[subset[i]]: Fraction(int(rounded[i])) for i in range(r) if rounded[i] != 0
            }
            if len(comp) == 1 and self_name in comp:
                continue
            key = tuple(sorted((k, int(v)) for k, v in comp.items()))
            if key in seen:
                continue
            seen.add(key)
            results.append(comp)
    results.sort(key=lambda c: (len(c), sum(abs(v) for v in c.values())))
    return results


def shortest_composition(dim: Dimension) -> dict[str, Fraction]:
    """Return the composition with the fewest terms, exactly.

    Parameters
    ----------
    dim : Dimension
        The target dimension.

    Returns
    -------
    dict of str to fractions.Fraction
        The shortest exact composition (name to exponent).  Dimensionless
        returns an empty mapping.

    Raises
    ------
    AnalysisError
        If the dimension has fractional base exponents and no exact
        integer-named composition exists (never a silently wrong answer).

    Examples
    --------
    >>> import duq
    >>> from duq.analysis import shortest_composition
    >>> shortest_composition(duq.dimension("energy"))
    {'energy': Fraction(1, 1)}
    """
    if dim.is_dimensionless:
        return {}
    named = name_of(dim)
    if named is not None and named != "dimensionless":
        return {named: Fraction(1)}

    base = decompose(dim)
    integral = all(v.denominator == 1 for v in base.values())
    candidates: list[dict[str, Fraction]] = []
    if integral:
        candidates.append(base)
    candidates.extend(compositions(dim))
    if not candidates:
        raise AnalysisError(f"no exact integer composition found for dimension {dim!r}")
    candidates.sort(key=lambda c: (len(c), sum(abs(v) for v in c.values())))
    return candidates[0]
