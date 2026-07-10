"""duq: dimensions, units and quantities for scientific computing.

A lean dimensional core carries units through **unit-carrying NumPy arrays**
*and* **jit/grad/vmap-safe JAX arrays** — the same :class:`Quantity` semantics on
both back-ends.  Units compose **as entered** (``kJ/mol`` stays ``kJ/mol``),
exponents are exact ``fractions.Fraction`` objects, conversions are **fail-loud**
(a dimensional mismatch raises; units are never silently dropped), and the
package ships CODATA-2022 constants and first-class dimensional analysis.

``import duq`` imports **no** array library, so it stays fast and never requires
NumPy or JAX at import time; the JAX back-end lives behind the ``duq[jax]`` extra.

A 30-second tour
----------------
>>> import duq

Build a quantity; units are kept as entered and arithmetic is dimension-checked:

>>> q = duq.Quantity(1.0, "kJ/mol")
>>> str(q.to("eV/mol").unit)
'eV·mol⁻¹'
>>> (duq.Quantity(2.0, "m") / duq.Quantity(4.0, "s")).value
0.5

Reach units and dimensions by attribute, and CODATA-2022 constants as quantities:

>>> str(duq.units.kJ), str(duq.dims.energy)
('kJ', 'M·L²·T⁻²')
>>> duq.constants.k_B.unit == duq.unit("J/K")
True

The same :class:`Quantity` wraps NumPy arrays; add the ``duq[jax]`` extra and
``import duq.jax`` for the ``jit``/``grad``/``vmap``-safe flavour.  Molar
equivalence (``kJ/mol`` to ``J`` via Avogadro's number) is opt-in:

>>> q.to("eV/mol").unit.dimension == duq.dimension("molar_energy")
True

See :mod:`duq.analysis`, :mod:`duq.constants`, :mod:`duq.jax`, and
:func:`duq.uconvert` / :func:`duq.ustrip`.
"""

from __future__ import annotations

from importlib import metadata

from . import analysis, constants, dims, units
from ._dimension import Dimension
from ._errors import (
    AffineUnitError,
    AnalysisError,
    DimensionalityError,
    DuqError,
    RegistryMismatchError,
    UndefinedUnitError,
    UnitParseError,
    UnsupportedOperationError,
)
from ._quantity import Quantity, QuantityLike, uconvert, ustrip
from ._registry import UnitRegistry, default_registry
from ._unit import Unit

__all__ = (
    "AffineUnitError",
    "AnalysisError",
    "Dimension",
    "DimensionalityError",
    "DuqError",
    "Quantity",
    "QuantityLike",
    "RegistryMismatchError",
    "UndefinedUnitError",
    "Unit",
    "UnitParseError",
    "UnitRegistry",
    "UnsupportedOperationError",
    "__version__",
    "analysis",
    "constants",
    "default_registry",
    "dimension",
    "dims",
    "uconvert",
    "unit",
    "units",
    "ustrip",
)

try:
    __version__ = metadata.version("duq")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkouts
    __version__ = "0.0.0.dev0"


def unit(expression: str) -> Unit:
    """Parse a unit expression against the default registry.

    Parameters
    ----------
    expression : str
        The unit expression, e.g. ``"kJ/mol"``.

    Returns
    -------
    Unit
        The parsed unit.

    Examples
    --------
    >>> import duq
    >>> str(duq.unit("kg.m/s^2"))
    'kg·m·s⁻²'
    """
    return default_registry.unit(expression)


def dimension(expression: str) -> Dimension:
    """Parse a dimension expression.

    Parameters
    ----------
    expression : str
        The dimension expression, e.g. ``"energy"`` or ``"M.L^2.T^-2"``.

    Returns
    -------
    Dimension
        The parsed dimension.

    Examples
    --------
    >>> import duq
    >>> duq.dimension("force") == duq.dimension("M.L.T^-2")
    True
    """
    return Dimension.parse(expression)
