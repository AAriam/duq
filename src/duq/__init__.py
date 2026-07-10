"""duq: dimensions, units and quantities for scientific computing.

The top-level namespace re-exports the public API.  The pure core imports no
array library, so ``import duq`` stays fast and never requires NumPy or JAX.

Examples
--------
>>> import duq
>>> q = duq.Quantity(1.0, "kJ/mol")
>>> q.to("eV/mol").unit.dimension == duq.dimension("molar_energy")
True
>>> duq.constants.k_B.unit == duq.unit("J/K")
True
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
from ._quantity import Quantity, uconvert, ustrip
from ._registry import UnitRegistry, default_registry
from ._unit import Unit

__all__ = (
    "AffineUnitError",
    "AnalysisError",
    "Dimension",
    "DimensionalityError",
    "DuqError",
    "Quantity",
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
