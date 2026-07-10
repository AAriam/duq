"""Attribute-style access to dimensions.

``duq.dims.energy`` parses ``"energy"`` and ``duq.dims.mass`` returns the base
mass dimension.

Examples
--------
>>> import duq
>>> duq.dims.energy == duq.dimension("M.L^2.T^-2")
True
>>> duq.dims.mass.is_dimensionless
False
"""

from __future__ import annotations

from ._dimension import Dimension
from ._errors import DuqError

__all__: list[str] = []


def __getattr__(name: str) -> Dimension:
    """Resolve an attribute to a parsed dimension."""
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        return Dimension.parse(name)
    except DuqError as exc:
        raise AttributeError(name) from exc
