"""Attribute-style access to units from the default registry.

``duq.units.kJ`` parses ``"kJ"`` against the default registry, so any prefixed
or aliased unit that is a valid Python identifier is reachable as an attribute.
Symbols that are not identifiers (``°C``, ``%``) are reached via
:func:`duq.unit`.

Examples
--------
>>> import duq
>>> str(duq.units.kJ)
'kJ'
>>> duq.units.metre == duq.unit("m")
True
"""

from __future__ import annotations

from ._errors import DuqError
from ._registry import default_registry
from ._unit import Unit

__all__: list[str] = []


def __getattr__(name: str) -> Unit:
    """Resolve an attribute to a unit from the default registry."""
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        return default_registry.unit(name)
    except DuqError as exc:
        for atom in default_registry.atoms:
            if atom.slug == name:
                return default_registry.unit(atom.symbol)
        raise AttributeError(name) from exc


def __dir__() -> list[str]:
    return sorted({atom.slug for atom in default_registry.atoms})
