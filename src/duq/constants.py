"""CODATA-versioned fundamental physical constants as :class:`Quantity` objects.

Constants are keyed by CODATA release.  The default set is CODATA 2022, reached
by attribute access on this module (``duq.constants.k_B``) or explicitly via
:func:`codata`.

Examples
--------
>>> import duq
>>> duq.constants.N_A.value
6.02214076e+23
>>> duq.constants.codata(2022).c.value
299792458.0
"""

from __future__ import annotations

import importlib.resources
import tomllib
from functools import cache
from types import MappingProxyType
from typing import Final

from ._quantity import Quantity

__all__ = ("codata",)

_DEFAULT_YEAR: Final[int] = 2022


@cache
def _load() -> MappingProxyType[str, MappingProxyType[str, dict[str, str]]]:
    resource = importlib.resources.files("duq.data").joinpath("constants.toml")
    with resource.open("rb") as handle:
        raw = tomllib.load(handle)
    sets: dict[str, MappingProxyType[str, dict[str, str]]] = {}
    for set_key, entries in raw.items():
        assert isinstance(entries, dict)
        sets[set_key] = MappingProxyType(
            {slug: {str(k): str(v) for k, v in entry.items()} for slug, entry in entries.items()}
        )
    return MappingProxyType(sets)


class ConstantSet:
    """A named set of physical constants for a single CODATA release.

    Attribute access returns a fresh :class:`Quantity`; metadata (uncertainty,
    symbol, name) is available through :meth:`info`.

    Examples
    --------
    >>> import duq
    >>> s = duq.constants.codata(2022)
    >>> s.h.unit.dimension == duq.dimension("action")
    True
    >>> s.info("h")["uncertainty"]
    'exact'
    """

    def __init__(self, year: int, entries: MappingProxyType[str, dict[str, str]]) -> None:
        self._year = year
        self._entries = entries

    @property
    def year(self) -> int:
        """Return the CODATA release year of this set."""
        return self._year

    def info(self, name: str) -> MappingProxyType[str, str]:
        """Return the raw metadata (value, unit, uncertainty, symbol, name).

        Parameters
        ----------
        name : str
            The constant's slug (e.g. ``"k_B"``).

        Returns
        -------
        types.MappingProxyType
            The metadata mapping.
        """
        try:
            return MappingProxyType(self._entries[name])
        except KeyError:
            raise AttributeError(f"no constant {name!r} in CODATA {self._year}") from None

    def quantity(self, name: str) -> Quantity:
        """Return a constant as a :class:`Quantity`.

        Parameters
        ----------
        name : str
            The constant's slug.

        Returns
        -------
        Quantity
            The constant, with its CODATA value and unit.
        """
        entry = self.info(name)
        return Quantity(float(entry["value"]), entry["unit"])

    def __getattr__(self, name: str) -> Quantity:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.quantity(name)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self._entries.keys()]


@cache
def codata(year: int = _DEFAULT_YEAR) -> ConstantSet:
    """Return the constant set for a CODATA release year.

    Parameters
    ----------
    year : int, optional
        The CODATA release year (default 2022).

    Returns
    -------
    ConstantSet
        The set of constants for that year.

    Raises
    ------
    ValueError
        If no data is bundled for the requested year.

    Examples
    --------
    >>> import duq
    >>> duq.constants.codata(2022).e.value
    1.602176634e-19
    """
    key = f"codata{year}"
    data = _load()
    if key not in data:
        raise ValueError(f"no bundled CODATA data for year {year}")
    return ConstantSet(year, data[key])


def __getattr__(name: str) -> Quantity:
    """Resolve module-level attribute access to the default CODATA set."""
    if name.startswith("_"):
        raise AttributeError(name)
    return codata(_DEFAULT_YEAR).quantity(name)


def __dir__() -> list[str]:
    return [*__all__, *codata(_DEFAULT_YEAR)._entries.keys()]
