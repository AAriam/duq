"""The :class:`Dimension` type: exact algebra over the seven SI base dimensions.

A :class:`Dimension` is an immutable, hashable vector of
:class:`fractions.Fraction` exponents over the seven SI base dimensions, held in
SI-brochure order.  Exponent arithmetic is exact, so ``L**Fraction(3, 2)`` is
represented as ``3/2`` and never ``1.4999...``.
"""

from __future__ import annotations

import importlib.resources
import tomllib
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType
from typing import Final

from ._errors import UnitParseError
from ._format import format_composition, order_terms
from ._parse import tokenize

__all__ = ("BASE_NAMES", "BASE_SYMBOLS", "Dimension")

#: Base-dimension symbols in SI-brochure order (time, length, mass, electric
#: current, thermodynamic temperature, amount of substance, luminous intensity).
BASE_SYMBOLS: Final[tuple[str, ...]] = ("T", "L", "M", "I", "Θ", "N", "J")

#: Base-dimension names, aligned with :data:`BASE_SYMBOLS`.
BASE_NAMES: Final[tuple[str, ...]] = (
    "time",
    "length",
    "mass",
    "electric_current",
    "temperature",
    "amount",
    "luminous_intensity",
)

_N = len(BASE_SYMBOLS)
_SYMBOL_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(BASE_SYMBOLS)}
_NAME_INDEX: Final[dict[str, int]] = {n: i for i, n in enumerate(BASE_NAMES)}

# Display order places mass, length and time first (M·L²·T⁻² for energy).
_DISPLAY_ORDER: Final[tuple[int, ...]] = tuple(
    BASE_SYMBOLS.index(s) for s in ("M", "L", "T", "I", "Θ", "N", "J")
)

_ZERO = Fraction(0)


def _coerce_exponent(value: object) -> Fraction:
    """Coerce an exponent to an exact :class:`~fractions.Fraction`.

    Floats are accepted only when they round-trip through
    :meth:`~fractions.Fraction.limit_denominator`, rejecting values such as
    ``0.333`` that are not simple fractions.
    """
    if isinstance(value, bool):
        raise TypeError("exponent must be a real number, not bool")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        exact = Fraction(value)
        simple = exact.limit_denominator(1_000_000)
        if exact == simple:
            return simple
        raise ValueError(f"exponent {value!r} is not an exact simple fraction")
    raise TypeError(f"exponent must be int, float or Fraction, not {type(value).__name__}")


@lru_cache(maxsize=1)
def _derived_catalog() -> Mapping[str, tuple[Fraction, ...]]:
    """Load and cache the named derived dimensions from ``dimensions.toml``."""
    resource = importlib.resources.files("duq.data").joinpath("dimensions.toml")
    with resource.open("rb") as handle:
        raw = tomllib.load(handle)
    catalog: dict[str, tuple[Fraction, ...]] = {}
    for name, entry in raw.get("derived", {}).items():
        exps = [_ZERO] * _N
        for sym, exp in entry.get("exponents", {}).items():
            exps[_SYMBOL_INDEX[sym]] = Fraction(exp)
        catalog[name] = tuple(exps)
    return MappingProxyType(catalog)


class Dimension:
    """An immutable vector of exponents over the seven SI base dimensions.

    Parameters
    ----------
    exponents : mapping of str to int or fractions.Fraction
        A mapping from base-dimension symbol (``"T"``, ``"L"``, ``"M"``,
        ``"I"``, ``"Θ"``, ``"N"``, ``"J"``) or base name (``"time"``,
        ``"length"``, ...) to its exponent.  Omitted bases have exponent zero.

    See Also
    --------
    Dimension.parse : Build a dimension from a string expression.

    Examples
    --------
    >>> from fractions import Fraction
    >>> energy = Dimension({"M": 1, "L": 2, "T": -2})
    >>> str(energy)
    'M·L²·T⁻²'
    >>> energy == Dimension.parse("energy")
    True
    >>> (energy / Dimension({"N": 1})).is_dimensionless
    False
    """

    __slots__ = ("_e",)
    _e: tuple[Fraction, ...]

    def __init__(self, exponents: Mapping[str, int | Fraction | float]) -> None:
        exps = [_ZERO] * _N
        for key, value in exponents.items():
            if key in _SYMBOL_INDEX:
                idx = _SYMBOL_INDEX[key]
            elif key in _NAME_INDEX:
                idx = _NAME_INDEX[key]
            else:
                raise ValueError(f"unknown base dimension {key!r}")
            exps[idx] = _coerce_exponent(value)
        object.__setattr__(self, "_e", tuple(exps))

    @classmethod
    def _from_tuple(cls, exps: tuple[Fraction, ...]) -> Dimension:
        """Construct directly from a validated 7-tuple of exponents."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "_e", exps)
        return obj

    # -- construction -------------------------------------------------------

    @classmethod
    def parse(cls, expression: str) -> Dimension:
        """Parse a dimension from a string expression.

        Parameters
        ----------
        expression : str
            A composition of base symbols/names or named derived dimensions,
            e.g. ``"M.L^2.T^-2"``, ``"mass*length**2/time**2"`` or ``"energy"``.

        Returns
        -------
        Dimension
            The parsed dimension.

        Raises
        ------
        UnitParseError
            If a token is not a known base or derived dimension.

        Examples
        --------
        >>> from fractions import Fraction
        >>> Dimension.parse("L^3/2") == Dimension({"L": Fraction(3, 2)})
        True
        >>> Dimension.parse("velocity") == Dimension({"L": 1, "T": -1})
        True
        """
        derived = _derived_catalog()
        exps = [_ZERO] * _N
        for name, exp in tokenize(expression):
            if name in _SYMBOL_INDEX:
                exps[_SYMBOL_INDEX[name]] += exp
            elif name in _NAME_INDEX:
                exps[_NAME_INDEX[name]] += exp
            elif name in derived:
                for i, base_exp in enumerate(derived[name]):
                    exps[i] += base_exp * exp
            else:
                raise UnitParseError(f"unknown dimension {name!r}")
        return cls._from_tuple(tuple(exps))

    # -- introspection ------------------------------------------------------

    @property
    def exponents(self) -> Mapping[str, Fraction]:
        """Return an immutable mapping of every base symbol to its exponent."""
        return MappingProxyType(dict(zip(BASE_SYMBOLS, self._e, strict=True)))

    @property
    def is_dimensionless(self) -> bool:
        """Return whether every base exponent is zero."""
        return all(e == 0 for e in self._e)

    def _terms(self) -> list[tuple[str, Fraction]]:
        """Return non-zero ``(symbol, exponent)`` terms in display order."""
        return [(BASE_SYMBOLS[i], self._e[i]) for i in _DISPLAY_ORDER if self._e[i] != 0]

    # -- algebra ------------------------------------------------------------

    def __mul__(self, other: object) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension._from_tuple(tuple(a + b for a, b in zip(self._e, other._e, strict=True)))

    def __truediv__(self, other: object) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension._from_tuple(tuple(a - b for a, b in zip(self._e, other._e, strict=True)))

    def __pow__(self, power: object) -> Dimension:
        try:
            exp = _coerce_exponent(power)
        except TypeError:
            return NotImplemented
        return Dimension._from_tuple(tuple(e * exp for e in self._e))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._e == other._e

    def __hash__(self) -> int:
        return hash(("duq.Dimension", self._e))

    # -- rendering ----------------------------------------------------------

    def __str__(self) -> str:
        return format_composition(self._terms(), style="unicode", empty="1")

    def __repr__(self) -> str:
        plain = format_composition(self._terms(), style="plain", empty="1")
        return f"Dimension.parse({plain!r})"

    def format(self, style: str = "unicode") -> str:
        """Render the dimension in the given style.

        Parameters
        ----------
        style : {"unicode", "plain", "latex"}, optional
            The rendering style (default ``"unicode"``).

        Returns
        -------
        str
            The rendered dimension.

        Examples
        --------
        >>> Dimension({"M": 1, "L": 2, "T": -2}).format("plain")
        'M.L^2.T^-2'
        """
        return format_composition(order_terms(self._terms()), style=style, empty="1")
