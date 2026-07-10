"""The :class:`Unit` type and its internal :class:`UnitAtom` building block.

A :class:`Unit` is an immutable, as-entered composition of prefixed unit atoms.
``kJ/mol`` is stored as the two factors ``kJ`` and ``mol⁻¹`` and is *displayed*
that way; it is never silently collapsed to base SI units (use
:meth:`Unit.to_coherent_si` for that).  Derived quantities such as ``scale``,
``offset`` and ``dimension`` are computed once and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from ._dimension import Dimension, _coerce_exponent
from ._errors import AffineUnitError, RegistryMismatchError
from ._format import format_composition, order_terms

if TYPE_CHECKING:
    from ._registry import UnitRegistry

__all__ = ("Unit",)

Scalar = Fraction | float
"""A scale or offset value: exact when a :class:`~fractions.Fraction`."""


@dataclass(frozen=True, slots=True)
class Prefix:
    """An SI decimal prefix (internal)."""

    name: str
    symbol: str
    exponent10: int


@dataclass(frozen=True, slots=True)
class UnitAtom:
    """A single catalogue unit (internal); prefixes are applied separately."""

    slug: str
    symbol: str
    aliases: tuple[str, ...]
    dimension: Dimension
    scale: Scalar
    offset: Scalar
    prefixable: bool
    kind: str  # "linear" | "affine" | "delta"


@dataclass(frozen=True, slots=True)
class Factor:
    """One term of a unit composition: an optional prefix, atom and exponent."""

    prefix: Prefix | None
    atom: UnitAtom
    exponent: Fraction

    @property
    def symbol(self) -> str:
        """Return the rendered symbol, e.g. ``"kJ"`` for kilo + joule."""
        pre = self.prefix.symbol if self.prefix is not None else ""
        return f"{pre}{self.atom.symbol}"


def _pow_scalar(base: Scalar, exponent: Fraction) -> Scalar:
    """Raise ``base`` to ``exponent``, staying exact when possible."""
    if isinstance(base, Fraction) and exponent.denominator == 1:
        exact: Scalar = base**exponent.numerator
        return exact
    inexact: float = float(base) ** float(exponent)
    return inexact


def _merge(factors: tuple[Factor, ...]) -> tuple[Factor, ...]:
    """Combine like factors, summing exponents and dropping zero results."""
    order: list[tuple[str | None, str]] = []
    grouped: dict[tuple[str | None, str], Factor] = {}
    for factor in factors:
        key = (factor.prefix.symbol if factor.prefix else None, factor.atom.slug)
        if key in grouped:
            existing = grouped[key]
            grouped[key] = Factor(
                existing.prefix, existing.atom, existing.exponent + factor.exponent
            )
        else:
            grouped[key] = factor
            order.append(key)
    return tuple(grouped[key] for key in order if grouped[key].exponent != 0)


class Unit:
    """An immutable, as-entered composition of prefixed unit atoms.

    Units are not constructed directly; obtain them from a registry via
    :func:`duq.unit`, attribute access on :mod:`duq.units`, or by combining
    existing units with ``*``, ``/`` and ``**``.

    Examples
    --------
    >>> import duq
    >>> u = duq.unit("kJ/mol")
    >>> str(u)
    'kJ·mol⁻¹'
    >>> u.dimension == duq.dimension("energy/amount")
    True
    >>> duq.unit("J") == duq.unit("kg.m^2/s^2")
    True
    """

    __slots__ = ("_dimension", "_factors", "_offset", "_registry", "_scale")
    _factors: tuple[Factor, ...]
    _registry: UnitRegistry
    _dimension: Dimension
    _scale: Scalar
    _offset: Scalar

    def __init__(self) -> None:  # pragma: no cover - constructed via _create
        raise TypeError("Unit objects are created by a UnitRegistry, not directly")

    @classmethod
    def _create(cls, registry: UnitRegistry, factors: tuple[Factor, ...]) -> Unit:
        """Validate and build a unit from a tuple of factors."""
        factors = _merge(factors)
        affine = [f for f in factors if f.atom.kind == "affine"]
        if affine and not (
            len(factors) == 1 and factors[0].exponent == 1 and factors[0].prefix is None
        ):
            raise AffineUnitError(
                f"affine unit {affine[0].atom.symbol!r} may only appear alone "
                "with exponent 1 and no prefix"
            )
        obj = object.__new__(cls)
        object.__setattr__(obj, "_registry", registry)
        object.__setattr__(obj, "_factors", factors)
        object.__setattr__(obj, "_dimension", cls._compute_dimension(factors))
        object.__setattr__(obj, "_scale", cls._compute_scale(factors))
        object.__setattr__(obj, "_offset", cls._compute_offset(factors))
        return obj

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("Unit is immutable")

    @staticmethod
    def _compute_dimension(factors: tuple[Factor, ...]) -> Dimension:
        dim = Dimension({})
        for factor in factors:
            dim = dim * (factor.atom.dimension**factor.exponent)
        return dim

    @staticmethod
    def _compute_scale(factors: tuple[Factor, ...]) -> Scalar:
        scale: Scalar = Fraction(1)
        for factor in factors:
            base: Scalar = factor.atom.scale
            if factor.prefix is not None:
                base = base * (Fraction(10) ** factor.prefix.exponent10)
            scale = scale * _pow_scalar(base, factor.exponent)
        return scale

    @staticmethod
    def _compute_offset(factors: tuple[Factor, ...]) -> Scalar:
        if (
            len(factors) == 1
            and factors[0].atom.kind == "affine"
            and factors[0].exponent == 1
            and factors[0].prefix is None
        ):
            return factors[0].atom.offset
        return Fraction(0)

    # -- properties ---------------------------------------------------------

    @property
    def registry(self) -> UnitRegistry:
        """Return the registry that owns this unit."""
        return self._registry

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimension of this unit."""
        return self._dimension

    @property
    def scale(self) -> Scalar:
        """Return the multiplicative factor to the coherent SI unit."""
        return self._scale

    @property
    def offset(self) -> Scalar:
        """Return the additive offset to coherent SI (non-zero only for affine)."""
        return self._offset

    @property
    def is_dimensionless(self) -> bool:
        """Return whether the unit's dimension is dimensionless."""
        return self._dimension.is_dimensionless

    @property
    def is_affine(self) -> bool:
        """Return whether the unit is a bare affine unit (e.g. ``°C``)."""
        return self._offset != 0

    @property
    def factors(self) -> tuple[Factor, ...]:
        """Return the ordered factors composing this unit (internal detail)."""
        return self._factors

    # -- algebra ------------------------------------------------------------

    def _check_registry(self, other: Unit) -> None:
        if other._registry is not self._registry:
            raise RegistryMismatchError("cannot combine units from different registries")

    def __mul__(self, other: object) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        self._check_registry(other)
        return Unit._create(self._registry, self._factors + other._factors)

    def __truediv__(self, other: object) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        self._check_registry(other)
        inverted = tuple(Factor(f.prefix, f.atom, -f.exponent) for f in other._factors)
        return Unit._create(self._registry, self._factors + inverted)

    def __pow__(self, power: object) -> Unit:
        try:
            exp = _coerce_exponent(power)
        except TypeError:
            return NotImplemented
        powered = tuple(Factor(f.prefix, f.atom, f.exponent * exp) for f in self._factors)
        return Unit._create(self._registry, powered)

    def __rtruediv__(self, other: object) -> Unit:
        if other != 1:
            return NotImplemented
        return self**-1

    # -- equality -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return (
            self._dimension == other._dimension
            and self._scale == other._scale
            and self._offset == other._offset
        )

    def __hash__(self) -> int:
        return hash(("duq.Unit", self._dimension, self._scale, self._offset))

    def same_expression(self, other: Unit) -> bool:
        """Return whether two units are the *same as-entered expression*.

        Unlike ``==`` (which compares dimension and scale), this compares the
        literal factor composition, so ``J`` and ``kg·m²/s²`` are unequal here.

        Parameters
        ----------
        other : Unit
            The unit to compare against.

        Returns
        -------
        bool
            Whether the factor compositions are identical.

        Examples
        --------
        >>> import duq
        >>> duq.unit("J").same_expression(duq.unit("kg.m^2/s^2"))
        False
        >>> duq.unit("kJ/mol").same_expression(duq.unit("kJ/mol"))
        True
        """
        mine = [(f.prefix, f.atom.slug, f.exponent) for f in self._factors]
        theirs = [(f.prefix, f.atom.slug, f.exponent) for f in other._factors]
        return mine == theirs

    # -- conversions --------------------------------------------------------

    def to_coherent_si(self) -> Unit:
        """Return the equivalent unit expressed in coherent SI base units.

        Returns
        -------
        Unit
            The unit rebuilt from base SI atoms (``s``, ``m``, ``kg``, ``A``,
            ``K``, ``mol``, ``cd``), with unit scale.

        Examples
        --------
        >>> import duq
        >>> str(duq.unit("kJ/mol").to_coherent_si())
        'kg·m²·s⁻²·mol⁻¹'
        """
        return self._registry.coherent_unit(self._dimension)

    # -- rendering ----------------------------------------------------------

    def _terms(self) -> list[tuple[str, Fraction]]:
        return [(f.symbol, f.exponent) for f in self._factors]

    def __str__(self) -> str:
        return format_composition(self._terms(), style="unicode", empty="1")

    def __repr__(self) -> str:
        return f"Unit({format_composition(self._terms(), style='plain', empty='1')!r})"

    def format(self, style: str = "unicode") -> str:
        """Render the unit in the given style.

        Parameters
        ----------
        style : {"unicode", "plain", "latex"}, optional
            The rendering style (default ``"unicode"``).

        Returns
        -------
        str
            The rendered unit expression.

        Examples
        --------
        >>> import duq
        >>> duq.unit("kJ/mol").format("plain")
        'kJ.mol^-1'
        """
        return format_composition(order_terms(self._terms()), style=style, empty="1")
