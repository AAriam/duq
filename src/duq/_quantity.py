"""The scalar :class:`Quantity`: an immutable value paired with a unit.

``Quantity`` couples a Python scalar (``int``, ``float``, ``complex``,
:class:`~fractions.Fraction` or :class:`~decimal.Decimal`) with a
:class:`~duq._unit.Unit`.  Arithmetic checks dimensions, addition converts the
right operand to the left operand's unit, equality is *exact* after conversion,
and everything is immutable and hashable.

Importing this module never imports NumPy; the NumPy protocol hooks are defined
but deliberately raise until the array layer ships.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final

from ._dimension import Dimension, _coerce_exponent
from ._errors import (
    AffineUnitError,
    DimensionalityError,
    UnsupportedOperationError,
)
from ._registry import default_registry
from ._rules import Rule, apply
from ._unit import Factor, Unit

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ("Quantity", "uconvert", "ustrip")

Number = int | float | complex | Fraction | Decimal
"""The scalar value types accepted by :class:`Quantity`."""

Scalar = Fraction | float

#: Avogadro constant as an exact integer Fraction (CODATA 2022, exact by SI).
_AVOGADRO: Final[Fraction] = Fraction(602214076 * 10**15)

_NUMPY_MESSAGE = "numpy support lands in duq.numpy (PR 3)"


def _to_decimal(x: Scalar | int) -> Decimal:
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(repr(x))


def _as_complex(factor: Scalar) -> complex | float:
    """Coerce a scale factor to something a complex value can combine with."""
    return complex(float(factor)) if isinstance(factor, Fraction) else factor


def _to_float(value: Number) -> float:
    """Return a real float magnitude (the modulus for complex values)."""
    if isinstance(value, complex):
        return abs(value)
    return float(value)


def _mul(value: Number, factor: Scalar) -> Number:
    if isinstance(value, Decimal):
        return value * _to_decimal(factor)
    if isinstance(value, complex):
        return value * _as_complex(factor)
    return value * factor


def _add(value: Number, addend: Scalar) -> Number:
    if addend == 0:
        return value
    if isinstance(value, Decimal):
        return value + _to_decimal(addend)
    if isinstance(value, complex):
        return value + _as_complex(addend)
    return value + addend


def _sub(value: Number, subtrahend: Scalar) -> Number:
    if subtrahend == 0:
        return value
    if isinstance(value, Decimal):
        return value - _to_decimal(subtrahend)
    if isinstance(value, complex):
        return value - _as_complex(subtrahend)
    return value - subtrahend


def _div(value: Number, divisor: Scalar) -> Number:
    if divisor == 1:
        return value
    if isinstance(value, Decimal):
        return value / _to_decimal(divisor)
    if isinstance(value, complex):
        return value / _as_complex(divisor)
    return value / divisor


def _avogadro_power(power: Fraction) -> Scalar:
    if power.denominator == 1:
        exact: Scalar = _AVOGADRO**power.numerator
        return exact
    inexact: float = float(_AVOGADRO) ** float(power)
    return inexact


# Value-by-value arithmetic across the numeric tower.  Mixing incompatible
# concrete types (e.g. a Decimal quantity with a float one) raises TypeError at
# runtime, which is the intended behaviour; the localized ignores acknowledge
# that mypy cannot prove the operands share a concrete type.
def _num_add(a: Number, b: Number) -> Number:
    return a + b  # type: ignore[operator]


def _num_sub(a: Number, b: Number) -> Number:
    return a - b  # type: ignore[operator]


def _num_mul(a: Number, b: Number) -> Number:
    return a * b  # type: ignore[operator]


def _num_div(a: Number, b: Number) -> Number:
    return a / b  # type: ignore[operator]


def _num_pow(a: Number, b: int | Fraction) -> Number:
    return a**b  # type: ignore[operator]


def _num_abs(a: Number) -> Number:
    return abs(a)  # type: ignore[return-value]


def _unit_is_affine(unit: Unit) -> bool:
    """Return whether a unit is affine (non-zero offset or an affine atom)."""
    return unit.offset != 0 or any(f.atom.kind == "affine" for f in unit.factors)


class Quantity:
    """An immutable scalar value with a physical unit.

    Parameters
    ----------
    value : int, float, complex, fractions.Fraction or decimal.Decimal
        The numeric magnitude.
    unit : str or Unit
        The unit; strings are parsed against the default registry.

    Examples
    --------
    >>> import duq
    >>> q = duq.Quantity(1.5, "kJ/mol")
    >>> q.to("J/mol").value
    1500.0
    >>> (duq.Quantity(2.0, "m") + duq.Quantity(30.0, "cm")).value
    2.3
    >>> duq.Quantity(1.0, "km") == duq.Quantity(1000.0, "m")
    True
    """

    __slots__ = ("_unit", "_value")
    _value: Number
    _unit: Unit

    def __init__(self, value: Number, unit: str | Unit) -> None:
        if isinstance(value, bool) or not isinstance(
            value, int | float | complex | Fraction | Decimal
        ):
            raise TypeError(
                f"value must be int, float, complex, Fraction or Decimal, "
                f"not {type(value).__name__}"
            )
        if isinstance(unit, str):
            resolved = default_registry.unit(unit)
        elif isinstance(unit, Unit):
            resolved = unit
        else:
            raise TypeError(f"unit must be a str or Unit, not {type(unit).__name__}")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_unit", resolved)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("Quantity is immutable")

    # -- properties ---------------------------------------------------------

    @property
    def value(self) -> Number:
        """Return the numeric magnitude."""
        return self._value

    @property
    def unit(self) -> Unit:
        """Return the unit."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimension."""
        return self._unit.dimension

    def _si_value(self) -> Number:
        """Return the magnitude expressed in the coherent SI unit."""
        return _add(_mul(self._value, self._unit.scale), self._unit.offset)

    # -- conversion ---------------------------------------------------------

    def to(self, unit: str | Unit, *, equivalence: str | None = None) -> Quantity:
        """Convert to another unit of the same dimension.

        Parameters
        ----------
        unit : str or Unit
            The target unit.
        equivalence : {None, "molar"}, optional
            When ``"molar"``, bridge per-amount and absolute quantities using
            the Avogadro constant (the dimensions must differ by a power of the
            amount-of-substance dimension).

        Returns
        -------
        Quantity
            The converted quantity.

        Raises
        ------
        DimensionalityError
            If the dimensions are incompatible.

        Examples
        --------
        >>> import duq
        >>> duq.Quantity(1.0, "eV").to("J").value
        1.602176634e-19
        >>> round(duq.Quantity(25.0, "degC").to("K").value, 2)
        298.15
        """
        target = self._unit.registry.unit(unit)
        if equivalence == "molar":
            return self._to_molar(target)
        if equivalence is not None:
            raise ValueError(f"unknown equivalence {equivalence!r}")
        apply(Rule.SAME_DIM, self.dimension, target.dimension)
        if self._unit.scale == target.scale and self._unit.offset == target.offset:
            return Quantity(self._value, target)
        si = self._si_value()
        new_value = _div(_sub(si, target.offset), target.scale)
        return Quantity(new_value, target)

    def _to_molar(self, target: Unit) -> Quantity:
        if _unit_is_affine(self._unit) or _unit_is_affine(target):
            raise AffineUnitError(
                "molar equivalence is undefined for affine units such as °C or °F; "
                "convert to an absolute unit (e.g. K) first"
            )
        src_n = self.dimension.exponents["N"]
        tgt_n = target.dimension.exponents["N"]
        k = tgt_n - src_n
        if self.dimension * Dimension({"N": k}) != target.dimension:
            raise DimensionalityError(
                "molar equivalence requires the dimensions to differ only by a "
                "power of the amount of substance"
            )
        si = _mul(self._value, self._unit.scale)
        si = _mul(si, _avogadro_power(-k))
        new_value = _div(si, target.scale)
        return Quantity(new_value, target)

    def to_si(self) -> Quantity:
        """Convert to the coherent SI unit for this dimension.

        Returns
        -------
        Quantity
            The quantity in coherent SI base units.

        Examples
        --------
        >>> import duq
        >>> str(duq.Quantity(1.0, "kJ/mol").to_si().unit)
        'kg·m²·s⁻²·mol⁻¹'
        """
        return self.to(self._unit.registry.coherent_unit(self.dimension))

    def value_in(self, unit: str | Unit) -> Number:
        """Return the magnitude expressed in ``unit``.

        Parameters
        ----------
        unit : str or Unit
            The target unit.

        Returns
        -------
        int, float, complex, fractions.Fraction or decimal.Decimal
            The converted magnitude.

        Examples
        --------
        >>> import duq
        >>> duq.Quantity(2.0, "m").value_in("cm")
        200.0
        """
        return self.to(unit).value

    def compact(self) -> Quantity:
        """Return an equal quantity with the SI prefix chosen so ``|value|`` is in [1, 1000).

        Only the leading prefixable linear factor is rescaled; compound or
        dimensionless units, and a zero value, are returned unchanged.

        Returns
        -------
        Quantity
            The rescaled quantity (a new object; nothing is mutated).

        Examples
        --------
        >>> import duq
        >>> c = duq.Quantity(1500.0, "m").compact()
        >>> c.value, str(c.unit)
        (1.5, 'km')
        """
        if self._unit.is_dimensionless or self._value == 0:
            return self
        factors = self._unit.factors
        index = next(
            (
                i
                for i, f in enumerate(factors)
                if f.atom.prefixable and f.exponent == 1 and f.atom.kind == "linear"
            ),
            None,
        )
        if index is None:
            return self
        magnitude = abs(_to_float(self._value))
        if magnitude == 0 or math.isnan(magnitude) or math.isinf(magnitude):
            return self
        active_prefix = factors[index].prefix
        current = active_prefix.exponent10 if active_prefix is not None else 0
        shift = 3 * math.floor(math.log10(magnitude) / 3)
        new_exp10 = current + shift
        prefix = self._unit.registry.si_prefix(new_exp10)
        if new_exp10 != 0 and prefix is None:
            return self
        rebuilt = list(factors)
        rebuilt[index] = Factor(prefix, factors[index].atom, Fraction(1))
        new_unit = Unit._create(self._unit.registry, tuple(rebuilt))
        new_value = _div(self._value, Fraction(10) ** shift)
        return Quantity(new_value, new_unit)

    # -- arithmetic ---------------------------------------------------------

    def _coerce_operand(self, other: object) -> Quantity | None:
        if isinstance(other, Quantity):
            return other
        if isinstance(other, bool):
            return None
        if isinstance(other, int | float | complex | Fraction | Decimal):
            if not self._unit.is_dimensionless:
                return None
            return Quantity(other, self._unit)
        return None

    def _add_sub(self, other: object, sign: int) -> Quantity | None:
        oq = self._coerce_operand(other)
        if oq is None:
            return None
        apply(Rule.SAME_DIM, self.dimension, oq.dimension)
        self_affine = self._unit.is_affine
        other_affine = oq._unit.is_affine
        if self_affine and other_affine:
            if sign > 0:
                raise AffineUnitError("cannot add two absolute (affine) quantities")
            coherent = self._unit.registry.coherent_unit(self.dimension)
            diff = _num_sub(self._si_value(), oq._si_value())  # offsets cancel
            return Quantity(diff, coherent)
        if other_affine and not self_affine:
            raise AffineUnitError(
                "cannot combine a relative quantity with an absolute (affine) quantity"
            )
        if self_affine:
            ratio = oq._unit.scale / self._unit.scale
            step = _mul(oq._value, ratio if isinstance(ratio, Fraction) else float(ratio))
            new_value = _num_add(self._value, step) if sign > 0 else _num_sub(self._value, step)
            return Quantity(new_value, self._unit)
        converted = oq.to(self._unit)
        new_value = (
            _num_add(self._value, converted._value)
            if sign > 0
            else _num_sub(self._value, converted._value)
        )
        return Quantity(new_value, self._unit)

    def __add__(self, other: object) -> Quantity:
        result = self._add_sub(other, 1)
        return NotImplemented if result is None else result

    def __radd__(self, other: object) -> Quantity:
        result = self._add_sub(other, 1)
        return NotImplemented if result is None else result

    def __sub__(self, other: object) -> Quantity:
        result = self._add_sub(other, -1)
        return NotImplemented if result is None else result

    def __rsub__(self, other: object) -> Quantity:
        result = self._add_sub(other, -1)
        if result is None:
            return NotImplemented
        return Quantity(_num_mul(result._value, -1), result._unit)

    def _scalar_mul(self, other: Number, invert: bool) -> Quantity:
        if self._unit.is_affine:
            raise AffineUnitError("cannot scale an affine quantity")
        new_value = _num_div(self._value, other) if invert else _num_mul(self._value, other)
        return Quantity(new_value, self._unit)

    def __mul__(self, other: object) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(_num_mul(self._value, other._value), self._unit * other._unit)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int | float | complex | Fraction | Decimal):
            return self._scalar_mul(other, invert=False)
        return NotImplemented

    def __rmul__(self, other: object) -> Quantity:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(_num_div(self._value, other._value), self._unit / other._unit)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int | float | complex | Fraction | Decimal):
            return self._scalar_mul(other, invert=True)
        return NotImplemented

    def __rtruediv__(self, other: object) -> Quantity:
        if isinstance(other, bool) or not isinstance(
            other, int | float | complex | Fraction | Decimal
        ):
            return NotImplemented
        if self._unit.is_affine:
            raise AffineUnitError("cannot invert an affine quantity")
        return Quantity(_num_div(other, self._value), self._unit**-1)

    def __pow__(self, power: object) -> Quantity:
        try:
            exp = _coerce_exponent(power)
        except TypeError:
            return NotImplemented
        base = exp.numerator if exp.denominator == 1 else exp
        return Quantity(_num_pow(self._value, base), self._unit**exp)

    def __neg__(self) -> Quantity:
        return self._scalar_mul(-1, invert=False)

    def __pos__(self) -> Quantity:
        return self

    def __abs__(self) -> Quantity:
        if self._unit.is_affine:
            raise AffineUnitError("cannot take the absolute value of an affine quantity")
        return Quantity(_num_abs(self._value), self._unit)

    # -- comparison ---------------------------------------------------------

    def _compare_value(self, other: Quantity) -> Number:
        return other.to(self._unit)._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        return self._value == self._compare_value(other)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return hash(("duq.Quantity", self.dimension, self._si_value()))

    def _order(self, other: object, op: Callable[[Any, Any], bool]) -> bool | None:
        if not isinstance(other, Quantity):
            return None
        apply(Rule.SAME_DIM, self.dimension, other.dimension)
        return op(self._value, self._compare_value(other))

    def __lt__(self, other: object) -> bool:
        result = self._order(other, lambda a, b: a < b)
        return NotImplemented if result is None else result

    def __le__(self, other: object) -> bool:
        result = self._order(other, lambda a, b: a <= b)
        return NotImplemented if result is None else result

    def __gt__(self, other: object) -> bool:
        result = self._order(other, lambda a, b: a > b)
        return NotImplemented if result is None else result

    def __ge__(self, other: object) -> bool:
        result = self._order(other, lambda a, b: a >= b)
        return NotImplemented if result is None else result

    def allclose(self, other: Quantity, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        """Return whether two quantities are close after conversion.

        Parameters
        ----------
        other : Quantity
            The quantity to compare against (converted to this unit first).
        rel_tol : float, optional
            Relative tolerance (default ``1e-9``).
        abs_tol : float, optional
            Absolute tolerance (default ``0.0``).

        Returns
        -------
        bool
            Whether the magnitudes are close in this unit.

        Examples
        --------
        >>> import duq
        >>> a = duq.Quantity(1.0, "J")
        >>> b = duq.Quantity(1.0000000001, "J")
        >>> a.allclose(b)
        True
        """
        apply(Rule.SAME_DIM, self.dimension, other.dimension)
        return math.isclose(
            _to_float(self._value),
            _to_float(self._compare_value(other)),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    # -- numpy stubs (fail loud until PR 3) ---------------------------------

    def __array_ufunc__(self, *args: object, **kwargs: object) -> object:
        """Reject NumPy ufuncs until the array layer ships."""
        raise UnsupportedOperationError(_NUMPY_MESSAGE)

    def __array_function__(self, *args: object, **kwargs: object) -> object:
        """Reject NumPy functions until the array layer ships."""
        raise UnsupportedOperationError(_NUMPY_MESSAGE)

    # -- rendering ----------------------------------------------------------

    def __repr__(self) -> str:
        return f"Quantity({self._value!r}, {self._unit!r})"

    def __str__(self) -> str:
        return f"{self._value} {self._unit}"


def uconvert(unit: str | Unit, q: Quantity) -> Quantity:
    """Convert a quantity to a unit (unit-first, JAX-idiomatic).

    Parameters
    ----------
    unit : str or Unit
        The target unit.
    q : Quantity
        The quantity to convert.

    Returns
    -------
    Quantity
        The converted quantity.

    Examples
    --------
    >>> import duq
    >>> duq.uconvert("cm", duq.Quantity(1.0, "m")).value
    100.0
    """
    return q.to(unit)


def ustrip(unit: str | Unit, q: Quantity) -> Number:
    """Return a quantity's magnitude in a unit (unit-first, JAX-idiomatic).

    Parameters
    ----------
    unit : str or Unit
        The target unit.
    q : Quantity
        The quantity to strip.

    Returns
    -------
    int, float, complex, fractions.Fraction or decimal.Decimal
        The bare magnitude in ``unit``.

    Examples
    --------
    >>> import duq
    >>> duq.ustrip("cm", duq.Quantity(1.0, "m"))
    100.0
    """
    return q.to(unit).value
