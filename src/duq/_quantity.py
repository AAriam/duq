"""The scalar :class:`Quantity`: an immutable value paired with a unit.

``Quantity`` couples a Python scalar (``int``, ``float``, ``complex``,
:class:`~fractions.Fraction` or :class:`~decimal.Decimal`) with a
:class:`~duq._unit.Unit`.  Arithmetic checks dimensions, addition converts the
right operand to the left operand's unit, equality is *exact* after conversion,
and everything is immutable and hashable.

Importing this module never imports NumPy: the scalar arithmetic path is pure
Python, and the NumPy protocol hooks (:meth:`Quantity.__array_ufunc__`,
:meth:`Quantity.__array_function__`, :meth:`Quantity.__array__`) import the
:mod:`duq._numpy` back-end lazily, only when NumPy actually calls them.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final

from ._arraytypes import is_numpy_array, is_numpy_magnitude
from ._dimension import Dimension, _coerce_exponent
from ._errors import (
    AffineUnitError,
    DimensionalityError,
    UnsupportedOperationError,
)
from ._registry import default_registry
from ._rules import Rule, apply
from ._unit import Factor, Unit

__all__ = ("Quantity", "uconvert", "ustrip")

Number = int | float | complex | Fraction | Decimal
"""The scalar value types accepted by :class:`Quantity`."""

Scalar = Fraction | float

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    import numpy.typing as npt

    #: A NumPy array magnitude (any dtype).
    NDArray = npt.NDArray[Any]
    #: Everything a :class:`Quantity` can wrap: a Python scalar, a NumPy array,
    #: or a NumPy scalar (``numpy.generic``).
    Magnitude = Number | NDArray | np.generic

#: Avogadro constant as an exact integer Fraction (CODATA 2022, exact by SI).
_AVOGADRO: Final[Fraction] = Fraction(602214076 * 10**15)


def _to_decimal(x: Scalar | int) -> Decimal:
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(repr(x))


def _as_complex(factor: Scalar) -> complex | float:
    """Coerce a scale factor to something a complex value can combine with."""
    return complex(float(factor)) if isinstance(factor, Fraction) else factor


def _to_float(value: Any) -> float:
    """Return a real float magnitude (the modulus for complex values)."""
    if isinstance(value, complex):
        return abs(value)
    return float(value)


def _as_array_factor(factor: Scalar) -> float | complex:
    """Coerce a scale/offset to a float a NumPy array can combine with.

    A NumPy float array multiplied by a :class:`~fractions.Fraction` silently
    produces an ``object``-dtype array; converting the factor to ``float`` first
    keeps the result a native numeric array.
    """
    return float(factor) if isinstance(factor, Fraction) else factor


def _mul(value: Any, factor: Scalar) -> Any:
    if isinstance(value, Decimal):
        return value * _to_decimal(factor)
    if isinstance(value, complex):
        return value * _as_complex(factor)
    if is_numpy_magnitude(value):
        return value * _as_array_factor(factor)
    return value * factor


def _add(value: Any, addend: Scalar) -> Any:
    if addend == 0:
        return value
    if isinstance(value, Decimal):
        return value + _to_decimal(addend)
    if isinstance(value, complex):
        return value + _as_complex(addend)
    if is_numpy_magnitude(value):
        return value + _as_array_factor(addend)
    return value + addend


def _sub(value: Any, subtrahend: Scalar) -> Any:
    if subtrahend == 0:
        return value
    if isinstance(value, Decimal):
        return value - _to_decimal(subtrahend)
    if isinstance(value, complex):
        return value - _as_complex(subtrahend)
    if is_numpy_magnitude(value):
        return value - _as_array_factor(subtrahend)
    return value - subtrahend


def _div(value: Any, divisor: Scalar) -> Any:
    if divisor == 1:
        return value
    if isinstance(value, Decimal):
        return value / _to_decimal(divisor)
    if isinstance(value, complex):
        return value / _as_complex(divisor)
    if is_numpy_magnitude(value):
        return value / _as_array_factor(divisor)
    return value / divisor


def _avogadro_power(power: Fraction) -> Scalar:
    if power.denominator == 1:
        exact: Scalar = _AVOGADRO**power.numerator
        return exact
    inexact: float = float(_AVOGADRO) ** float(power)
    return inexact


# Value-by-value arithmetic that is polymorphic over the whole numeric tower
# *and* NumPy magnitudes; typed ``Any`` because a single static type cannot span
# int/float/complex/Fraction/Decimal and ndarray simultaneously.  Mixing
# incompatible concrete types (e.g. a Decimal with a float) raises TypeError at
# runtime, which is the intended behaviour.
def _num_add(a: Any, b: Any) -> Any:
    return a + b


def _num_sub(a: Any, b: Any) -> Any:
    return a - b


def _num_mul(a: Any, b: Any) -> Any:
    return a * b


def _num_div(a: Any, b: Any) -> Any:
    return a / b


def _num_pow(a: Any, b: int | Fraction) -> Any:
    return a**b


def _num_abs(a: Any) -> Any:
    return abs(a)


def _unit_is_affine(unit: Unit) -> bool:
    """Return whether a unit is affine (non-zero offset or an affine atom)."""
    return unit.offset != 0 or any(f.atom.kind == "affine" for f in unit.factors)


class Quantity:
    """An immutable scalar value with a physical unit.

    Parameters
    ----------
    value : int, float, complex, fractions.Fraction, decimal.Decimal or numpy.ndarray
        The numeric magnitude.  A NumPy array (or NumPy scalar) is stored as-is
        -- no copy is made -- turning the quantity into a unit-carrying array.
    unit : str or Unit
        The unit; strings are parsed against the default registry.

    See Also
    --------
    Quantity.from_array : Build an array quantity from any array-like (e.g. a list).

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

    A NumPy array magnitude turns the quantity into a unit-carrying array; units
    propagate through ufuncs, reductions and ``__array_function__``:

    >>> import numpy as np
    >>> q = duq.Quantity(np.array([1.0, 2.0, 3.0]), "m")
    >>> np.sqrt(q * q).unit == duq.unit("m")
    True
    >>> float(q.sum().value)
    6.0
    """

    __slots__ = ("_unit", "_value")
    _value: Magnitude
    _unit: Unit

    def __init__(self, value: Magnitude, unit: str | Unit) -> None:
        if isinstance(value, bool) or not (
            isinstance(value, int | float | complex | Fraction | Decimal)
            or is_numpy_magnitude(value)
        ):
            raise TypeError(
                f"value must be int, float, complex, Fraction, Decimal or a NumPy "
                f"array, not {type(value).__name__}"
            )
        if isinstance(unit, str):
            resolved = default_registry.unit(unit)
        elif isinstance(unit, Unit):
            resolved = unit
        else:
            raise TypeError(f"unit must be a str or Unit, not {type(unit).__name__}")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_unit", resolved)

    @classmethod
    def from_array(cls, value: object, unit: str | Unit) -> Quantity:
        """Build an array quantity from any array-like magnitude.

        Unlike the constructor (which requires a NumPy array or a Python scalar),
        this coerces ``value`` with :func:`numpy.asarray`, so Python lists and
        other sequences become array quantities.

        Parameters
        ----------
        value : array_like
            Anything :func:`numpy.asarray` accepts (list, tuple, ndarray, ...).
        unit : str or Unit
            The unit; strings are parsed against the default registry.

        Returns
        -------
        Quantity
            An array-backed quantity.

        Examples
        --------
        >>> import duq
        >>> q = duq.Quantity.from_array([1.0, 2.0, 3.0], "m")
        >>> q.shape
        (3,)
        """
        import numpy as np

        return cls(np.asarray(value), unit)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("Quantity is immutable")

    # -- properties ---------------------------------------------------------

    @property
    def value(self) -> Magnitude:
        """Return the numeric magnitude."""
        return self._value

    @property
    def is_array(self) -> bool:
        """Return whether the magnitude is a NumPy array."""
        return is_numpy_array(self._value)

    @property
    def unit(self) -> Unit:
        """Return the unit."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimension."""
        return self._unit.dimension

    def _si_value(self) -> Magnitude:
        """Return the magnitude expressed in the coherent SI unit."""
        result: Magnitude = _add(_mul(self._value, self._unit.scale), self._unit.offset)
        return result

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

    def value_in(self, unit: str | Unit) -> Magnitude:
        """Return the magnitude expressed in ``unit``.

        Parameters
        ----------
        unit : str or Unit
            The target unit.

        Returns
        -------
        int, float, complex, fractions.Fraction, decimal.Decimal or numpy.ndarray
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
        if self.is_array:
            raise UnsupportedOperationError(
                "compact() is defined only for scalar quantities, not array magnitudes"
            )
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
                # A bare number has no reflected handler that could succeed,
                # so fail with a helpful message instead of NotImplemented.
                raise DimensionalityError(
                    "cannot add or subtract a bare number and a quantity of "
                    f"dimension {self.dimension}; only dimensionless "
                    "quantities accept bare numbers"
                )
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

    def _compare_value(self, other: Quantity) -> Magnitude:
        return other.to(self._unit)._value

    def _array_compare(self, other: object) -> bool:
        """Return whether a comparison with ``other`` involves a NumPy array."""
        if is_numpy_magnitude(self._value):
            return True
        if isinstance(other, Quantity):
            return is_numpy_magnitude(other._value)
        return is_numpy_magnitude(other)

    def __eq__(self, other: object) -> bool | NDArray:  # type: ignore[override]
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("equal", self, other)
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        return bool(self._value == self._compare_value(other))

    def __ne__(self, other: object) -> bool | NDArray:  # type: ignore[override]
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("not_equal", self, other)
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

    def __lt__(self, other: object) -> bool | NDArray:
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("less", self, other)
        result = self._order(other, lambda a, b: a < b)
        return NotImplemented if result is None else result

    def __le__(self, other: object) -> bool | NDArray:
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("less_equal", self, other)
        result = self._order(other, lambda a, b: a <= b)
        return NotImplemented if result is None else result

    def __gt__(self, other: object) -> bool | NDArray:
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("greater", self, other)
        result = self._order(other, lambda a, b: a > b)
        return NotImplemented if result is None else result

    def __ge__(self, other: object) -> bool | NDArray:
        if self._array_compare(other):
            from ._numpy import richcompare

            return richcompare("greater_equal", self, other)
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
        if self.is_array or other.is_array:
            raise UnsupportedOperationError(
                "Quantity.allclose is for scalars; use numpy.allclose on array quantities"
            )
        apply(Rule.SAME_DIM, self.dimension, other.dimension)
        return math.isclose(
            _to_float(self._value),
            _to_float(self._compare_value(other)),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    # -- numpy protocol (NEP 13 / NEP 18) -----------------------------------

    def __array_ufunc__(
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        """Dispatch a NumPy ufunc through the curated duq unit-rule table.

        Any ufunc/method without a registered rule -- and any ``out=`` argument
        -- raises :class:`~duq.UnsupportedOperationError`; units are never
        silently dropped.
        """
        from ._numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Dispatch a NumPy function through the curated duq unit-rule table."""
        from ._numpy import dispatch_function

        return dispatch_function(func, types, args, kwargs)

    def __array__(self, dtype: object = None, copy: object = None) -> object:
        """Refuse silent unit-stripping conversion to a bare NumPy array.

        Closing this classic hole is a design goal: ``numpy.array(q)`` and
        ``numpy.asarray(q)`` must never quietly discard the unit.  Use
        :func:`duq.ustrip` (or :attr:`value`) to obtain the raw magnitude.
        """
        raise UnsupportedOperationError(
            "refusing to convert a Quantity to a bare NumPy array, which would "
            "silently drop its unit; use duq.ustrip(unit, q) to get the magnitude "
            "in a chosen unit (or q.value for the stored magnitude)"
        )

    def __matmul__(self, other: object) -> object:
        from ._numpy import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: object) -> object:
        from ._numpy import matmul

        return matmul(other, self)

    # -- array introspection ------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the magnitude (``()`` for a scalar magnitude).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> duq.Quantity(np.zeros((2, 3)), "m").shape
        (2, 3)
        """
        from ._numpy import magnitude_shape

        return magnitude_shape(self._value)

    @property
    def ndim(self) -> int:
        """Return the number of magnitude dimensions (``0`` for a scalar)."""
        from ._numpy import magnitude_ndim

        return magnitude_ndim(self._value)

    @property
    def size(self) -> int:
        """Return the number of magnitude elements (``1`` for a scalar)."""
        from ._numpy import magnitude_size

        return magnitude_size(self._value)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the NumPy dtype of the magnitude (array magnitudes only)."""
        from ._numpy import magnitude_dtype

        return magnitude_dtype(self._value)

    @property
    def T(self) -> Quantity:  # noqa: N802 - mirrors ``ndarray.T``
        """Return the transposed quantity (unit preserved).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> duq.Quantity(np.zeros((2, 3)), "m").T.shape
        (3, 2)
        """
        from ._numpy import q_transpose

        return q_transpose(self)

    def __len__(self) -> int:
        from ._numpy import magnitude_len

        return magnitude_len(self._value)

    def __iter__(self) -> Iterator[Quantity]:
        from ._numpy import iter_quantity

        return iter_quantity(self)

    def __getitem__(self, key: object) -> Quantity:
        from ._numpy import get_item

        return get_item(self, key)

    def item(self, *args: object) -> Quantity:
        """Return a single element as a scalar :class:`Quantity` (unit preserved).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> duq.Quantity(np.array([5.0]), "m").item().value
        5.0
        """
        from ._numpy import scalar_item

        return scalar_item(self, args)

    # -- array reductions / reshaping (thin delegates) ----------------------

    def sum(self, **kwargs: object) -> Quantity:
        """Sum the magnitude, preserving the unit (see :func:`numpy.sum`).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> float(duq.Quantity(np.array([1.0, 2.0, 3.0]), "m").sum().value)
        6.0
        """
        from ._numpy import q_reduce

        return q_reduce(self, "sum", kwargs)

    def mean(self, **kwargs: object) -> Quantity:
        """Average the magnitude, preserving the unit (see :func:`numpy.mean`)."""
        from ._numpy import q_reduce

        return q_reduce(self, "mean", kwargs)

    def std(self, **kwargs: object) -> Quantity:
        """Return the standard deviation, preserving the unit (see :func:`numpy.std`)."""
        from ._numpy import q_reduce

        return q_reduce(self, "std", kwargs)

    def var(self, **kwargs: object) -> Quantity:
        """Return the variance, squaring the unit (see :func:`numpy.var`).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> duq.Quantity(np.array([1.0, 2.0, 3.0]), "m").var().unit == duq.unit("m^2")
        True
        """
        from ._numpy import q_reduce

        return q_reduce(self, "var", kwargs, unit_power=2)

    def min(self, **kwargs: object) -> Quantity:
        """Return the minimum, preserving the unit (see :func:`numpy.min`)."""
        from ._numpy import q_reduce

        return q_reduce(self, "min", kwargs)

    def max(self, **kwargs: object) -> Quantity:
        """Return the maximum, preserving the unit (see :func:`numpy.max`)."""
        from ._numpy import q_reduce

        return q_reduce(self, "max", kwargs)

    def reshape(self, *shape: object, **kwargs: object) -> Quantity:
        """Reshape the magnitude, preserving the unit (see :func:`numpy.reshape`).

        Examples
        --------
        >>> import numpy as np
        >>> import duq
        >>> duq.Quantity(np.arange(6.0), "m").reshape(2, 3).shape
        (2, 3)
        """
        from ._numpy import q_reshape

        return q_reshape(self, shape, kwargs)

    def ravel(self, **kwargs: object) -> Quantity:
        """Flatten the magnitude, preserving the unit (see :func:`numpy.ravel`)."""
        from ._numpy import q_ravel

        return q_ravel(self, kwargs)

    def astype(self, dtype: object, **kwargs: object) -> Quantity:
        """Cast the magnitude to ``dtype``, preserving the unit."""
        from ._numpy import q_astype

        return q_astype(self, dtype, kwargs)

    # -- scalar coercion (dimensionless converts; dimensional fails loud) ---

    def _as_dimensionless(self, target: str) -> Any:
        """Return the magnitude in the bare scale-1 dimensionless unit, or fail loud.

        A dimensionless quantity has one unambiguous numeric value once its
        scale is applied (``50 %`` is ``0.5``; radian-labelled values pass
        through unchanged), so coercing it never loses information.  Anything
        dimensional refuses, pointing at :func:`duq.ustrip`.
        """
        if self._unit.is_dimensionless:
            return self._si_value()
        raise UnsupportedOperationError(
            f"refusing to coerce a quantity of dimension {self.dimension} to a "
            f"bare {target}, which would drop its unit; use duq.ustrip(unit, q) "
            "to get the magnitude in a chosen unit (only dimensionless "
            "quantities coerce directly)"
        )

    def __bool__(self) -> bool:
        """Return the truthiness of a dimensionless quantity's magnitude.

        Dimensionless (including angle-labelled and scaled units like ``%``)
        follows the magnitude's own truthiness -- for array magnitudes that is
        NumPy's rule (one element coerces, more raise ``ValueError``).
        Dimensional quantities raise :class:`~duq.UnsupportedOperationError`.

        Examples
        --------
        >>> import duq
        >>> bool(duq.Quantity(0.0, "1")), bool(duq.Quantity(50.0, "%"))
        (False, True)
        """
        return bool(self._as_dimensionless("bool"))

    def __float__(self) -> float:
        """Return a dimensionless quantity's magnitude as a float (scale applied).

        Dimensional quantities raise :class:`~duq.UnsupportedOperationError`
        with guidance to use :func:`duq.ustrip` (pint-consistent behaviour).

        Examples
        --------
        >>> import duq
        >>> float(duq.Quantity(50.0, "%"))
        0.5
        >>> float(duq.Quantity(1.5, "rad"))
        1.5
        """
        return float(self._as_dimensionless("float"))

    def __int__(self) -> int:
        """Return a dimensionless quantity's magnitude as an int (scale applied).

        Examples
        --------
        >>> import duq
        >>> int(duq.Quantity(200.0, "%"))
        2
        """
        return int(self._as_dimensionless("int"))

    def __complex__(self) -> complex:
        """Return a dimensionless quantity's magnitude as a complex (scale applied).

        Examples
        --------
        >>> import duq
        >>> complex(duq.Quantity(50.0, "%"))
        (0.5+0j)
        """
        return complex(self._as_dimensionless("complex"))

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


def ustrip(unit: str | Unit, q: Quantity) -> Magnitude:
    """Return a quantity's magnitude in a unit (unit-first, JAX-idiomatic).

    Parameters
    ----------
    unit : str or Unit
        The target unit.
    q : Quantity
        The quantity to strip.

    Returns
    -------
    int, float, complex, fractions.Fraction, decimal.Decimal or numpy.ndarray
        The bare magnitude in ``unit``.

    Examples
    --------
    >>> import duq
    >>> duq.ustrip("cm", duq.Quantity(1.0, "m"))
    100.0
    """
    return q.to(unit).value
