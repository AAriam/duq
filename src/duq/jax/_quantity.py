"""The JAX :class:`Quantity`: a quax ``ArrayValue`` with a static unit.

The magnitude is the traced pytree leaf; the unit is static metadata
(``equinox.field(static=True)``), so it participates in ``jit``'s cache key
and is checked at trace time.  ``materialise()`` refuses, which is what makes
uncovered primitives fail loud instead of silently dropping units.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import quax

from duq._dimension import Dimension
from duq._errors import (
    AffineUnitError,
    DimensionalityError,
    UnsupportedOperationError,
)
from duq._quantity import Quantity as CoreQuantity
from duq._quantity import _avogadro_power
from duq._registry import default_registry
from duq._rules import Rule, apply
from duq._unit import Unit

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ("Quantity",)


def _quaxed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a ``jax.numpy`` function once with :func:`quax.quaxify`."""
    return quax.quaxify(fn)


# Memoised quaxified jnp entry points backing the Python operators.
_add = _quaxed(jnp.add)
_subtract = _quaxed(jnp.subtract)
_multiply = _quaxed(jnp.multiply)
_divide = _quaxed(jnp.divide)
_floor_divide = _quaxed(jnp.floor_divide)
_remainder = _quaxed(jnp.remainder)
_power = _quaxed(jnp.power)
_matmul = _quaxed(jnp.matmul)
_negative = _quaxed(jnp.negative)
_absolute = _quaxed(jnp.absolute)
_equal = _quaxed(jnp.equal)
_not_equal = _quaxed(jnp.not_equal)
_less = _quaxed(jnp.less)
_less_equal = _quaxed(jnp.less_equal)
_greater = _quaxed(jnp.greater)
_greater_equal = _quaxed(jnp.greater_equal)


def _getitem(x: Any, key: Any) -> Any:
    return x[key]


_quaxed_getitem = _quaxed(_getitem)


def _is_operand(x: object) -> bool:
    """Return whether ``x`` may appear opposite a JAX quantity in an operator."""
    if isinstance(x, Quantity):
        return True
    return eqx.is_array_like(x) and not isinstance(x, CoreQuantity)


def _reject_core_quantity(x: object) -> None:
    if isinstance(x, CoreQuantity):
        raise TypeError(
            "cannot mix a core duq.Quantity with a duq.jax.Quantity; convert it "
            "first with duq.jax.Quantity.from_core(q)"
        )


class Quantity(quax.ArrayValue):
    """An immutable JAX array magnitude paired with a static physical unit.

    Parameters
    ----------
    value : array_like
        Anything :func:`jax.numpy.asarray` accepts (Python scalars, lists,
        NumPy arrays, JAX arrays and tracers).  :class:`~fractions.Fraction`
        and :class:`~decimal.Decimal` magnitudes are converted to ``float``.
    unit : str or Unit
        The unit; strings are parsed against the default registry.

    See Also
    --------
    duq.Quantity : The scalar/NumPy core quantity.

    Examples
    --------
    >>> import duq.jax
    >>> q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
    >>> str((q * q).unit)
    'm²'
    """

    magnitude: jax.Array
    unit: Unit = eqx.field(static=True)

    #: Opt out of NumPy's ufunc machinery so ``ndarray * q`` (etc.) defers to
    #: the reflected operators instead of calling ``np.asarray`` on the
    #: quantity (which fails loud by design).
    __array_ufunc__ = None

    def __init__(self, value: Any, unit: str | Unit) -> None:
        if isinstance(value, CoreQuantity):
            raise TypeError(
                "cannot wrap a core duq.Quantity in a duq.jax.Quantity; use "
                "duq.jax.Quantity.from_core(q) instead"
            )
        if isinstance(value, Fraction | Decimal):
            value = float(value)
        if isinstance(unit, str):
            resolved = default_registry.unit(unit)
        elif isinstance(unit, Unit):
            resolved = unit
        else:
            raise TypeError(f"unit must be a str or Unit, not {type(unit).__name__}")
        self.magnitude = jnp.asarray(value)
        self.unit = resolved

    # -- quax interface -------------------------------------------------------

    def aval(self) -> jax.core.ShapedArray:
        """Return the abstract value JAX sees: the magnitude's shape and dtype."""
        return jax.core.ShapedArray(jnp.shape(self.magnitude), jnp.result_type(self.magnitude))

    def materialise(self) -> Any:
        """Refuse to collapse to a plain array (units are never dropped silently)."""
        raise UnsupportedOperationError(
            "refusing to materialise a unit-carrying array, which would silently "
            "drop its unit; use duq.ustrip(unit, q) to get the magnitude in a "
            "chosen unit"
        )

    @staticmethod
    def default(
        primitive: Any, values: Sequence[Any], params: dict[str, Any]
    ) -> Any:  # numpydoc ignore=PR01,RT01
        """Fail loud on any JAX primitive without a registered duq unit rule."""
        raise UnsupportedOperationError(
            f"the JAX primitive {primitive.name!r} has no duq unit rule and is "
            "unsupported for unit-carrying arrays; call duq.ustrip(unit, q) to "
            "drop units explicitly first (see docs/dev/jax_coverage.md for the "
            "covered primitives)"
        )

    # -- properties -----------------------------------------------------------

    @property
    def value(self) -> jax.Array:
        """Return the numeric magnitude (alias of ``magnitude``)."""
        return self.magnitude

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimension."""
        return self.unit.dimension

    # -- conversion -----------------------------------------------------------

    def to(self, unit: str | Unit, *, equivalence: str | None = None) -> Quantity:
        """Convert to another unit of the same dimension.

        The conversion is ordinary traced arithmetic (scale and, for affine
        units such as °C, shift), so it works eagerly and under ``jit``.

        Parameters
        ----------
        unit : str or Unit
            The target unit.
        equivalence : {None, "molar"}, optional
            When ``"molar"``, bridge per-amount and absolute quantities using
            the Avogadro constant.

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
        >>> import duq.jax
        >>> duq.jax.Quantity(1.0, "km").to("m").value
        Array(1000., dtype=float32, weak_type=True)
        """
        target = self.unit.registry.unit(unit)
        if equivalence == "molar":
            return self._to_molar(target)
        if equivalence is not None:
            raise ValueError(f"unknown equivalence {equivalence!r}")
        apply(Rule.SAME_DIM, self.dimension, target.dimension)
        if self.unit.scale == target.scale and self.unit.offset == target.offset:
            return Quantity(self.magnitude, target)
        si = self.magnitude * float(self.unit.scale) + float(self.unit.offset)
        return Quantity((si - float(target.offset)) / float(target.scale), target)

    def _to_molar(self, target: Unit) -> Quantity:
        if self.unit.is_affine or target.is_affine:
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
        factor = float(self.unit.scale) * float(_avogadro_power(-k)) / float(target.scale)
        return Quantity(self.magnitude * factor, target)

    def value_in(self, unit: str | Unit) -> jax.Array:
        """Return the magnitude expressed in ``unit``.

        Parameters
        ----------
        unit : str or Unit
            The target unit.

        Returns
        -------
        jax.Array
            The converted magnitude.

        Examples
        --------
        >>> import duq.jax
        >>> duq.jax.Quantity(2.0, "m").value_in("cm")
        Array(200., dtype=float32, weak_type=True)
        """
        return self.to(unit).magnitude

    @classmethod
    def from_core(cls, q: CoreQuantity) -> Quantity:
        """Build a JAX quantity from a core :class:`duq.Quantity`.

        Parameters
        ----------
        q : duq.Quantity
            The core quantity; its magnitude is device-put via
            :func:`jax.numpy.asarray` and its unit is kept as-is.

        Returns
        -------
        Quantity
            The JAX-backed quantity.

        Examples
        --------
        >>> import duq, duq.jax
        >>> duq.jax.Quantity.from_core(duq.Quantity(1.5, "kJ/mol")).unit
        Unit('kJ.mol^-1')
        """
        return cls(q.value, q.unit)

    def to_core(self) -> CoreQuantity:
        """Return a core :class:`duq.Quantity` (device-to-host copy).

        Returns
        -------
        duq.Quantity
            A NumPy-backed core quantity with the same unit.

        Examples
        --------
        >>> import duq.jax
        >>> duq.jax.Quantity([1.0, 2.0], "m").to_core().is_array
        True
        """
        return CoreQuantity(np.asarray(self.magnitude), self.unit)

    # -- scalar coercion (dimensionless converts; dimensional fails loud) -----

    def _as_dimensionless(self, target: str) -> jax.Array:
        """Return the magnitude in the bare scale-1 dimensionless unit, or fail loud."""
        if self.unit.is_dimensionless:
            return self.magnitude * float(self.unit.scale)
        raise UnsupportedOperationError(
            f"refusing to coerce a quantity of dimension {self.dimension} to a "
            f"bare {target}, which would drop its unit; use duq.ustrip(unit, q) "
            "to get the magnitude in a chosen unit (only dimensionless "
            "quantities coerce directly)"
        )

    def __bool__(self) -> bool:
        """Return the truthiness of a dimensionless quantity's magnitude."""
        return bool(self._as_dimensionless("bool"))

    def __float__(self) -> float:
        """Return a dimensionless quantity's magnitude as a float (scale applied)."""
        return float(self._as_dimensionless("float"))

    def __int__(self) -> int:
        """Return a dimensionless quantity's magnitude as an int (scale applied)."""
        return int(self._as_dimensionless("int"))

    def __complex__(self) -> complex:
        """Return a dimensionless quantity's magnitude as a complex (scale applied)."""
        return complex(self._as_dimensionless("complex"))

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        """Refuse silent unit-stripping conversion to a bare array."""
        raise UnsupportedOperationError(
            "refusing to convert a duq.jax.Quantity to a bare array, which would "
            "silently drop its unit; use duq.ustrip(unit, q) to get the magnitude "
            "in a chosen unit (or q.value for the stored magnitude)"
        )

    # -- arithmetic operators (all route through the primitive rules) ---------

    def _binary(self, fn: Callable[..., Any], a: object, b: object) -> Any:
        _reject_core_quantity(a)
        _reject_core_quantity(b)
        if not (_is_operand(a) and _is_operand(b)):
            return NotImplemented
        return fn(a, b)

    def __add__(self, other: object) -> Quantity:
        return self._binary(_add, self, other)  # type: ignore[no-any-return]

    def __radd__(self, other: object) -> Quantity:
        return self._binary(_add, other, self)  # type: ignore[no-any-return]

    def __sub__(self, other: object) -> Quantity:
        return self._binary(_subtract, self, other)  # type: ignore[no-any-return]

    def __rsub__(self, other: object) -> Quantity:
        return self._binary(_subtract, other, self)  # type: ignore[no-any-return]

    def __mul__(self, other: object) -> Quantity:
        return self._binary(_multiply, self, other)  # type: ignore[no-any-return]

    def __rmul__(self, other: object) -> Quantity:
        return self._binary(_multiply, other, self)  # type: ignore[no-any-return]

    def __truediv__(self, other: object) -> Quantity:
        return self._binary(_divide, self, other)  # type: ignore[no-any-return]

    def __rtruediv__(self, other: object) -> Quantity:
        return self._binary(_divide, other, self)  # type: ignore[no-any-return]

    def __floordiv__(self, other: object) -> Quantity:
        return self._binary(_floor_divide, self, other)  # type: ignore[no-any-return]

    def __rfloordiv__(self, other: object) -> Quantity:
        return self._binary(_floor_divide, other, self)  # type: ignore[no-any-return]

    def __mod__(self, other: object) -> Quantity:
        return self._binary(_remainder, self, other)  # type: ignore[no-any-return]

    def __rmod__(self, other: object) -> Quantity:
        return self._binary(_remainder, other, self)  # type: ignore[no-any-return]

    def __pow__(self, power: object) -> Quantity:
        return self._binary(_power, self, power)  # type: ignore[no-any-return]

    def __matmul__(self, other: object) -> Quantity:
        return self._binary(_matmul, self, other)  # type: ignore[no-any-return]

    def __rmatmul__(self, other: object) -> Quantity:
        return self._binary(_matmul, other, self)  # type: ignore[no-any-return]

    def __neg__(self) -> Quantity:
        result: Quantity = _negative(self)
        return result

    def __pos__(self) -> Quantity:
        return self

    def __abs__(self) -> Quantity:
        result: Quantity = _absolute(self)
        return result

    # -- comparisons (plain bool arrays out) -----------------------------------

    def __eq__(self, other: object) -> jax.Array:  # type: ignore[override]
        return self._binary(_equal, self, other)  # type: ignore[no-any-return]

    def __ne__(self, other: object) -> jax.Array:  # type: ignore[override]
        return self._binary(_not_equal, self, other)  # type: ignore[no-any-return]

    def __lt__(self, other: object) -> jax.Array:
        return self._binary(_less, self, other)  # type: ignore[no-any-return]

    def __le__(self, other: object) -> jax.Array:
        return self._binary(_less_equal, self, other)  # type: ignore[no-any-return]

    def __gt__(self, other: object) -> jax.Array:
        return self._binary(_greater, self, other)  # type: ignore[no-any-return]

    def __ge__(self, other: object) -> jax.Array:
        return self._binary(_greater_equal, self, other)  # type: ignore[no-any-return]

    # Array quantities are unhashable, exactly like numpy.ndarray; the static
    # *unit* stays hashable, which is all jit's cache key needs.
    __hash__ = None  # type: ignore[assignment]

    # -- array ergonomics -------------------------------------------------------

    def __getitem__(self, key: object) -> Quantity:
        """Index the magnitude, preserving the unit."""
        result: Quantity = _quaxed_getitem(self, key)
        return result

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized object (0-d quantity)")
        return int(shape[0])

    @property
    def at(self) -> _IndexUpdateHelper:
        """Return the indexed-update helper (mirrors ``jax.Array.at``).

        Updates are unit-checked: a :class:`Quantity` update is converted to
        this quantity's unit before the scatter.

        Examples
        --------
        >>> import duq.jax
        >>> q = duq.jax.Quantity([1.0, 2.0], "m")
        >>> q.at[0].set(duq.jax.Quantity(300.0, "cm")).value
        Array([3., 2.], dtype=float32)
        """
        return _IndexUpdateHelper(self)

    # -- rendering --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Quantity({self.magnitude!r}, {self.unit!r})"

    def __str__(self) -> str:
        return f"{self.magnitude} {self.unit}"


def _make_at_op(name: str) -> Callable[..., Any]:
    """Build a quaxified ``x.at[key].<name>(value)`` entry point."""

    def apply_op(x: Any, key: Any, value: Any) -> Any:
        return getattr(x.at[key], name)(value)

    return _quaxed(apply_op)


_AT_OPS: dict[str, Callable[..., Any]] = {
    name: _make_at_op(name) for name in ("set", "add", "subtract", "min", "max")
}


class _IndexUpdateHelper:
    """Unit-aware mirror of ``jax.Array.at`` (see :attr:`Quantity.at`)."""

    __slots__ = ("_quantity",)

    def __init__(self, quantity: Quantity) -> None:
        self._quantity = quantity

    def __getitem__(self, key: object) -> _IndexUpdateRef:
        return _IndexUpdateRef(self._quantity, key)


class _IndexUpdateRef:
    """One pending indexed update; exposes ``set``/``add``/``subtract``/``min``/``max``."""

    __slots__ = ("_key", "_quantity")

    def __init__(self, quantity: Quantity, key: object) -> None:
        self._quantity = quantity
        self._key = key

    def get(self) -> Quantity:
        """Return the indexed elements (unit preserved)."""
        return self._quantity[self._key]

    def _apply(self, name: str, value: object) -> Quantity:
        result: Quantity = _AT_OPS[name](self._quantity, self._key, value)
        return result

    def set(self, value: object) -> Quantity:
        """Return a copy with the indexed elements set to ``value`` (converted)."""
        return self._apply("set", value)

    def add(self, value: object) -> Quantity:
        """Return a copy with ``value`` (converted) added to the indexed elements."""
        return self._apply("add", value)

    def subtract(self, value: object) -> Quantity:
        """Return a copy with ``value`` (converted) subtracted from the indexed elements."""
        return self._apply("subtract", value)

    def min(self, value: object) -> Quantity:
        """Return a copy with the elementwise minimum of ``value`` (converted)."""
        return self._apply("min", value)

    def max(self, value: object) -> Quantity:
        """Return a copy with the elementwise maximum of ``value`` (converted)."""
        return self._apply("max", value)
