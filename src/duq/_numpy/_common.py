"""Shared primitives for the NumPy front-end.

Importing this module imports NumPy, so it is only ever reached lazily from
:class:`duq.Quantity`'s array hooks.  Everything here is small and pure: unit
resolution, magnitude conversion, dimensionless/angle stripping and result
boxing, on top of the core :mod:`duq._rules` engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from duq._dimension import Dimension, _coerce_exponent
from duq._errors import AffineUnitError, DimensionalityError, UnsupportedOperationError
from duq._quantity import Quantity
from duq._rules import Rule, apply
from duq._unit import Unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duq._registry import UnitRegistry

__all__ = (
    "Quantity",
    "Unit",
    "all_false",
    "all_true",
    "box",
    "compare_core",
    "convert_magnitude",
    "dimensionless_unit",
    "find_quantity",
    "magnitude",
    "registry_of",
    "reject_affine",
    "require_dimensionless_raw",
    "to_pure",
    "to_unit",
    "unit_of",
)

_DIMENSIONLESS = Dimension({})

_COMPARE_UFUNCS: dict[str, Any] = {
    "equal": np.equal,
    "not_equal": np.not_equal,
    "less": np.less,
    "less_equal": np.less_equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
}


def magnitude(x: object) -> Any:
    """Return the raw magnitude of a Quantity, or ``x`` unchanged."""
    return x.value if isinstance(x, Quantity) else x


def unit_of(x: object) -> Unit | None:
    """Return the unit of a Quantity, or ``None`` for a plain operand."""
    return x.unit if isinstance(x, Quantity) else None


def dimension_of(x: object) -> Dimension:
    """Return the dimension of a Quantity, or dimensionless for a plain operand."""
    return x.dimension if isinstance(x, Quantity) else _DIMENSIONLESS


def find_quantity(obj: object) -> Quantity | None:
    """Return the first Quantity in ``obj`` (searching one level into sequences)."""
    if isinstance(obj, Quantity):
        return obj
    if isinstance(obj, list | tuple):
        for item in obj:
            if isinstance(item, Quantity):
                return item
    return None


def registry_of(*objs: object) -> UnitRegistry:
    """Return the registry of the first Quantity found among ``objs``."""
    for obj in objs:
        found = find_quantity(obj)
        if found is not None:
            return found.unit.registry
    raise UnsupportedOperationError("expected at least one duq.Quantity operand")


def dimensionless_unit(registry: UnitRegistry) -> Unit:
    """Return the registry's bare dimensionless unit (``"1"``)."""
    return registry.unit("1")


def box(mag: object, unit: Unit) -> Quantity:
    """Wrap a magnitude in a :class:`Quantity` with ``unit``."""
    return Quantity(mag, unit)  # type: ignore[arg-type]


def reject_affine(*units: Unit | None, action: str = "scale") -> None:
    """Raise :class:`AffineUnitError` if any of ``units`` is affine (e.g. ``°C``)."""
    for unit in units:
        if unit is not None and unit.is_affine:
            raise AffineUnitError(f"cannot {action} an affine quantity such as °C or °F")


def convert_magnitude(mag: Any, src: Unit, tgt: Unit) -> Any:
    """Convert a bare magnitude from ``src`` to ``tgt`` using float scale/offset."""
    if src.scale == tgt.scale and src.offset == tgt.offset:
        return mag
    s, s_off = float(src.scale), float(src.offset)
    t, t_off = float(tgt.scale), float(tgt.offset)
    return (mag * s + s_off - t_off) / t


def to_unit(x: object, unit: Unit) -> Any:
    """Convert operand ``x`` to ``unit`` and return its bare magnitude.

    A plain (unitless) operand is accepted only when ``unit`` is dimensionless;
    a Quantity of an incompatible dimension raises :class:`DimensionalityError`.
    """
    if isinstance(x, Quantity):
        apply(Rule.SAME_DIM, x.dimension, unit.dimension)
        return convert_magnitude(x.value, x.unit, unit)
    if not unit.is_dimensionless:
        raise DimensionalityError(
            f"cannot combine a bare array/number with a quantity of dimension {unit.dimension}"
        )
    return x


def to_pure(x: object) -> Any:
    """Return the plain dimensionless value of ``x`` (``%`` and ``deg`` rescaled).

    A dimensionless (or angle) Quantity is stripped after multiplying by its
    scale, so ``50 %`` becomes ``0.5`` and ``90 deg`` becomes ``pi/2``.
    """
    if isinstance(x, Quantity):
        u = x.unit
        if not u.is_dimensionless:
            raise DimensionalityError(
                f"expected a dimensionless or angle operand, got dimension {u.dimension}"
            )
        mag: Any = x.value
        return mag * float(u.scale)
    return x


def require_dimensionless_raw(x: object) -> Any:
    """Return the raw magnitude of ``x``, requiring a dimensionless operand.

    Unlike :func:`to_pure` the scale is *not* applied; this suits the angle
    conversion ufuncs (``deg2rad``/``rad2deg``) which do the scaling themselves.
    """
    if isinstance(x, Quantity):
        if not x.unit.is_dimensionless:
            raise DimensionalityError(
                f"expected a dimensionless operand, got dimension {x.unit.dimension}"
            )
        return x.value
    return x


def _broadcast_shape(a: object, b: object) -> tuple[int, ...]:
    return np.broadcast_shapes(np.shape(magnitude(a)), np.shape(magnitude(b)))


def all_false(a: object, b: object) -> Any:
    """Return a broadcast ``bool`` array of ``False`` (incompatible ``equal``)."""
    return np.zeros(_broadcast_shape(a, b), dtype=bool)


def all_true(a: object, b: object) -> Any:
    """Return a broadcast ``bool`` array of ``True`` (incompatible ``not_equal``)."""
    return np.ones(_broadcast_shape(a, b), dtype=bool)


def compare_core(op_name: str, a: object, b: object) -> Any:
    """Compare two operands unit-aware, returning a plain ``bool`` ndarray.

    ``equal``/``not_equal`` on incompatible dimensions short-circuit to an
    all-``False``/all-``True`` array (parity with the scalar ``==`` policy);
    ordering on incompatible dimensions raises :class:`DimensionalityError`.
    """
    da, db = dimension_of(a), dimension_of(b)
    if da != db:
        if op_name == "equal":
            return all_false(a, b)
        if op_name == "not_equal":
            return all_true(a, b)
        raise DimensionalityError(f"cannot order-compare quantities of dimension {da} and {db}")
    ua, ub = unit_of(a), unit_of(b)
    reference = ua if ua is not None else ub
    if reference is None:  # pragma: no cover - dispatch guarantees a Quantity operand
        raise UnsupportedOperationError("expected at least one duq.Quantity operand")
    ma = to_unit(a, reference)
    mb = to_unit(b, reference)
    return _COMPARE_UFUNCS[op_name](ma, mb)


def convert_sequence(items: Sequence[object], unit: Unit) -> list[Any]:
    """Convert every operand in ``items`` to ``unit`` (for concatenate/stack)."""
    return [to_unit(item, unit) for item in items]


def unit_power(unit: Unit, exponent: object) -> Unit:
    """Raise ``unit`` to an exact ``exponent`` (coercing floats such as ``0.5``)."""
    return unit ** _coerce_exponent(exponent)
