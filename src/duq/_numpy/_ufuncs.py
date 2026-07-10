"""The curated ``__array_ufunc__`` table: NumPy ufunc -> duq unit rule.

Every supported ufunc maps to a small handler that strips its operands to bare
magnitudes, applies the unit rule from :mod:`duq._rules` (or direct unit
algebra), calls the underlying ufunc on the magnitudes, and re-boxes the result.
Anything absent from :data:`UFUNC_TABLE` fails loud in :mod:`duq._numpy._dispatch`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from duq._errors import AffineUnitError, UnsupportedOperationError

from ._common import (
    Quantity,
    Unit,
    box,
    compare_core,
    dimensionless_unit,
    magnitude,
    registry_of,
    reject_affine,
    require_dimensionless_raw,
    to_pure,
    to_unit,
    unit_of,
    unit_power,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    Handler = Callable[[Any, tuple[object, ...], dict[str, Any]], object]

__all__ = ("REDUCE_PRESERVE", "UFUNC_TABLE")


# -- helpers ----------------------------------------------------------------


def _binary(inputs: tuple[object, ...]) -> tuple[object, object]:
    a, b = inputs
    return a, b


def _resolved_unit(x: object, dimensionless: Unit) -> Unit:
    unit = unit_of(x)
    return dimensionless if unit is None else unit


# -- multiplicative ---------------------------------------------------------


def _h_multiply(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    a, b = _binary(inputs)
    reject_affine(unit_of(a), unit_of(b))
    dl = dimensionless_unit(registry_of(a, b))
    ua = _resolved_unit(a, dl)
    ub = _resolved_unit(b, dl)
    result = ufunc(magnitude(a), magnitude(b), **kwargs)
    return box(result, ua * ub)


def _h_divide(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    a, b = _binary(inputs)
    reject_affine(unit_of(a), unit_of(b))
    dl = dimensionless_unit(registry_of(a, b))
    ua = _resolved_unit(a, dl)
    ub = _resolved_unit(b, dl)
    result = ufunc(magnitude(a), magnitude(b), **kwargs)
    return box(result, ua / ub)


# -- additive / same-dimension ----------------------------------------------


def _box_pair(inputs: tuple[object, ...]) -> tuple[Any, Any]:
    a, b = _binary(inputs)
    dl = dimensionless_unit(registry_of(a, b))
    qa = a if isinstance(a, Quantity) else Quantity(a, dl)  # type: ignore[arg-type]
    qb = b if isinstance(b, Quantity) else Quantity(b, dl)  # type: ignore[arg-type]
    return qa, qb


def _h_add(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    qa, qb = _box_pair(inputs)
    return qa + qb


def _h_subtract(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    qa, qb = _box_pair(inputs)
    return qa - qb


def _h_same_dim(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    a, b = _binary(inputs)
    ua, ub = unit_of(a), unit_of(b)
    reference = ua if ua is not None else ub
    assert reference is not None  # a ufunc is dispatched only with a Quantity operand
    ma = to_unit(a, reference)
    mb = to_unit(b, reference)
    result = ufunc(ma, mb, **kwargs)
    return box(result, reference)


# -- powers -----------------------------------------------------------------


def _h_power(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    base, exponent = _binary(inputs)
    exp_pure = to_pure(exponent)
    exp_arr = np.asarray(exp_pure)
    if exp_arr.ndim == 0:
        exp_value = exp_arr.item()
    else:
        first = exp_arr.reshape(-1)[0].item()
        if not bool(np.all(exp_arr == first)):
            raise UnsupportedOperationError(
                "numpy power with an array of differing exponents is unsupported: "
                "each element would carry a different unit"
            )
        exp_value = first
    result = ufunc(magnitude(base), exp_pure, **kwargs)
    ubase = unit_of(base)
    if ubase is None:
        return result
    return box(result, unit_power(ubase, exp_value))


def _make_unit_pow(exponent: Fraction) -> Handler:
    def handler(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        (x,) = inputs
        result = ufunc(magnitude(x), **kwargs)
        unit = unit_of(x)
        if unit is None:
            return result
        return box(result, unit**exponent)

    return handler


# -- unary preserving / stripping -------------------------------------------

_AFFINE_FORBIDDEN = frozenset({np.negative, np.absolute, np.fabs})


def _h_preserve_unary(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    (x,) = inputs
    unit = unit_of(x)
    if unit is not None and unit.is_affine and ufunc in _AFFINE_FORBIDDEN:
        raise AffineUnitError(f"cannot apply {ufunc.__name__} to an affine quantity such as °C")
    result = ufunc(magnitude(x), **kwargs)
    return result if unit is None else box(result, unit)


def _h_strip_unary(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    """Apply a ufunc to the raw magnitude and return a plain array (sign, isnan...)."""
    (x,) = inputs
    return ufunc(magnitude(x), **kwargs)


# -- dimensionless / angle ---------------------------------------------------


def _h_dimensionless_unary(
    ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]
) -> object:
    (x,) = inputs
    result = ufunc(to_pure(x), **kwargs)
    return box(result, dimensionless_unit(registry_of(x)))


def _h_dimensionless_binary(
    ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]
) -> object:
    a, b = _binary(inputs)
    result = ufunc(to_pure(a), to_pure(b), **kwargs)
    return box(result, dimensionless_unit(registry_of(a, b)))


def _h_inverse_trig(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    (x,) = inputs
    result = ufunc(to_pure(x), **kwargs)
    return box(result, registry_of(x).unit("rad"))


def _h_arctan2(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    a, b = _binary(inputs)
    ua, ub = unit_of(a), unit_of(b)
    reference = ua if ua is not None else ub
    assert reference is not None
    result = ufunc(to_unit(a, reference), to_unit(b, reference), **kwargs)
    return box(result, registry_of(a, b).unit("rad"))


def _make_angle_convert(out_unit: str) -> Handler:
    def handler(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        (x,) = inputs
        result = ufunc(require_dimensionless_raw(x), **kwargs)
        return box(result, registry_of(x).unit(out_unit))

    return handler


def _h_copysign(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    a, b = _binary(inputs)
    ua = unit_of(a)
    assert ua is not None
    # Only the sign of ``b`` is used; a same-dimension check keeps it meaningful.
    mb = to_unit(b, ua) if unit_of(b) is not None else magnitude(b)
    result = ufunc(magnitude(a), mb, **kwargs)
    return box(result, ua)


# -- comparisons -------------------------------------------------------------


def _make_compare(op_name: str) -> Handler:
    def handler(ufunc: Any, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        a, b = _binary(inputs)
        return compare_core(op_name, a, b)

    return handler


# -- table -------------------------------------------------------------------

#: Ufuncs whose ``reduce``/``accumulate`` preserve the unit.
REDUCE_PRESERVE: frozenset[Any] = frozenset({np.add, np.maximum, np.minimum})


def _build_table() -> dict[Any, Handler]:
    table: dict[Any, Handler] = {}

    def each(handler: Handler, *ufuncs: Any) -> None:
        for ufunc in ufuncs:
            table[ufunc] = handler

    each(_h_multiply, np.multiply, np.matmul)
    each(_h_divide, np.divide, np.true_divide, np.floor_divide)
    each(_h_add, np.add)
    each(_h_subtract, np.subtract)
    each(
        _h_same_dim,
        np.hypot,
        np.maximum,
        np.minimum,
        np.fmax,
        np.fmin,
        np.remainder,
        np.mod,
        np.fmod,
        np.nextafter,
    )
    each(_h_copysign, np.copysign)
    each(_h_power, np.power, np.float_power)

    each(_make_unit_pow(Fraction(1, 2)), np.sqrt)
    each(_make_unit_pow(Fraction(1, 3)), np.cbrt)
    each(_make_unit_pow(Fraction(2)), np.square)
    each(_make_unit_pow(Fraction(-1)), np.reciprocal)

    each(
        _h_preserve_unary,
        np.negative,
        np.positive,
        np.absolute,
        np.fabs,
        np.conjugate,
        np.floor,
        np.ceil,
        np.trunc,
        np.rint,
    )
    each(_h_strip_unary, np.sign, np.isfinite, np.isnan, np.isinf, np.signbit)

    each(
        _h_dimensionless_unary,
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.sin,
        np.cos,
        np.tan,
    )
    each(_h_dimensionless_binary, np.logaddexp, np.logaddexp2)
    each(_h_inverse_trig, np.arcsin, np.arccos, np.arctan)
    each(_h_arctan2, np.arctan2)

    each(_make_angle_convert("rad"), np.deg2rad, np.radians)
    each(_make_angle_convert("deg"), np.rad2deg, np.degrees)

    for name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"):
        each(_make_compare(name), getattr(np, name))

    return table


#: The curated ufunc dispatch table (keyed by ufunc object).
UFUNC_TABLE: dict[Any, Handler] = _build_table()
