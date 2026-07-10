"""The quax primitive rules: ``jax.lax`` primitive -> duq unit rule.

Each rule strips its operands to bare magnitudes, applies the unit rule from
:mod:`duq._rules` (or direct unit algebra), binds the underlying primitive on
the magnitudes, and re-boxes the result.  Uncovered primitives fail loud via
:meth:`duq.jax.Quantity.default`, naming the primitive.

Plain (unit-free) operands mirror the NumPy layer's policy: arithmetic
accepts them only where the reference quantity is dimensionless, and rejects
them otherwise.  Two JAX-specific relaxations keep the ``jax.numpy``
compositions (``trunc``, ``remainder``, ``var``, ``isinf``, ``pad``, ...)
working, since they inject plain literals at the primitive level:

- comparisons accept a *concrete* plain operand whose elements are all
  scale-invariant (``0``, ``±inf`` or ``nan``) against any linear unit;
- the structural combination primitives (``select_n``, ``pad``,
  ``dynamic_update_slice``, ``scatter*``) let a plain operand adopt the
  reference quantity's unit (unxt-parity), because their internal literals
  are already traced inside jit and cannot be inspected.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import quax
from jax import lax
from jax.typing import ArrayLike

from duq._dimension import Dimension, _coerce_exponent
from duq._errors import (
    AffineUnitError,
    DimensionalityError,
    UnsupportedOperationError,
)
from duq._rules import Rule, apply
from duq._unit import Unit

from ._quantity import Quantity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ("PRIMITIVE_COVERAGE",)

_DIMENSIONLESS = Dimension({})

#: Registered primitive names -> short unit-rule description (drives the
#: coverage table in ``docs/dev/jax_coverage.md`` and its sync test).
_COVERAGE: dict[str, str] = {}


def _record(primitive: Any, description: str) -> None:
    """Record a registered primitive for the coverage table."""
    _COVERAGE[str(primitive.name)] = description


# -- small helpers -------------------------------------------------------------


def _mag(x: object) -> Any:
    """Return the raw magnitude of a Quantity, or ``x`` unchanged."""
    return x.magnitude if isinstance(x, Quantity) else x


def _unit_of(x: object) -> Unit | None:
    """Return the unit of a Quantity, or ``None`` for a plain operand."""
    return x.unit if isinstance(x, Quantity) else None


def _dim_of(x: object) -> Dimension:
    """Return the dimension of a Quantity, or dimensionless for a plain operand."""
    return x.dimension if isinstance(x, Quantity) else _DIMENSIONLESS


def _registry_of(*xs: object) -> Any:
    """Return the registry of the first Quantity among ``xs``."""
    for x in xs:
        if isinstance(x, Quantity):
            return x.unit.registry
    raise UnsupportedOperationError(
        "expected at least one duq.jax.Quantity operand"
    )  # pragma: no cover - rules dispatch only with a Quantity operand


def _dimensionless(*xs: object) -> Unit:
    """Return the bare dimensionless unit of the operands' registry."""
    return _registry_of(*xs).unit("1")  # type: ignore[no-any-return]


def _radian(*xs: object) -> Unit:
    """Return the radian unit of the operands' registry."""
    return _registry_of(*xs).unit("rad")  # type: ignore[no-any-return]


def _convert_mag(mag: Any, src: Unit, tgt: Unit) -> Any:
    """Convert a bare magnitude from ``src`` to ``tgt`` using float scale/offset."""
    if src.scale == tgt.scale and src.offset == tgt.offset:
        return mag
    s, s_off = float(src.scale), float(src.offset)
    t, t_off = float(tgt.scale), float(tgt.offset)
    return (mag * s + s_off - t_off) / t


def _is_scale_invariant(x: object) -> bool:
    """Return whether ``x`` is a concrete plain operand of only 0/±inf/nan."""
    if isinstance(x, Quantity | jax.core.Tracer):
        return False
    arr = np.asarray(x)
    if arr.dtype == np.bool_:
        return not bool(arr.any())
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        return bool(np.all(np.isnan(arr) | np.isinf(arr) | (arr == 0)))
    return bool(np.all(arr == 0))


def _to_unit(x: object, unit: Unit, *, invariant_ok: bool = False) -> Any:
    """Convert operand ``x`` to ``unit`` and return its bare magnitude.

    A plain operand is accepted as-is when ``unit`` is dimensionless (NumPy
    layer parity), or -- when ``invariant_ok`` -- when it is concrete and
    scale-invariant (all 0/±inf/nan) against a non-affine unit.
    """
    if isinstance(x, Quantity):
        apply(Rule.SAME_DIM, x.dimension, unit.dimension)
        return _convert_mag(x.magnitude, x.unit, unit)
    if unit.is_dimensionless:
        return x
    if invariant_ok and not unit.is_affine and _is_scale_invariant(x):
        return x
    raise DimensionalityError(
        f"cannot combine a bare array/number with a quantity of dimension {unit.dimension}"
    )


def _to_pure(x: object) -> Any:
    """Return the plain dimensionless value of ``x`` (``%`` and ``deg`` rescaled)."""
    if isinstance(x, Quantity):
        if not x.unit.is_dimensionless:
            raise DimensionalityError(
                f"expected a dimensionless or angle operand, got dimension {x.unit.dimension}"
            )
        return x.magnitude * float(x.unit.scale)
    return x


def _reject_affine(*units: Unit | None, action: str = "scale") -> None:
    """Raise :class:`AffineUnitError` if any of ``units`` is affine (e.g. ``°C``)."""
    for unit in units:
        if unit is not None and unit.is_affine:
            raise AffineUnitError(f"cannot {action} an affine quantity such as °C or °F")


def _si_mag(q: Quantity) -> Any:
    """Return the magnitude expressed in the coherent SI unit."""
    return q.magnitude * float(q.unit.scale) + float(q.unit.offset)


# -- additive / same-dimension ---------------------------------------------------


def _add_sub(prim: Any, x: object, y: object, params: dict[str, Any], sign: int) -> Quantity:
    """Bind an add/sub primitive under the core affine-aware policy."""
    dl = _dimensionless(x, y)
    qx = x if isinstance(x, Quantity) else Quantity(x, dl)
    qy = y if isinstance(y, Quantity) else Quantity(y, dl)
    apply(Rule.SAME_DIM, qx.dimension, qy.dimension)
    x_affine, y_affine = qx.unit.is_affine, qy.unit.is_affine
    if x_affine and y_affine:
        if sign > 0:
            raise AffineUnitError("cannot add two absolute (affine) quantities")
        coherent = qx.unit.registry.coherent_unit(qx.dimension)
        return Quantity(prim.bind(_si_mag(qx), _si_mag(qy), **params), coherent)
    if y_affine:
        raise AffineUnitError(
            "cannot combine a relative quantity with an absolute (affine) quantity"
        )
    if x_affine:
        ratio = float(qy.unit.scale) / float(qx.unit.scale)
        return Quantity(prim.bind(qx.magnitude, qy.magnitude * ratio, **params), qx.unit)
    return Quantity(prim.bind(qx.magnitude, _to_unit(qy, qx.unit), **params), qx.unit)


@quax.register(lax.add_p)
def _add_qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
    return _add_sub(lax.add_p, x, y, params, 1)


@quax.register(lax.add_p)
def _add_qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
    return _add_sub(lax.add_p, x, y, params, 1)


@quax.register(lax.add_p)
def _add_aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
    return _add_sub(lax.add_p, x, y, params, 1)


_record(lax.add_p, "same dimension; right operand converted to the left unit")


@quax.register(lax.sub_p)
def _sub_qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
    return _add_sub(lax.sub_p, x, y, params, -1)


@quax.register(lax.sub_p)
def _sub_qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
    return _add_sub(lax.sub_p, x, y, params, -1)


@quax.register(lax.sub_p)
def _sub_aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
    return _add_sub(lax.sub_p, x, y, params, -1)


_record(lax.sub_p, "same dimension; °C − °C yields the coherent SI difference (K)")


def _same_dim(prim: Any, x: object, y: object, params: dict[str, Any]) -> Quantity:
    """Bind a same-dimension primitive; the result keeps the reference unit."""
    reference = _unit_of(x) or _unit_of(y)
    assert reference is not None  # dispatch guarantees a Quantity operand
    return Quantity(
        prim.bind(_to_unit(x, reference), _to_unit(y, reference), **params), reference
    )


def _register_same_dim(prim: Any, description: str) -> None:
    """Register QQ/QA/AQ same-dimension rules for ``prim``."""

    @quax.register(prim)
    def _qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
        return _same_dim(prim, x, y, params)

    @quax.register(prim)
    def _qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
        return _same_dim(prim, x, y, params)

    @quax.register(prim)
    def _aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
        return _same_dim(prim, x, y, params)

    _record(prim, description)


_register_same_dim(lax.max_p, "same dimension; unit preserved")
_register_same_dim(lax.min_p, "same dimension; unit preserved")
_register_same_dim(lax.rem_p, "same dimension; unit preserved")
_register_same_dim(lax.nextafter_p, "same dimension; unit preserved")


# -- multiplicative ----------------------------------------------------------------


def _mul_div(prim: Any, x: object, y: object, params: dict[str, Any], *, div: bool) -> Quantity:
    ux, uy = _unit_of(x), _unit_of(y)
    _reject_affine(ux, uy)
    dl = _dimensionless(x, y)
    ux = dl if ux is None else ux
    uy = dl if uy is None else uy
    unit = ux / uy if div else ux * uy
    return Quantity(prim.bind(_mag(x), _mag(y), **params), unit)


def _register_mul_div(prim: Any, *, div: bool, description: str) -> None:
    """Register QQ/QA/AQ multiply/divide rules for ``prim``."""

    @quax.register(prim)
    def _qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
        return _mul_div(prim, x, y, params, div=div)

    @quax.register(prim)
    def _qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
        return _mul_div(prim, x, y, params, div=div)

    @quax.register(prim)
    def _aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
        return _mul_div(prim, x, y, params, div=div)

    _record(prim, description)


_register_mul_div(lax.mul_p, div=False, description="units multiply")
_register_mul_div(lax.div_p, div=True, description="units divide")


@quax.register(lax.dot_general_p)
def _dot_qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
    _reject_affine(x.unit, y.unit)
    return Quantity(lax.dot_general_p.bind(x.magnitude, y.magnitude, **params), x.unit * y.unit)


@quax.register(lax.dot_general_p)
def _dot_qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
    _reject_affine(x.unit)
    return Quantity(lax.dot_general_p.bind(x.magnitude, y, **params), x.unit)


@quax.register(lax.dot_general_p)
def _dot_aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
    _reject_affine(y.unit)
    return Quantity(lax.dot_general_p.bind(x, y.magnitude, **params), y.unit)


_record(lax.dot_general_p, "units multiply (matmul/dot/einsum contractions)")


# -- powers ------------------------------------------------------------------------


def _concrete_exponent(y: object) -> Fraction | None:
    """Return a uniform concrete exponent, or ``None`` when ``y`` is traced."""
    value = _to_pure(y) if isinstance(y, Quantity) else y
    if isinstance(value, jax.core.Tracer):
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _coerce_exponent(arr.item())
    first = arr.reshape(-1)[0].item()
    if not bool(np.all(arr == first)):
        raise UnsupportedOperationError(
            "power with an array of differing exponents is unsupported: each "
            "element would carry a different unit"
        )
    return _coerce_exponent(first)


def _pow(x: object, y: object, params: dict[str, Any]) -> Quantity | Any:
    unit = _unit_of(x)
    exponent = _concrete_exponent(y)
    y_mag = _to_pure(y) if isinstance(y, Quantity) else y
    result = lax.pow_p.bind(_mag(x), y_mag, **params)
    if unit is None:
        return result
    if exponent is None:
        if unit.is_dimensionless and unit.scale == 1:
            return Quantity(result, unit)
        raise UnsupportedOperationError(
            "a traced power exponent is only supported for a scale-1 "
            "dimensionless base; the output unit would depend on a runtime value"
        )
    return Quantity(result, unit**exponent)


@quax.register(lax.pow_p)
def _pow_qq(x: Quantity, y: Quantity, **params: Any) -> Quantity | Any:
    return _pow(x, y, params)


@quax.register(lax.pow_p)
def _pow_qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity | Any:
    return _pow(x, y, params)


@quax.register(lax.pow_p)
def _pow_aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity | Any:
    return _pow(x, y, params)


_record(lax.pow_p, "unit ** concrete dimensionless exponent (traced exponent rejected)")


@quax.register(lax.integer_pow_p)
def _integer_pow_q(x: Quantity, *, y: int) -> Quantity:
    return Quantity(lax.integer_pow_p.bind(x.magnitude, y=y), x.unit**y)


_record(lax.integer_pow_p, "unit ** y (static integer exponent)")


def _register_unit_power(prim: Any, exponent: Fraction, description: str) -> None:
    """Register a unary rule raising the unit to a fixed ``exponent``."""

    @quax.register(prim)
    def _rule(x: Quantity, **params: Any) -> Quantity:
        return Quantity(prim.bind(x.magnitude, **params), x.unit**exponent)

    _record(prim, description)


_register_unit_power(lax.square_p, Fraction(2), "unit squared")
_register_unit_power(lax.sqrt_p, Fraction(1, 2), "unit ** 1/2")
_register_unit_power(lax.rsqrt_p, Fraction(-1, 2), "unit ** -1/2")
_register_unit_power(lax.cbrt_p, Fraction(1, 3), "unit ** 1/3")


# -- unary preserving / plain ---------------------------------------------------------


def _register_preserve(prim: Any, description: str, *, no_affine: bool = False) -> None:
    """Register a unary rule that preserves the operand's unit."""

    @quax.register(prim)
    def _rule(x: Quantity, **params: Any) -> Quantity:
        if no_affine:
            _reject_affine(x.unit, action=f"apply {prim.name} to")
        return Quantity(prim.bind(x.magnitude, **params), x.unit)

    _record(prim, description)


_register_preserve(lax.neg_p, "unit preserved (affine rejected)", no_affine=True)
_register_preserve(lax.abs_p, "unit preserved (affine rejected)", no_affine=True)
_register_preserve(lax.floor_p, "unit preserved")
_register_preserve(lax.ceil_p, "unit preserved")
_register_preserve(lax.round_p, "unit preserved")
_register_preserve(lax.real_p, "unit preserved")
_register_preserve(lax.imag_p, "unit preserved")
_register_preserve(lax.conj_p, "unit preserved")

for _prim in (
    lax.broadcast_in_dim_p,
    lax.reshape_p,
    lax.transpose_p,
    lax.squeeze_p,
    lax.rev_p,
    lax.slice_p,
    lax.copy_p,
    lax.tile_p,
    lax.convert_element_type_p,
    lax.stop_gradient_p,
):
    _register_preserve(_prim, "structural; unit preserved")

for _prim in (lax.reduce_sum_p, lax.reduce_max_p, lax.reduce_min_p):
    _register_preserve(_prim, "reduction; unit preserved")

for _prim in (lax.cumsum_p, lax.cummax_p, lax.cummin_p):
    _register_preserve(_prim, "cumulative; unit preserved")


def _register_plain_out(prim: Any, description: str) -> None:
    """Register a unary rule that strips the unit (plain array out)."""

    @quax.register(prim)
    def _rule(x: Quantity, **params: Any) -> Any:
        return prim.bind(x.magnitude, **params)

    _record(prim, description)


_register_plain_out(lax.sign_p, "plain array out (sign is scale-free)")
_register_plain_out(lax.is_finite_p, "plain bool array out")
_register_plain_out(lax.argmax_p, "plain index array out")
_register_plain_out(lax.argmin_p, "plain index array out")
_register_plain_out(lax.reduce_and_p, "plain bool array out")
_register_plain_out(lax.reduce_or_p, "plain bool array out")
_register_plain_out(lax.bitcast_convert_type_p, "plain array out (raw bits carry no unit)")


@quax.register(lax.reduce_prod_p)
def _reduce_prod_q(x: Quantity, *, axes: tuple[int, ...], **params: Any) -> Quantity:
    _reject_affine(x.unit)
    n = math.prod(x.magnitude.shape[axis] for axis in axes)
    return Quantity(lax.reduce_prod_p.bind(x.magnitude, axes=axes, **params), x.unit**n)


_record(lax.reduce_prod_p, "unit ** reduced_size (shapes are static)")


@quax.register(lax.cumprod_p)
def _cumprod_q(x: Quantity, **params: Any) -> Quantity:
    if not (x.unit.is_dimensionless and x.unit.scale == 1):
        raise UnsupportedOperationError(
            "cumprod is unsupported for dimensional (or scaled-dimensionless) "
            "quantities: each element would carry a different unit"
        )
    return Quantity(lax.cumprod_p.bind(x.magnitude, **params), x.unit)


_record(lax.cumprod_p, "scale-1 dimensionless only (per-element unit is ambiguous)")


# -- dimensionless-in and angle rules ---------------------------------------------------


def _register_dimensionless_in(prim: Any, description: str) -> None:
    """Register a unary rule requiring a dimensionless input (scale applied)."""

    @quax.register(prim)
    def _rule(x: Quantity, **params: Any) -> Quantity:
        return Quantity(prim.bind(_to_pure(x), **params), _dimensionless(x))

    _record(prim, description)


for _prim in (
    lax.exp_p,
    lax.exp2_p,
    lax.expm1_p,
    lax.log_p,
    lax.log1p_p,
    lax.logistic_p,
    lax.tanh_p,
    lax.sinh_p,
    lax.cosh_p,
    lax.asinh_p,
    lax.acosh_p,
    lax.atanh_p,
    lax.erf_p,
    lax.erfc_p,
    lax.erf_inv_p,
):
    _register_dimensionless_in(_prim, "dimensionless in (scale applied), dimensionless out")

for _prim in (lax.sin_p, lax.cos_p, lax.tan_p):
    _register_dimensionless_in(
        _prim, "angle in (rad/deg scale-converted at trace time), dimensionless out"
    )


def _register_radian_out(prim: Any, description: str) -> None:
    """Register a unary inverse-trig rule (dimensionless in, radian out)."""

    @quax.register(prim)
    def _rule(x: Quantity, **params: Any) -> Quantity:
        return Quantity(prim.bind(_to_pure(x), **params), _radian(x))

    _record(prim, description)


_register_radian_out(lax.asin_p, "dimensionless in, radian out")
_register_radian_out(lax.acos_p, "dimensionless in, radian out")
_register_radian_out(lax.atan_p, "dimensionless in, radian out")


def _atan2(x: object, y: object, params: dict[str, Any]) -> Quantity:
    reference = _unit_of(x) or _unit_of(y)
    assert reference is not None  # dispatch guarantees a Quantity operand
    result = lax.atan2_p.bind(_to_unit(x, reference), _to_unit(y, reference), **params)
    return Quantity(result, _radian(x, y))


@quax.register(lax.atan2_p)
def _atan2_qq(x: Quantity, y: Quantity, **params: Any) -> Quantity:
    return _atan2(x, y, params)


@quax.register(lax.atan2_p)
def _atan2_qa(x: Quantity, y: ArrayLike, **params: Any) -> Quantity:
    return _atan2(x, y, params)


@quax.register(lax.atan2_p)
def _atan2_aq(x: ArrayLike, y: Quantity, **params: Any) -> Quantity:
    return _atan2(x, y, params)


_record(lax.atan2_p, "same dimension in, radian out")


# -- comparisons -------------------------------------------------------------------------


def _broadcast_bool(x: object, y: object, *, fill: bool) -> Any:
    shape = jnp.broadcast_shapes(jnp.shape(_mag(x)), jnp.shape(_mag(y)))
    return jnp.full(shape, fill, dtype=bool)


def _compare(prim: Any, x: object, y: object, params: dict[str, Any], kind: str) -> Any:
    """Bind a comparison primitive with unit conversion (plain bool array out).

    ``eq``/``ne`` on incompatible dimensions produce constant all-False /
    all-True arrays (NumPy layer parity); ordering raises.  A concrete
    scale-invariant plain operand (all 0/±inf/nan) compares raw against any
    linear unit -- ``jax.numpy`` compositions rely on such literals.
    """
    ux, uy = _unit_of(x), _unit_of(y)
    reference = ux if ux is not None else uy
    assert reference is not None  # dispatch guarantees a Quantity operand
    plain = y if ux is not None else x
    if not isinstance(plain, Quantity) and not reference.is_affine and _is_scale_invariant(plain):
        return prim.bind(_mag(x), _mag(y), **params)
    if _dim_of(x) != _dim_of(y):
        if kind == "eq":
            return _broadcast_bool(x, y, fill=False)
        if kind == "ne":
            return _broadcast_bool(x, y, fill=True)
        raise DimensionalityError(
            f"cannot order-compare quantities of dimension {_dim_of(x)} and {_dim_of(y)}"
        )
    return prim.bind(_to_unit(x, reference), _to_unit(y, reference), **params)


def _register_compare(prim: Any, kind: str, description: str) -> None:
    """Register QQ/QA/AQ comparison rules for ``prim``."""

    @quax.register(prim)
    def _qq(x: Quantity, y: Quantity, **params: Any) -> Any:
        return _compare(prim, x, y, params, kind)

    @quax.register(prim)
    def _qa(x: Quantity, y: ArrayLike, **params: Any) -> Any:
        return _compare(prim, x, y, params, kind)

    @quax.register(prim)
    def _aq(x: ArrayLike, y: Quantity, **params: Any) -> Any:
        return _compare(prim, x, y, params, kind)

    _record(prim, description)


_register_compare(lax.eq_p, "eq", "converted compare; incompatible dims -> all-False")
_register_compare(lax.ne_p, "ne", "converted compare; incompatible dims -> all-True")
_register_compare(lax.lt_p, "order", "converted compare; incompatible dims raise")
_register_compare(lax.le_p, "order", "converted compare; incompatible dims raise")
_register_compare(lax.gt_p, "order", "converted compare; incompatible dims raise")
_register_compare(lax.ge_p, "order", "converted compare; incompatible dims raise")
_register_compare(lax.eq_to_p, "eq", "total-order compare (converted)")
_register_compare(lax.lt_to_p, "order", "total-order compare (converted)")
_register_compare(lax.le_to_p, "order", "total-order compare (converted)")


# -- n-ary structure ----------------------------------------------------------------------


def _first_unit(xs: Sequence[object]) -> Unit | None:
    for x in xs:
        if isinstance(x, Quantity):
            return x.unit
    return None


@quax.register(lax.concatenate_p)
def _concatenate(*xs: Quantity | ArrayLike, **params: Any) -> Quantity | Any:
    unit = _first_unit(xs)
    if unit is None:
        return lax.concatenate_p.bind(*xs, **params)
    mags = [_to_unit(x, unit) for x in xs]
    return Quantity(lax.concatenate_p.bind(*mags, **params), unit)


_record(lax.concatenate_p, "all operands converted to the first quantity's unit")


@quax.register(lax.stack_p)
def _stack(*xs: Quantity | ArrayLike, **params: Any) -> Quantity | Any:
    unit = _first_unit(xs)
    if unit is None:
        return lax.stack_p.bind(*xs, **params)
    mags = [_to_unit(x, unit) for x in xs]
    return Quantity(lax.stack_p.bind(*mags, **params), unit)


_record(lax.stack_p, "all operands converted to the first quantity's unit")


@quax.register(lax.select_n_p)
def _select_n(pred: ArrayLike, *cases: Quantity | ArrayLike, **params: Any) -> Quantity | Any:
    # Plain branches pass raw and adopt the reference unit (unxt-parity): the
    # jax.numpy compositions (var, hypot, trunc, take, ...) select against
    # internal literals that are already traced inside jit and cannot be
    # inspected for scale invariance.
    unit = _first_unit(cases)
    if unit is None:
        return lax.select_n_p.bind(pred, *cases, **params)
    mags = [_to_unit(case, unit) if isinstance(case, Quantity) else case for case in cases]
    return Quantity(lax.select_n_p.bind(pred, *mags, **params), unit)


_record(
    lax.select_n_p,
    "quantity branches converted to the first quantity's unit; plain branches "
    "adopt it; predicate plain",
)


@quax.register(lax.sort_p)
def _sort(*xs: Quantity | ArrayLike, **params: Any) -> list[Any]:
    units = [_unit_of(x) for x in xs]
    outs = lax.sort_p.bind(*(_mag(x) for x in xs), **params)
    return [
        Quantity(out, unit) if unit is not None else out
        for out, unit in zip(outs, units, strict=True)
    ]


_record(lax.sort_p, "per-operand; each output keeps its operand's unit")


@quax.register(lax.split_p)
def _split(x: Quantity, **params: Any) -> list[Any]:
    outs = lax.split_p.bind(x.magnitude, **params)
    return [Quantity(out, x.unit) for out in outs]


_record(lax.split_p, "structural; unit preserved on every output")


@quax.register(lax.pad_p)
def _pad_qq(x: Quantity, padding_value: Quantity, **params: Any) -> Quantity:
    return Quantity(
        lax.pad_p.bind(x.magnitude, _to_unit(padding_value, x.unit), **params), x.unit
    )


@quax.register(lax.pad_p)
def _pad_qa(x: Quantity, padding_value: ArrayLike, **params: Any) -> Quantity:
    # A plain padding value adopts the operand's unit (unxt parity):
    # ``jnp.pad``'s default zero is already traced inside its internal jit.
    return Quantity(lax.pad_p.bind(x.magnitude, padding_value, **params), x.unit)


_record(lax.pad_p, "quantity padding value converted to the operand's unit; plain adopts it")


# -- indexing ---------------------------------------------------------------------------


@quax.register(lax.gather_p)
def _gather(x: Quantity, indices: ArrayLike, **params: Any) -> Quantity:
    return Quantity(lax.gather_p.bind(x.magnitude, indices, **params), x.unit)


_record(lax.gather_p, "indexing; unit preserved")


@quax.register(lax.dynamic_slice_p)
def _dynamic_slice(x: Quantity, *starts: ArrayLike, **params: Any) -> Quantity:
    return Quantity(lax.dynamic_slice_p.bind(x.magnitude, *starts, **params), x.unit)


_record(lax.dynamic_slice_p, "indexing; unit preserved")


@quax.register(lax.dynamic_update_slice_p)
def _dus_qq(x: Quantity, update: Quantity, *starts: ArrayLike, **params: Any) -> Quantity:
    mag = _to_unit(update, x.unit)
    return Quantity(lax.dynamic_update_slice_p.bind(x.magnitude, mag, *starts, **params), x.unit)


@quax.register(lax.dynamic_update_slice_p)
def _dus_qa(x: Quantity, update: ArrayLike, *starts: ArrayLike, **params: Any) -> Quantity:
    # A plain update adopts the operand's unit (unxt parity; see ``_select_n``).
    return Quantity(
        lax.dynamic_update_slice_p.bind(x.magnitude, update, *starts, **params), x.unit
    )


_record(
    lax.dynamic_update_slice_p,
    "quantity update converted to the operand's unit; plain update adopts it",
)


def _register_scatter(prim: Any, description: str) -> None:
    """Register scatter rules converting the updates to the operand's unit."""

    @quax.register(prim)
    def _qq(x: Quantity, indices: ArrayLike, updates: Quantity, **params: Any) -> Quantity:
        mag = _to_unit(updates, x.unit)
        return Quantity(prim.bind(x.magnitude, indices, mag, **params), x.unit)

    @quax.register(prim)
    def _qa(x: Quantity, indices: ArrayLike, updates: ArrayLike, **params: Any) -> Quantity:
        # Plain updates adopt the operand's unit (unxt parity; see ``_select_n``).
        return Quantity(prim.bind(x.magnitude, indices, updates, **params), x.unit)

    _record(prim, description)


_SCATTER_RULE = "quantity updates converted to the operand's unit; plain updates adopt it"
_register_scatter(lax.scatter_p, _SCATTER_RULE)
_register_scatter(lax.scatter_add_p, _SCATTER_RULE)
_register_scatter(lax.scatter_sub_p, _SCATTER_RULE)
_register_scatter(lax.scatter_min_p, _SCATTER_RULE)
_register_scatter(lax.scatter_max_p, _SCATTER_RULE)


#: Immutable coverage table: (primitive name, unit-rule description) pairs,
#: sorted by primitive name.  Control-flow primitives (``cond``/``while``/
#: ``scan``/``pjit``) are handled by quax itself and are not listed here.
PRIMITIVE_COVERAGE: tuple[tuple[str, str], ...] = tuple(sorted(_COVERAGE.items()))
