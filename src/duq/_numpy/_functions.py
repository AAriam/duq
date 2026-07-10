"""The curated ``__array_function__`` table: NumPy function -> duq unit rule.

Each handler receives the original ``(args, kwargs)``, strips Quantity operands
to bare magnitudes (converting where a rule demands it), calls the underlying
NumPy function, and re-boxes the result with the correct unit.  Functions absent
from :data:`FUNCTION_TABLE` fail loud in :mod:`duq._numpy._dispatch`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from duq._errors import DimensionalityError, UnsupportedOperationError

from ._common import (
    Quantity,
    box,
    convert_sequence,
    dimensionless_unit,
    find_quantity,
    magnitude,
    registry_of,
    to_unit,
    unit_of,
)
from ._common import dimension_of as _dimension_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from duq._unit import Unit

    Handler = Callable[[tuple[object, ...], dict[str, Any]], object]

__all__ = ("FUNCTION_TABLE",)

# A dynamically-typed view of NumPy: these handlers feed already-stripped (unit-
# free) magnitudes plus forwarded ``*args``/``**kwargs`` straight through to the
# corresponding NumPy function, so per-call static overload resolution adds no
# safety here and only fights the deliberately generic ``object`` signatures.
_np: Any = np


def _rebox(result: object, unit: Unit) -> object:
    if isinstance(result, tuple):
        return tuple(box(r, unit) for r in result)
    if isinstance(result, list):
        return [box(r, unit) for r in result]
    return box(result, unit)


def _reference_unit(*operands: object) -> Unit:
    for operand in operands:
        found = find_quantity(operand)
        if found is not None:
            return found.unit
    raise UnsupportedOperationError("expected at least one duq.Quantity operand")


# -- preserve / squared ------------------------------------------------------


def _make_preserve_first(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        first = args[0]
        unit = unit_of(first)
        result = func(magnitude(first), *args[1:], **kwargs)
        return _rebox(result, unit) if unit is not None else result

    return handler


def _make_squared_first(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        first = args[0]
        unit = unit_of(first)
        assert unit is not None
        result = func(magnitude(first), *args[1:], **kwargs)
        return _rebox(result, unit**2)

    return handler


def _make_plain_first(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        return func(magnitude(args[0]), *args[1:], **kwargs)

    return handler


# -- joining / selection -----------------------------------------------------


def _make_concat(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        sequence = args[0]
        assert isinstance(sequence, list | tuple)
        unit = _reference_unit(sequence)
        mags = convert_sequence(sequence, unit)
        result = func(mags, *args[1:], **kwargs)
        return box(result, unit)

    return handler


def _h_append(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    arr, values = args[0], args[1]
    unit = unit_of(arr) if unit_of(arr) is not None else unit_of(values)
    assert unit is not None
    result = _np.append(to_unit(arr, unit), to_unit(values, unit), *args[2:], **kwargs)
    return box(result, unit)


def _h_insert(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    arr, obj = args[0], args[1]
    values = args[2] if len(args) > 2 else kwargs.pop("values")
    unit = unit_of(arr)
    assert unit is not None
    result = _np.insert(magnitude(arr), magnitude(obj), to_unit(values, unit), *args[3:], **kwargs)
    return box(result, unit)


def _h_where(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    if len(args) == 1:
        return _np.where(magnitude(args[0]))
    cond, a, b = args[0], args[1], args[2]
    unit = unit_of(a) if unit_of(a) is not None else unit_of(b)
    if unit is None:
        return _np.where(magnitude(cond), magnitude(a), magnitude(b))
    return box(_np.where(magnitude(cond), to_unit(a, unit), to_unit(b, unit)), unit)


def _h_clip(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    a = args[0]
    unit = unit_of(a)
    assert unit is not None
    lo = args[1] if len(args) > 1 else kwargs.pop("a_min", kwargs.pop("min", None))
    hi = args[2] if len(args) > 2 else kwargs.pop("a_max", kwargs.pop("max", None))
    m_lo = None if lo is None else to_unit(lo, unit)
    m_hi = None if hi is None else to_unit(hi, unit)
    return box(_np.clip(magnitude(a), m_lo, m_hi, **kwargs), unit)


def _h_compress(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    condition, a = args[0], args[1]
    unit = unit_of(a)
    result = _np.compress(magnitude(condition), magnitude(a), *args[2:], **kwargs)
    return box(result, unit) if unit is not None else result


# -- linalg-ish products -----------------------------------------------------


def _make_product(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        a, b = args[0], args[1]
        dl = dimensionless_unit(registry_of(a, b))
        ua = unit_of(a)
        ub = unit_of(b)
        ua = dl if ua is None else ua
        ub = dl if ub is None else ub
        result = func(magnitude(a), magnitude(b), *args[2:], **kwargs)
        return box(result, ua * ub)

    return handler


def _h_einsum(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    subscripts, operands = args[0], args[1:]
    unit: Unit | None = None
    mags = []
    for operand in operands:
        current = unit_of(operand)
        if current is not None:
            unit = current if unit is None else unit * current
        mags.append(magnitude(operand))
    result = _np.einsum(subscripts, *mags, **kwargs)
    return box(result, unit) if unit is not None else result


# -- comparison / search -----------------------------------------------------


def _make_close(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        kwargs = dict(kwargs)
        a, b = args[0], args[1]
        unit = unit_of(a) if unit_of(a) is not None else unit_of(b)
        assert unit is not None
        ma, mb = to_unit(a, unit), to_unit(b, unit)
        rtol = kwargs.pop("rtol", args[2] if len(args) > 2 else 1e-05)
        atol_positional = args[3] if len(args) > 3 else None
        atol_given = "atol" in kwargs or len(args) > 3
        atol_raw = kwargs.pop("atol", atol_positional)
        if isinstance(atol_raw, Quantity):
            atol: Any = to_unit(atol_raw, unit)
        elif atol_given:
            if not unit.is_dimensionless:
                raise DimensionalityError(
                    "atol must be a Quantity of the same dimension as the compared "
                    "arrays (a bare atol is only valid for dimensionless quantities)"
                )
            atol = atol_raw
        else:
            if not unit.is_dimensionless:
                raise DimensionalityError(
                    "the default atol is only valid for dimensionless quantities; "
                    "pass an explicit Quantity atol of the same dimension"
                )
            atol = 1e-08
        return func(ma, mb, rtol=rtol, atol=atol, **kwargs)

    return handler


def _make_array_equal(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        a, b = args[0], args[1]
        if _dimension_of(a) != _dimension_of(b):
            return False
        unit = unit_of(a) if unit_of(a) is not None else unit_of(b)
        assert unit is not None
        return func(to_unit(a, unit), to_unit(b, unit), *args[2:], **kwargs)

    return handler


def _h_searchsorted(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    a = args[0]
    v = args[1] if len(args) > 1 else kwargs.pop("v")
    unit = unit_of(a)
    assert unit is not None
    return _np.searchsorted(magnitude(a), to_unit(v, unit), *args[2:], **kwargs)


def _h_digitize(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    x = args[0]
    bins = args[1] if len(args) > 1 else kwargs.pop("bins")
    unit = unit_of(x)
    assert unit is not None
    return _np.digitize(magnitude(x), to_unit(bins, unit), *args[2:], **kwargs)


def _h_unique(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    q = args[0]
    unit = unit_of(q)
    assert unit is not None
    result = _np.unique(magnitude(q), *args[1:], **kwargs)
    if isinstance(result, tuple):
        return (box(result[0], unit), *result[1:])
    return box(result, unit)


def _h_interp(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    x, xp, fp = args[0], args[1], args[2]
    x_unit = unit_of(xp) if unit_of(xp) is not None else unit_of(x)
    mx = to_unit(x, x_unit) if x_unit is not None else magnitude(x)
    mxp = to_unit(xp, x_unit) if x_unit is not None else magnitude(xp)
    fp_unit = unit_of(fp)
    result = _np.interp(mx, mxp, magnitude(fp), *args[3:], **kwargs)
    return box(result, fp_unit) if fp_unit is not None else result


def _h_meshgrid(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    units = [unit_of(a) for a in args]
    grids = _np.meshgrid(*(magnitude(a) for a in args), **kwargs)
    return [box(g, u) if u is not None else g for g, u in zip(grids, units, strict=True)]


def _h_histogram(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    a = args[0]
    unit = unit_of(a)
    assert unit is not None
    bins = kwargs.pop("bins", args[1] if len(args) > 1 else 10)
    if isinstance(bins, Quantity):
        bins = to_unit(bins, unit)
    hist_range = kwargs.pop("range", None)
    if isinstance(hist_range, Quantity):
        hist_range = to_unit(hist_range, unit)
    counts, edges = _np.histogram(magnitude(a), bins=bins, range=hist_range, **kwargs)
    return counts, box(edges, unit)


# -- calculus ----------------------------------------------------------------


def _h_gradient(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    f = args[0]
    f_unit = unit_of(f)
    assert f_unit is not None
    out_unit = f_unit
    mags = [magnitude(f)]
    for spacing in args[1:]:
        spacing_unit = unit_of(spacing)
        if spacing_unit is not None:
            out_unit = f_unit / spacing_unit
        mags.append(magnitude(spacing))
    result = _np.gradient(*mags, **kwargs)
    if isinstance(result, list):
        return [box(r, out_unit) for r in result]
    return box(result, out_unit)


def _make_trapezoid(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        kwargs = dict(kwargs)
        y = args[0]
        y_unit = unit_of(y)
        assert y_unit is not None
        out_unit = y_unit
        x = kwargs.pop("x", args[1] if len(args) > 1 else None)
        dx = kwargs.pop("dx", None)
        if x is not None:
            x_unit = unit_of(x)
            if x_unit is not None:
                out_unit = y_unit * x_unit
            kwargs["x"] = magnitude(x)
        if dx is not None:
            dx_unit = unit_of(dx)
            if dx_unit is not None:
                out_unit = y_unit * dx_unit
            kwargs["dx"] = magnitude(dx)
        return box(func(magnitude(y), **kwargs), out_unit)

    return handler


def _h_average(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    a = args[0]
    unit = unit_of(a)
    assert unit is not None
    weights = kwargs.get("weights")
    if isinstance(weights, Quantity):
        kwargs["weights"] = magnitude(weights)
    result = _np.average(magnitude(a), *args[1:], **kwargs)
    if isinstance(result, tuple):
        return box(result[0], unit), result[1]
    return box(result, unit)


# -- creation-like -----------------------------------------------------------


def _make_like_preserve(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        first = args[0]
        unit = unit_of(first)
        assert unit is not None
        return box(func(magnitude(first), *args[1:], **kwargs), unit)

    return handler


def _h_full_like(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    a = args[0]
    unit = unit_of(a)
    assert unit is not None
    fill = args[1] if len(args) > 1 else kwargs.pop("fill_value")
    if isinstance(fill, Quantity):
        m_fill: Any = to_unit(fill, unit)
    elif unit.is_dimensionless:
        m_fill = fill
    else:
        raise DimensionalityError(
            "full_like fill_value must be a Quantity of the same dimension for a "
            "dimensional template array"
        )
    return box(_np.full_like(magnitude(a), m_fill, *args[2:], **kwargs), unit)


# -- shape/layout with multiple outputs --------------------------------------

_PAD_MODES = frozenset({"constant", "edge", "reflect", "symmetric", "wrap"})


def _h_pad(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    kwargs = dict(kwargs)
    array = args[0]
    unit = unit_of(array)
    assert unit is not None
    pad_width = args[1] if len(args) > 1 else kwargs.pop("pad_width")
    mode = args[2] if len(args) > 2 else kwargs.pop("mode", "constant")
    if mode not in _PAD_MODES:
        raise UnsupportedOperationError(
            f"numpy.pad mode {mode!r} is not supported by duq; use 'constant' or an "
            "edge-family mode (edge/reflect/symmetric/wrap)"
        )
    if "constant_values" in kwargs and isinstance(kwargs["constant_values"], Quantity):
        kwargs["constant_values"] = to_unit(kwargs["constant_values"], unit)
    return box(_np.pad(magnitude(array), pad_width, mode=mode, **kwargs), unit)


def _h_broadcast_arrays(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    units = [unit_of(a) for a in args]
    results = _np.broadcast_arrays(*(magnitude(a) for a in args), **kwargs)
    return [box(r, u) if u is not None else r for r, u in zip(results, units, strict=True)]


def _make_atleast(func: Any) -> Handler:
    def handler(args: tuple[object, ...], kwargs: dict[str, Any]) -> object:
        units = [unit_of(a) for a in args]
        result = func(*(magnitude(a) for a in args))
        if len(args) == 1:
            unit = units[0]
            return box(result, unit) if unit is not None else result
        return [box(r, u) if u is not None else r for r, u in zip(result, units, strict=True)]

    return handler


# -- table -------------------------------------------------------------------


def _build_table() -> dict[Any, Handler]:
    table: dict[Any, Handler] = {}

    def register(name: str, handler: Handler, *, module: Any = np) -> None:
        """Register an already-built handler by function name (if it exists)."""
        func = getattr(module, name, None)
        if func is not None:
            table[func] = handler

    def register_factory(name: str, factory: Any, *, module: Any = np) -> None:
        """Register ``factory(func)`` for ``module.name`` (skipping absent names)."""
        func = getattr(module, name, None)
        if func is not None:
            table[func] = factory(func)

    preserve_names = (
        # shape / layout
        "reshape",
        "ravel",
        "transpose",
        "permute_dims",
        "swapaxes",
        "moveaxis",
        "squeeze",
        "expand_dims",
        "broadcast_to",
        "flip",
        "fliplr",
        "flipud",
        "roll",
        "rot90",
        "tile",
        "repeat",
        "delete",
        "take",
        "take_along_axis",
        "diagonal",
        "diag",
        "real",
        "imag",
        "copy",
        "sort",
        "flatnonzero",
        "split",
        "array_split",
        "hsplit",
        "vsplit",
        "dsplit",
        # reductions preserving the unit
        "sum",
        "nansum",
        "cumsum",
        "nancumsum",
        "mean",
        "nanmean",
        "median",
        "nanmedian",
        "quantile",
        "percentile",
        "nanquantile",
        "nanpercentile",
        "min",
        "amin",
        "max",
        "amax",
        "nanmin",
        "nanmax",
        "ptp",
        "std",
        "nanstd",
        "diff",
        "ediff1d",
        "trace",
        "around",
        "round",
    )
    for name in preserve_names:
        register_factory(name, _make_preserve_first)

    for name in ("var", "nanvar"):
        register_factory(name, _make_squared_first)

    plain_names = (
        "argmin",
        "argmax",
        "nanargmin",
        "nanargmax",
        "argsort",
        "count_nonzero",
        "nonzero",
        "shape",
        "ndim",
        "size",
    )
    for name in plain_names:
        register_factory(name, _make_plain_first)

    for name in ("concatenate", "stack", "hstack", "vstack", "dstack", "column_stack"):
        register_factory(name, _make_concat)
    register("append", _h_append)
    register("insert", _h_insert)
    register("where", _h_where)
    register("clip", _h_clip)
    register("compress", _h_compress)

    for name in ("dot", "vdot", "inner", "outer", "tensordot", "cross", "kron"):
        register_factory(name, _make_product)
    register("einsum", _h_einsum)
    register_factory("norm", _make_preserve_first, module=np.linalg)

    register_factory("allclose", _make_close)
    register_factory("isclose", _make_close)
    register_factory("array_equal", _make_array_equal)
    register_factory("array_equiv", _make_array_equal)
    register("searchsorted", _h_searchsorted)
    register("digitize", _h_digitize)
    register("unique", _h_unique)
    register("interp", _h_interp)
    register("meshgrid", _h_meshgrid)
    register("histogram", _h_histogram)
    register("average", _h_average)

    register("gradient", _h_gradient)
    for name in ("trapezoid", "trapz"):  # numpy>=2 vs 1.26 naming
        register_factory(name, _make_trapezoid)

    for name in ("zeros_like", "empty_like", "ones_like"):
        register_factory(name, _make_like_preserve)
    register("full_like", _h_full_like)

    register("pad", _h_pad)
    register("broadcast_arrays", _h_broadcast_arrays)
    for name in ("atleast_1d", "atleast_2d", "atleast_3d"):
        register_factory(name, _make_atleast)

    return table


#: The curated ``__array_function__`` dispatch table (keyed by function object).
FUNCTION_TABLE: dict[Any, Handler] = _build_table()
