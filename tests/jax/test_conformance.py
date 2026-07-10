"""Shared conformance: the JAX layer must match the NumPy layer op-for-op.

Each case runs the same operation twice -- ``numpy.<f>`` on core
``duq.Quantity`` arrays and ``duq.jax.numpy.<f>`` on ``duq.jax.Quantity``
arrays -- and asserts identical magnitudes (allclose) and identical units
(or a plain, unit-free array on both sides).
"""

# The dispatch under test is deliberately dynamic; see tests/numpy/test_ufuncs.py.
# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, operator, index"

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import duq
import duq.jax
import duq.jax.numpy as djnp

P = [1.0, 2.0, 3.0]
Q2 = [2.0, 5.0, 7.0]
R = [0.25, 0.5, 0.75]
COND = [True, False, True]

# Each case: op name -> (function name, [(values, unit-or-None), ...], kwargs).
# A unit of None marks a plain (unit-free) operand.
Case = tuple[str, list[tuple[object, str | None]], dict[str, object]]
_CASES: dict[str, Case] = {
    # multiplicative
    "multiply": ("multiply", [(P, "m"), (Q2, "s")], {}),
    "divide": ("divide", [(P, "m"), (Q2, "s")], {}),
    "true_divide": ("true_divide", [(P, "m"), (Q2, "s")], {}),
    "floor_divide": ("floor_divide", [(Q2, "m"), (P, "m")], {}),
    "reciprocal": ("reciprocal", [(P, "m")], {}),
    # additive / same dimension (right operand converted)
    "add": ("add", [(P, "m"), (Q2, "m")], {}),
    "add_convert": ("add", [(P, "m"), ([200.0, 500.0, 700.0], "cm")], {}),
    "subtract": ("subtract", [(Q2, "m"), (P, "m")], {}),
    "maximum": ("maximum", [(P, "m"), (Q2, "m")], {}),
    "minimum": ("minimum", [(P, "m"), (Q2, "m")], {}),
    "fmax": ("fmax", [(P, "m"), (Q2, "m")], {}),
    "fmin": ("fmin", [(P, "m"), (Q2, "m")], {}),
    "remainder": ("remainder", [(Q2, "m"), (P, "m")], {}),
    "mod": ("mod", [(Q2, "m"), (P, "m")], {}),
    "fmod": ("fmod", [(Q2, "m"), (P, "m")], {}),
    "hypot": ("hypot", [([3.0], "m"), ([4.0], "m")], {}),
    "copysign": ("copysign", [(P, "m"), ([-1.0, 1.0, -1.0], "m")], {}),
    "nextafter": ("nextafter", [(P, "m"), (Q2, "m")], {}),
    # powers
    "power": ("power", [(P, "m"), (2, None)], {}),
    "float_power_like": ("power", [(P, "m"), (2.0, None)], {}),
    "sqrt": ("sqrt", [([1.0, 4.0, 9.0], "m^2")], {}),
    "cbrt": ("cbrt", [([1.0, 8.0, 27.0], "m^3")], {}),
    "square": ("square", [(P, "m")], {}),
    # unary preserving
    "negative": ("negative", [(P, "m")], {}),
    "positive": ("positive", [(P, "m")], {}),
    "absolute": ("absolute", [([-1.0, 2.0, -3.0], "m")], {}),
    "fabs": ("fabs", [([-1.0, 2.0, -3.0], "m")], {}),
    "floor": ("floor", [([1.7, -1.2], "m")], {}),
    "ceil": ("ceil", [([1.2, -1.7], "m")], {}),
    "trunc": ("trunc", [([1.7, -1.2], "m")], {}),
    "rint": ("rint", [([1.5, 2.5], "m")], {}),
    "round": ("round", [([1.4, 2.6], "m")], {}),
    "conjugate": ("conjugate", [(P, "m")], {}),
    "real": ("real", [([1.0 + 2.0j, 3.0 - 1.0j], "m")], {}),
    "imag": ("imag", [([1.0 + 2.0j, 3.0 - 1.0j], "m")], {}),
    # plain out
    "sign": ("sign", [([-2.0, 0.0, 3.0], "m")], {}),
    "isfinite": ("isfinite", [([1.0, np.inf, np.nan], "m")], {}),
    "isnan": ("isnan", [([1.0, np.nan, 3.0], "m")], {}),
    "isinf": ("isinf", [([1.0, np.inf, 3.0], "m")], {}),
    # dimensionless in / out
    "exp": ("exp", [(R, "1")], {}),
    "exp_percent": ("exp", [([50.0], "%")], {}),
    "exp2": ("exp2", [(R, "1")], {}),
    "expm1": ("expm1", [(R, "1")], {}),
    "log": ("log", [(P, "1")], {}),
    "log2": ("log2", [(P, "1")], {}),
    "log10": ("log10", [(P, "1")], {}),
    "log1p": ("log1p", [(R, "1")], {}),
    "sinh": ("sinh", [(R, "1")], {}),
    "cosh": ("cosh", [(R, "1")], {}),
    "tanh": ("tanh", [(R, "1")], {}),
    "arcsinh": ("arcsinh", [(R, "1")], {}),
    "arccosh": ("arccosh", [(P, "1")], {}),
    "arctanh": ("arctanh", [(R, "1")], {}),
    "logaddexp": ("logaddexp", [(R, "1"), (P, "1")], {}),
    # angle in
    "sin_rad": ("sin", [([0.0, np.pi / 2], "rad")], {}),
    "cos_deg": ("cos", [([0.0, 180.0], "deg")], {}),
    "tan_rad": ("tan", [(R, "rad")], {}),
    # inverse trig
    "arcsin": ("arcsin", [(R, "1")], {}),
    "arccos": ("arccos", [(R, "1")], {}),
    "arctan": ("arctan", [(P, "1")], {}),
    "arctan2": ("arctan2", [(P, "m"), (Q2, "m")], {}),
    # angle conversion (curated overrides in duq.jax.numpy)
    "deg2rad": ("deg2rad", [([90.0, 180.0], "deg")], {}),
    "radians": ("radians", [([90.0], "deg")], {}),
    "rad2deg": ("rad2deg", [([np.pi], "rad")], {}),
    "degrees": ("degrees", [([np.pi], "rad")], {}),
    # joining / selection
    "concatenate": ("concatenate", [([(P, "m"), ([100.0], "cm")], "SEQ")], {}),
    "stack": ("stack", [([(P, "m"), (Q2, "m")], "SEQ")], {}),
    "where": ("where", [(COND, None), (P, "m"), ([100.0, 200.0, 300.0], "cm")], {}),
    "clip": ("clip", [(P, "m"), (150.0, "cm"), (2.5, "m")], {}),
    "take": ("take", [(P, "m"), ([0, 2], None)], {}),
    # reductions
    "sum": ("sum", [(P, "m")], {}),
    "nansum": ("nansum", [([1.0, np.nan, 3.0], "m")], {}),
    "mean": ("mean", [(P, "m")], {}),
    "nanmean": ("nanmean", [([1.0, np.nan, 3.0], "m")], {}),
    "std": ("std", [(P, "m")], {}),
    "var": ("var", [(P, "m")], {}),
    "min": ("min", [(P, "m")], {}),
    "max": ("max", [(P, "m")], {}),
    "ptp": ("ptp", [(P, "m")], {}),
    "median": ("median", [(P, "m")], {}),
    "quantile": ("quantile", [(P, "m"), (0.5, None)], {}),
    "cumsum": ("cumsum", [(P, "m")], {}),
    "argmin": ("argmin", [(Q2, "m")], {}),
    "argmax": ("argmax", [(Q2, "m")], {}),
    # sorting / search
    "sort": ("sort", [([3.0, 1.0, 2.0], "m")], {}),
    "argsort": ("argsort", [([3.0, 1.0, 2.0], "m")], {}),
    "searchsorted": ("searchsorted", [(P, "m"), (250.0, "cm")], {}),
    # linalg-ish products
    "dot": ("dot", [(P, "m"), (Q2, "s")], {}),
    "inner": ("inner", [(P, "m"), (Q2, "s")], {}),
    "outer": ("outer", [(P, "m"), (Q2, "s")], {}),
    "matmul": ("matmul", [(P, "m"), (Q2, "s")], {}),
    "einsum": ("einsum", [("i,i->", None), (P, "m"), (Q2, "s")], {}),
    # shape / layout
    "reshape": ("reshape", [(P, "m"), ((3, 1), None)], {}),
    "ravel": ("ravel", [(P, "m")], {}),
    "transpose": ("transpose", [(P, "m")], {}),
    "squeeze": ("squeeze", [([[1.0], [2.0]], "m")], {}),
    "flip": ("flip", [(P, "m")], {}),
    "roll": ("roll", [(P, "m"), (1, None)], {}),
    "repeat": ("repeat", [(P, "m"), (2, None)], {}),
    "tile": ("tile", [(P, "m"), (2, None)], {}),
    "broadcast_to": ("broadcast_to", [(P, "m"), ((2, 3), None)], {}),
    "expand_dims": ("expand_dims", [(P, "m"), (0, None)], {}),
    "atleast_1d": ("atleast_1d", [(P, "m")], {}),
    "atleast_2d": ("atleast_2d", [(P, "m")], {}),
    "diff": ("diff", [(Q2, "m")], {}),
    "pad": ("pad", [(P, "m"), (1, None)], {}),
    "diagonal": ("diagonal", [([[1.0, 2.0], [3.0, 4.0]], "m")], {}),
    "trace_fn": ("trace", [([[1.0, 2.0], [3.0, 4.0]], "m")], {}),
    "gradient": ("gradient", [(Q2, "m")], {}),
}


def _build_numpy(values: object, unit: str | None) -> object:
    if unit == "SEQ":
        return [_build_numpy(v, u) for v, u in values]
    if unit is None:
        return np.asarray(values) if isinstance(values, list) else values
    return duq.Quantity(np.asarray(values), unit)


def _build_jax(values: object, unit: str | None) -> object:
    if unit == "SEQ":
        return [_build_jax(v, u) for v, u in values]
    if unit is None:
        return jnp.asarray(values) if isinstance(values, list) else values
    return duq.jax.Quantity(values, unit)


@pytest.mark.parametrize("name", list(_CASES))
def test_conformance_with_numpy_layer(name: str) -> None:
    fname, arg_specs, kwargs = _CASES[name]
    np_args = [_build_numpy(v, u) for v, u in arg_specs]
    jax_args = [_build_jax(v, u) for v, u in arg_specs]
    np_result = getattr(np, fname)(*np_args, **kwargs)
    jax_result = getattr(djnp, fname)(*jax_args, **kwargs)

    if isinstance(np_result, list):  # np.gradient returns a list for >1 axis
        np_result = np_result[0] if len(np_result) == 1 else np_result
    if isinstance(np_result, duq.Quantity):
        assert isinstance(jax_result, duq.jax.Quantity), f"{name}: jax result lost its unit"
        assert jax_result.unit == np_result.unit, (
            f"{name}: unit mismatch {jax_result.unit} != {np_result.unit}"
        )
        np.testing.assert_allclose(
            np.asarray(jax_result.value), np.asarray(np_result.value), err_msg=name
        )
    else:
        assert not isinstance(jax_result, duq.jax.Quantity), (
            f"{name}: jax result must be plain (numpy layer returned plain)"
        )
        np.testing.assert_allclose(np.asarray(jax_result), np.asarray(np_result), err_msg=name)


def test_reduce_prod_unit_power() -> None:
    # The JAX layer intentionally exceeds the NumPy layer here: reduction
    # shapes are static under tracing, so prod's output unit is well-defined.
    result = djnp.prod(duq.jax.Quantity([2.0, 3.0, 4.0], "m"))
    assert result.unit == duq.unit("m^3")
    np.testing.assert_allclose(float(result.value), 24.0)
    matrix = djnp.prod(
        duq.jax.Quantity([[2.0, 3.0], [4.0, 5.0]], "s"),
        axis=0,
    )
    assert matrix.unit == duq.unit("s^2")


def test_var_of_2d_along_axis_units() -> None:
    a = duq.jax.Quantity([[1.0, 2.0], [3.0, 5.0]], "m")
    result = djnp.var(a, axis=0)
    assert result.unit == duq.unit("m^2")
    np_ref = np.var(np.array([[1.0, 2.0], [3.0, 5.0]]), axis=0)
    np.testing.assert_allclose(np.asarray(result.value), np_ref)
