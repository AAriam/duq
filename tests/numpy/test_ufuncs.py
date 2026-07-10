"""Table-driven conformance tests for every covered ufunc.

Each case computes the duq result and the raw-NumPy reference on the *stripped*
magnitudes, then asserts both equal magnitudes and the expected output unit
(or a plain, unit-free ndarray where that is the rule).
"""

# NumPy's type stubs cannot describe a duck array, so passing a duq.Quantity
# to np.<func>/np.<ufunc> trips call-overload/arg-type/attr-defined; these are
# false positives for the very dispatch these conformance tests exercise.
# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, dict-item"

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import duq
from duq import Quantity

P = np.array([1.0, 2.0, 3.0])
Q = np.array([2.0, 5.0, 7.0])
R = np.array([0.25, 0.5, 0.75])  # inside (-1, 1) for the inverse trig domain


def qm(values: np.ndarray = P) -> Quantity:
    return Quantity(values, "m")


def qs(values: np.ndarray = Q) -> Quantity:
    return Quantity(values, "s")


# id -> (duq result, expected raw magnitude, expected unit str or None-for-plain)
Case = tuple[object, np.ndarray, str | None]
_CASES: dict[str, Callable[[], Case]] = {
    # multiplicative
    "multiply": lambda: (np.multiply(qm(), qs()), P * Q, "m*s"),
    "matmul": lambda: (np.matmul(qm(), qs()), P @ Q, "m*s"),
    "divide": lambda: (np.divide(qm(), qs()), P / Q, "m/s"),
    "true_divide": lambda: (np.true_divide(qm(), qs()), P / Q, "m/s"),
    "floor_divide": lambda: (np.floor_divide(qm(Q), qs(P)), Q // P, "m/s"),
    # additive / same-dimension (right operand converted to the left unit)
    "add": lambda: (np.add(qm(), qm(Q)), P + Q, "m"),
    "add_convert": lambda: (np.add(qm(), Quantity(Q * 100, "cm")), P + Q, "m"),
    "subtract": lambda: (np.subtract(qm(Q), qm(P)), Q - P, "m"),
    "hypot": lambda: (np.hypot(qm(np.array([3.0])), qm(np.array([4.0]))), np.array([5.0]), "m"),
    "maximum": lambda: (np.maximum(qm(), qm(Q)), np.maximum(P, Q), "m"),
    "minimum": lambda: (np.minimum(qm(), qm(Q)), np.minimum(P, Q), "m"),
    "fmax": lambda: (np.fmax(qm(), qm(Q)), np.fmax(P, Q), "m"),
    "fmin": lambda: (np.fmin(qm(), qm(Q)), np.fmin(P, Q), "m"),
    "remainder": lambda: (np.remainder(qm(Q), qm(P)), np.remainder(Q, P), "m"),
    "mod": lambda: (np.mod(qm(Q), qm(P)), np.mod(Q, P), "m"),
    "fmod": lambda: (np.fmod(qm(Q), qm(P)), np.fmod(Q, P), "m"),
    "nextafter": lambda: (np.nextafter(qm(), qm(Q)), np.nextafter(P, Q), "m"),
    "copysign": lambda: (np.copysign(qm(), qm(-Q)), np.copysign(P, -Q), "m"),
    # powers
    "power": lambda: (np.power(qm(), 2), P**2, "m^2"),
    "float_power": lambda: (np.float_power(qm(), 2), P**2.0, "m^2"),
    "sqrt": lambda: (np.sqrt(Quantity(P**2, "m^2")), P, "m"),
    "cbrt": lambda: (np.cbrt(Quantity(P**3, "m^3")), P, "m"),
    "square": lambda: (np.square(qm()), P**2, "m^2"),
    "reciprocal": lambda: (np.reciprocal(qm()), 1 / P, "m^-1"),
    # unary preserving
    "negative": lambda: (np.negative(qm()), -P, "m"),
    "positive": lambda: (np.positive(qm()), P, "m"),
    "absolute": lambda: (np.absolute(qm(-P)), P, "m"),
    "fabs": lambda: (np.fabs(qm(-P)), P, "m"),
    "conjugate": lambda: (np.conjugate(qm()), P, "m"),
    "floor": lambda: (np.floor(qm(np.array([1.7]))), np.array([1.0]), "m"),
    "ceil": lambda: (np.ceil(qm(np.array([1.2]))), np.array([2.0]), "m"),
    "trunc": lambda: (np.trunc(qm(np.array([1.7]))), np.array([1.0]), "m"),
    "rint": lambda: (np.rint(qm(np.array([1.5, 2.5]))), np.array([2.0, 2.0]), "m"),
    # plain (unit-free) unary
    "sign": lambda: (np.sign(qm(np.array([-2.0, 0.0, 3.0]))), np.array([-1.0, 0.0, 1.0]), None),
    "isfinite": lambda: (np.isfinite(qm()), np.array([True, True, True]), None),
    "isnan": lambda: (np.isnan(qm()), np.array([False, False, False]), None),
    "isinf": lambda: (np.isinf(qm()), np.array([False, False, False]), None),
    "signbit": lambda: (np.signbit(qm(-P)), np.array([True, True, True]), None),
    # dimensionless-in, dimensionless-out
    "exp": lambda: (np.exp(Quantity(R, "1")), np.exp(R), "1"),
    "exp2": lambda: (np.exp2(Quantity(R, "1")), np.exp2(R), "1"),
    "expm1": lambda: (np.expm1(Quantity(R, "1")), np.expm1(R), "1"),
    "log": lambda: (np.log(Quantity(P, "1")), np.log(P), "1"),
    "log2": lambda: (np.log2(Quantity(P, "1")), np.log2(P), "1"),
    "log10": lambda: (np.log10(Quantity(P, "1")), np.log10(P), "1"),
    "log1p": lambda: (np.log1p(Quantity(R, "1")), np.log1p(R), "1"),
    "sinh": lambda: (np.sinh(Quantity(R, "1")), np.sinh(R), "1"),
    "cosh": lambda: (np.cosh(Quantity(R, "1")), np.cosh(R), "1"),
    "tanh": lambda: (np.tanh(Quantity(R, "1")), np.tanh(R), "1"),
    "arcsinh": lambda: (np.arcsinh(Quantity(R, "1")), np.arcsinh(R), "1"),
    "arccosh": lambda: (np.arccosh(Quantity(P, "1")), np.arccosh(P), "1"),
    "arctanh": lambda: (np.arctanh(Quantity(R, "1")), np.arctanh(R), "1"),
    "logaddexp": lambda: (
        np.logaddexp(Quantity(R, "1"), Quantity(P, "1")),
        np.logaddexp(R, P),
        "1",
    ),
    "logaddexp2": lambda: (
        np.logaddexp2(Quantity(R, "1"), Quantity(P, "1")),
        np.logaddexp2(R, P),
        "1",
    ),
    # percent is a scaled dimensionless unit: 50 % -> 0.5 before exp
    "exp_percent": lambda: (np.exp(Quantity(np.array([50.0]), "%")), np.exp(np.array([0.5])), "1"),
    # angle-in (radian or degree), dimensionless out
    "sin_rad": lambda: (
        np.sin(Quantity(np.array([0.0, np.pi / 2]), "rad")),
        np.sin(np.array([0.0, np.pi / 2])),
        "1",
    ),
    "cos_deg": lambda: (
        np.cos(Quantity(np.array([0.0, 180.0]), "deg")),
        np.array([1.0, -1.0]),
        "1",
    ),
    "tan_rad": lambda: (np.tan(Quantity(R, "rad")), np.tan(R), "1"),
    # inverse trig -> radian-labelled
    "arcsin": lambda: (np.arcsin(Quantity(R, "1")), np.arcsin(R), "rad"),
    "arccos": lambda: (np.arccos(Quantity(R, "1")), np.arccos(R), "rad"),
    "arctan": lambda: (np.arctan(Quantity(P, "1")), np.arctan(P), "rad"),
    "arctan2": lambda: (np.arctan2(qm(), qm(Q)), np.arctan2(P, Q), "rad"),
    # angle conversion ufuncs
    "deg2rad": lambda: (np.deg2rad(Quantity(np.array([90.0]), "deg")), np.deg2rad([90.0]), "rad"),
    "radians": lambda: (np.radians(Quantity(np.array([90.0]), "deg")), np.radians([90.0]), "rad"),
    "rad2deg": lambda: (
        np.rad2deg(Quantity(np.array([np.pi]), "rad")),
        np.rad2deg([np.pi]),
        "deg",
    ),
    "degrees": lambda: (
        np.degrees(Quantity(np.array([np.pi]), "rad")),
        np.degrees([np.pi]),
        "deg",
    ),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_ufunc_conformance(name: str) -> None:
    result, expected, unit = _CASES[name]()
    if unit is None:
        assert not isinstance(result, Quantity), f"{name} must return a plain ndarray"
        np.testing.assert_allclose(np.asarray(result), expected)
    else:
        assert isinstance(result, Quantity), f"{name} must return a Quantity"
        assert result.unit == duq.unit(unit), f"{name}: {result.unit} != {unit}"
        np.testing.assert_allclose(np.asarray(result.value), expected)


_COMPARISONS = {
    "equal": (np.equal, lambda a, b: a == b),
    "not_equal": (np.not_equal, lambda a, b: a != b),
    "less": (np.less, lambda a, b: a < b),
    "less_equal": (np.less_equal, lambda a, b: a <= b),
    "greater": (np.greater, lambda a, b: a > b),
    "greater_equal": (np.greater_equal, lambda a, b: a >= b),
}


@pytest.mark.parametrize("name", list(_COMPARISONS))
def test_comparison_ufuncs_return_plain_bool(name: str) -> None:
    ufunc, ref = _COMPARISONS[name]
    left = Quantity(P, "m")
    right = Quantity(Q * 100, "cm")  # different unit, same dimension -> converted
    result = ufunc(left, right)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, ref(P, Q))


def test_comparison_reflected_operator_matches_ufunc() -> None:
    left = Quantity(P, "m")
    right = Quantity(Q, "m")
    np.testing.assert_array_equal(left < right, np.less(left, right))
    np.testing.assert_array_equal(left == right, np.equal(left, right))


def test_reduce_and_accumulate_preserve_units() -> None:
    q = Quantity(P, "m")
    assert np.add.reduce(q).unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray(np.add.reduce(q).value), P.sum())
    assert np.maximum.reduce(q).unit == duq.unit("m")
    assert np.minimum.reduce(q).unit == duq.unit("m")
    acc = np.add.accumulate(q)
    assert acc.unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray(acc.value), np.add.accumulate(P))


def test_power_scalar_and_0d_exponents() -> None:
    q = Quantity(P, "m")
    for exponent in (2, 2.0, np.array(2.0), np.float64(2.0), Quantity(np.array(2.0), "1")):
        result = np.power(q, exponent)
        assert isinstance(result, Quantity), repr(exponent)
        assert result.unit == duq.unit("m^2"), repr(exponent)
        np.testing.assert_allclose(np.asarray(result.value), P**2)


def test_power_accepts_uniform_array_exponent() -> None:
    q = Quantity(P, "m")
    result = np.power(q, np.array([2.0, 2.0, 2.0]))
    assert result.unit == duq.unit("m^2")
    np.testing.assert_allclose(np.asarray(result.value), P**2)


def test_scalar_boxing_from_0d_and_numpy_scalar() -> None:
    q = Quantity(np.float64(2.0), "m")
    assert isinstance(np.negative(q), Quantity)
    zero_d = Quantity(np.array(3.0), "m")
    assert np.sqrt(Quantity(np.array(4.0), "m^2")).unit == duq.unit("m")
    assert zero_d.ndim == 0
