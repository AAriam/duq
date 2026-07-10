"""Targeted tests exercising the less-common array branches."""

# NumPy's type stubs cannot describe a duck array, so passing a duq.Quantity
# to np.<func>/np.<ufunc> trips call-overload/arg-type/attr-defined; these are
# false positives for the very dispatch these conformance tests exercise.
# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, dict-item"

from __future__ import annotations

import numpy as np
import pytest

import duq
from duq import Quantity

V = np.array([1.0, 2.0, 3.0])


def test_all_array_comparison_operators() -> None:
    a = Quantity(V, "m")
    b = Quantity(V * 100, "cm")  # equal after conversion
    np.testing.assert_array_equal(a == b, np.equal(V, V))
    np.testing.assert_array_equal(a != b, np.not_equal(V, V))
    np.testing.assert_array_equal(a <= b, V <= V)
    np.testing.assert_array_equal(a >= b, V >= V)
    np.testing.assert_array_equal(a > Quantity(V - 1, "m"), V > V - 1)
    np.testing.assert_array_equal(a < Quantity(V + 1, "m"), V < V + 1)
    # a bare array on either side still routes through the array comparison path
    np.testing.assert_array_equal(Quantity(V, "1") == V, np.equal(V, V))
    np.testing.assert_array_equal(V == Quantity(V, "1"), np.equal(V, V))  # noqa: SIM300


def test_reflected_matmul_direct() -> None:
    q = Quantity(np.eye(3), "s")
    result = q.__rmatmul__(V)
    assert isinstance(result, Quantity)
    assert result.unit == duq.unit("s")
    np.testing.assert_allclose(np.asarray(result.value), V @ np.eye(3))


def test_where_single_arg_and_all_plain() -> None:
    mask = Quantity(np.array([0.0, 1.0, 0.0]), "1")
    (indices,) = np.where(mask)
    np.testing.assert_array_equal(indices, np.where(np.array([0.0, 1.0, 0.0]))[0])
    # cond is a Quantity but the branches are plain -> plain result
    result = np.where(mask, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert not isinstance(result, Quantity)


def test_average_returned_tuple() -> None:
    q = Quantity(V, "m")
    avg, weight_sum = np.average(q, weights=Quantity(V, "s"), returned=True)
    assert isinstance(avg, Quantity)
    assert avg.unit == duq.unit("m")
    assert not isinstance(weight_sum, Quantity)


def test_full_like_rejects_bare_fill_for_dimensional() -> None:
    with pytest.raises(duq.DimensionalityError, match="full_like"):
        np.full_like(Quantity(V, "m"), 2.0)
    # a dimensionless template accepts a bare fill value
    filled = np.full_like(Quantity(V, "1"), 2.0)
    assert filled.unit == duq.unit("1")


def test_same_dim_ufunc_rejects_bare_array_for_dimensional() -> None:
    with pytest.raises(duq.DimensionalityError):
        np.maximum(Quantity(V, "m"), np.array([1.0, 2.0, 3.0]))


def test_angle_convert_requires_dimensionless() -> None:
    with pytest.raises(duq.DimensionalityError):
        np.deg2rad(Quantity(V, "m"))
    # a plain array passes through unchanged (treated as raw degrees)
    np.testing.assert_allclose(np.asarray(np.rad2deg(Quantity(V, "1")).value), np.rad2deg(V))


def test_rsub_and_scalar_paths_still_work_for_scalars() -> None:
    # keep the scalar reflected-subtraction path exercised alongside arrays
    assert (Quantity(1.0, "1").__rsub__(3)).value == 2.0
    assert (5 - Quantity(2.0, "1")).value == 3.0


def test_dtype_of_scalar_quantity() -> None:
    assert Quantity(2.0, "m").dtype == np.dtype("float64")
