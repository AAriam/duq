"""Array construction, introspection, iteration, methods and the unit ergonomics."""

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
M2 = np.arange(6.0).reshape(2, 3)


def test_construct_stores_array_without_copy() -> None:
    q = Quantity(V, "m")
    assert q.is_array
    assert q.value is V  # stored as-is, no copy


def test_from_array_coerces_lists() -> None:
    q = Quantity.from_array([1.0, 2.0, 3.0], "m")
    assert q.is_array
    assert q.shape == (3,)
    # a bare list still cannot go through the plain constructor
    with pytest.raises(TypeError):
        Quantity([1.0, 2.0], "m")


def test_numpy_scalar_is_accepted() -> None:
    q = Quantity(np.float64(2.5), "m")
    assert q.value == np.float64(2.5)


def test_shape_ndim_size_dtype() -> None:
    q = Quantity(M2, "m")
    assert q.shape == (2, 3)
    assert q.ndim == 2
    assert q.size == 6
    assert q.dtype == M2.dtype
    # np.shape / np.ndim / np.size route through the properties
    assert np.shape(q) == (2, 3)
    assert np.ndim(q) == 2
    assert np.size(q) == 6


def test_scalar_quantity_shape_is_empty() -> None:
    q = Quantity(2.0, "m")
    assert q.shape == ()
    assert q.ndim == 0
    assert q.size == 1


def test_len_iter_getitem_yield_quantities() -> None:
    q = Quantity(V, "m")
    assert len(q) == 3
    elements = list(q)
    assert all(isinstance(e, Quantity) and e.unit == duq.unit("m") for e in elements)
    assert elements[0].value == 1.0
    assert q[1].value == 2.0
    assert q[1].unit == duq.unit("m")
    assert q[1:].shape == (2,)


def test_item_returns_scalar_quantity() -> None:
    q = Quantity(np.array([[5.0]]), "m")
    item = q.item()
    assert isinstance(item, Quantity)
    assert item.value == 5.0
    assert item.unit == duq.unit("m")


def test_len_and_iter_reject_scalar() -> None:
    q = Quantity(2.0, "m")
    with pytest.raises(TypeError):
        len(q)
    with pytest.raises(TypeError):
        next(iter(q))


def test_reduction_and_reshape_methods() -> None:
    q = Quantity(V, "m")
    assert q.sum().unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray(q.sum().value), V.sum())
    assert q.mean().unit == duq.unit("m")
    assert q.std().unit == duq.unit("m")
    assert q.var().unit == duq.unit("m^2")
    assert q.min().value == 1.0
    assert q.max().value == 3.0
    assert q.reshape(3, 1).shape == (3, 1)
    assert q.reshape(3, 1).unit == duq.unit("m")
    assert q.ravel().shape == (3,)
    assert q.astype(np.float32).dtype == np.float32
    assert Quantity(M2, "m").T.shape == (3, 2)


def test_matmul_operator_and_reflected() -> None:
    a = Quantity(V, "m")
    b = Quantity(V, "s")
    assert (a @ b).unit == duq.unit("m*s")
    np.testing.assert_allclose(np.asarray((a @ b).value), V @ V)
    # ndarray @ quantity defers to __rmatmul__
    result = V @ b
    assert result.unit == duq.unit("s")


def test_bool_of_array_quantity_matches_numpy() -> None:
    assert bool(Quantity(np.array([1.0]), "m"))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(Quantity(V, "m"))


@pytest.mark.parametrize(
    "make",
    [
        lambda u: np.linspace(0, 1, 5) * u,  # array * unit (numpy defers)
        lambda u: u * np.linspace(0, 1, 5),  # unit * array
    ],
)
def test_array_times_unit_makes_quantity(make: object) -> None:
    q = make(duq.units.m)  # type: ignore[operator]
    assert isinstance(q, Quantity)
    assert q.unit == duq.unit("m")
    assert q.shape == (5,)


def test_array_divided_by_unit_makes_quantity() -> None:
    q = np.arange(3.0) / duq.units.s
    assert isinstance(q, Quantity)
    assert q.unit == duq.unit("s^-1")


def test_scalar_number_times_unit_makes_quantity() -> None:
    assert isinstance(5 * duq.units.m, Quantity)
    assert (5 * duq.units.m).value == 5
    assert (duq.units.m * 5).value == 5
    assert (duq.units.m / 5).value == 0.2
    assert (10 / duq.units.s).unit == duq.unit("s^-1")


def test_unit_array_ufunc_is_none() -> None:
    assert duq.Unit.__array_ufunc__ is None


def test_mixed_operand_matrix() -> None:
    q = Quantity(V, "m")
    # Quantity (x) ndarray  -> the ndarray is treated as dimensionless, so only a
    # dimensionless quantity may combine additively; multiply always works.
    assert np.multiply(q, np.array([2.0, 2.0, 2.0])).unit == duq.unit("m")
    assert np.multiply(np.array([2.0, 2.0, 2.0]), q).unit == duq.unit("m")
    assert (q * 2).unit == duq.unit("m")
    # Quantity (x) Quantity with different-but-compatible units -> left convention
    total = np.add(q, Quantity(V * 100, "cm"))
    assert total.unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray(total.value), V + V)
