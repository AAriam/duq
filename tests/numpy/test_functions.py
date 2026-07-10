"""Table-driven conformance tests for the ``__array_function__`` coverage."""

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

V = np.array([1.0, 2.0, 3.0, 4.0])
W = np.array([10.0, 20.0, 30.0, 40.0])
M2 = np.arange(6.0).reshape(2, 3)


def qv(values: np.ndarray = V, unit: str = "m") -> Quantity:
    return Quantity(values, unit)


# name -> (duq result, expected raw magnitude, expected unit str)
Case = tuple[object, object, str]
_PRESERVE: dict[str, Callable[[], Case]] = {
    "reshape": lambda: (np.reshape(qv(), (2, 2)), V.reshape(2, 2), "m"),
    "ravel": lambda: (np.ravel(qv(M2)), M2.ravel(), "m"),
    "transpose": lambda: (np.transpose(qv(M2)), M2.T, "m"),
    "swapaxes": lambda: (np.swapaxes(qv(M2), 0, 1), np.swapaxes(M2, 0, 1), "m"),
    "moveaxis": lambda: (np.moveaxis(qv(M2), 0, 1), np.moveaxis(M2, 0, 1), "m"),
    "squeeze": lambda: (np.squeeze(qv(V.reshape(1, 4))), V, "m"),
    "expand_dims": lambda: (np.expand_dims(qv(), 0), V.reshape(1, 4), "m"),
    "broadcast_to": lambda: (np.broadcast_to(qv(), (2, 4)), np.broadcast_to(V, (2, 4)), "m"),
    "flip": lambda: (np.flip(qv()), np.flip(V), "m"),
    "fliplr": lambda: (np.fliplr(qv(M2)), np.fliplr(M2), "m"),
    "flipud": lambda: (np.flipud(qv(M2)), np.flipud(M2), "m"),
    "roll": lambda: (np.roll(qv(), 1), np.roll(V, 1), "m"),
    "rot90": lambda: (np.rot90(qv(M2)), np.rot90(M2), "m"),
    "tile": lambda: (np.tile(qv(), 2), np.tile(V, 2), "m"),
    "repeat": lambda: (np.repeat(qv(), 2), np.repeat(V, 2), "m"),
    "delete": lambda: (np.delete(qv(), 0), np.delete(V, 0), "m"),
    "take": lambda: (np.take(qv(), [0, 2]), np.take(V, [0, 2]), "m"),
    "diagonal": lambda: (np.diagonal(qv(M2)), np.diagonal(M2), "m"),
    "diag": lambda: (np.diag(qv()), np.diag(V), "m"),
    "real": lambda: (np.real(qv()), V, "m"),
    "imag": lambda: (np.imag(qv()), np.zeros_like(V), "m"),
    "copy": lambda: (np.copy(qv()), V, "m"),
    "sort": lambda: (np.sort(qv(np.array([3.0, 1.0, 2.0]))), np.array([1.0, 2.0, 3.0]), "m"),
    "cumsum": lambda: (np.cumsum(qv()), np.cumsum(V), "m"),
    "nancumsum": lambda: (np.nancumsum(qv()), np.nancumsum(V), "m"),
    "diff": lambda: (np.diff(qv()), np.diff(V), "m"),
    "ediff1d": lambda: (np.ediff1d(qv()), np.ediff1d(V), "m"),
    "sum": lambda: (np.sum(qv()), V.sum(), "m"),
    "nansum": lambda: (np.nansum(qv()), np.nansum(V), "m"),
    "mean": lambda: (np.mean(qv()), V.mean(), "m"),
    "nanmean": lambda: (np.nanmean(qv()), np.nanmean(V), "m"),
    "median": lambda: (np.median(qv()), np.median(V), "m"),
    "nanmedian": lambda: (np.nanmedian(qv()), np.nanmedian(V), "m"),
    "min": lambda: (np.min(qv()), V.min(), "m"),
    "max": lambda: (np.max(qv()), V.max(), "m"),
    "amin": lambda: (np.amin(qv()), V.min(), "m"),
    "amax": lambda: (np.amax(qv()), V.max(), "m"),
    "nanmin": lambda: (np.nanmin(qv()), np.nanmin(V), "m"),
    "nanmax": lambda: (np.nanmax(qv()), np.nanmax(V), "m"),
    "ptp": lambda: (np.ptp(qv()), np.ptp(V), "m"),
    "std": lambda: (np.std(qv()), V.std(), "m"),
    "nanstd": lambda: (np.nanstd(qv()), np.nanstd(V), "m"),
    "trace": lambda: (np.trace(qv(M2)), np.trace(M2), "m"),
    "around": lambda: (np.around(qv(np.array([1.24, 2.55])), 1), np.array([1.2, 2.6]), "m"),
    "round": lambda: (np.round(qv(np.array([1.4, 2.6]))), np.array([1.0, 3.0]), "m"),
    "quantile": lambda: (np.quantile(qv(), 0.5), np.quantile(V, 0.5), "m"),
    "percentile": lambda: (np.percentile(qv(), 50), np.percentile(V, 50), "m"),
    "norm": lambda: (np.linalg.norm(qv()), np.linalg.norm(V), "m"),
    "average": lambda: (np.average(qv()), np.average(V), "m"),
    "average_weighted": lambda: (
        np.average(qv(), weights=Quantity(W, "s")),
        np.average(V, weights=W),
        "m",
    ),
    "clip": lambda: (
        np.clip(qv(), Quantity(1.5, "m"), Quantity(3.5, "m")),
        np.clip(V, 1.5, 3.5),
        "m",
    ),
    "compress": lambda: (
        np.compress(np.array([True, False, True, False]), qv()),
        np.compress([True, False, True, False], V),
        "m",
    ),
    "append": lambda: (np.append(qv(), Quantity(np.array([500.0]), "cm")), np.append(V, 5.0), "m"),
    "insert": lambda: (np.insert(qv(), 0, Quantity(0.5, "m")), np.insert(V, 0, 0.5), "m"),
    "where": lambda: (
        np.where(V > 2, qv(), Quantity(W, "m")),
        np.where(V > 2, V, W),
        "m",
    ),
    "unique": lambda: (np.unique(qv(np.array([2.0, 1.0, 2.0]))), np.array([1.0, 2.0]), "m"),
    "pad": lambda: (np.pad(qv(), 1, constant_values=Quantity(0.0, "m")), np.pad(V, 1), "m"),
    "pad_edge": lambda: (np.pad(qv(), 1, mode="edge"), np.pad(V, 1, mode="edge"), "m"),
    "zeros_like": lambda: (np.zeros_like(qv()), np.zeros_like(V), "m"),
    "empty_like": lambda: (np.zeros_like(qv()) * 0, np.zeros_like(V), "m"),
    "ones_like": lambda: (np.ones_like(qv()), np.ones_like(V), "m"),
    "full_like_quantity": lambda: (
        np.full_like(qv(), Quantity(2.0, "m")),
        np.full_like(V, 2.0),
        "m",
    ),
}


@pytest.mark.parametrize("name", list(_PRESERVE))
def test_function_preserves_unit(name: str) -> None:
    result, expected, unit = _PRESERVE[name]()
    assert isinstance(result, Quantity), f"{name} must return a Quantity"
    assert result.unit == duq.unit(unit), f"{name}: {result.unit} != {unit}"
    np.testing.assert_allclose(np.asarray(result.value), expected)


def test_variance_squares_the_unit() -> None:
    assert np.var(qv()).unit == duq.unit("m^2")
    np.testing.assert_allclose(np.asarray(np.var(qv()).value), V.var())
    assert np.nanvar(qv()).unit == duq.unit("m^2")


def test_products_multiply_units() -> None:
    a, b = qv(V, "m"), qv(W, "s")
    assert np.dot(a, b).unit == duq.unit("m*s")
    np.testing.assert_allclose(np.asarray(np.dot(a, b).value), V @ W)
    assert np.inner(a, b).unit == duq.unit("m*s")
    assert np.vdot(a, b).unit == duq.unit("m*s")
    assert np.outer(a, b).unit == duq.unit("m*s")
    assert np.kron(a, b).unit == duq.unit("m*s")
    assert np.tensordot(a, b, axes=1).unit == duq.unit("m*s")
    cross = np.cross(qv(np.array([1.0, 0.0, 0.0]), "m"), qv(np.array([0.0, 1.0, 0.0]), "s"))
    assert cross.unit == duq.unit("m*s")


def test_einsum_multiplies_units() -> None:
    a, b = qv(V, "m"), qv(W, "s")
    result = np.einsum("i,i->", a, b)
    assert result.unit == duq.unit("m*s")
    np.testing.assert_allclose(np.asarray(result.value), np.einsum("i,i->", V, W))


@pytest.mark.parametrize(
    "func", [np.concatenate, np.stack, np.hstack, np.vstack, np.dstack, np.column_stack]
)
def test_joining_converts_to_first_unit(func: object) -> None:
    a = qv(V, "m")
    b = Quantity(W * 100, "cm")  # same dimension, different unit
    result = func([a, b])  # type: ignore[operator]
    assert isinstance(result, Quantity)
    assert result.unit == duq.unit("m")


def test_split_family_returns_list_of_quantities() -> None:
    q = qv(V, "m")
    parts = np.split(q, 2)
    assert all(isinstance(p, Quantity) and p.unit == duq.unit("m") for p in parts)
    assert all(isinstance(p, Quantity) for p in np.array_split(q, 3))


def test_atleast_and_broadcast_arrays() -> None:
    q = qv(V, "m")
    assert np.atleast_2d(q).unit == duq.unit("m")
    one, two = np.atleast_1d(q, Quantity(np.array([1.0]), "s"))
    assert one.unit == duq.unit("m")
    assert two.unit == duq.unit("s")
    b1, b2 = np.broadcast_arrays(qv(np.array([[1.0], [2.0]]), "m"), Quantity(V, "s"))
    assert b1.unit == duq.unit("m")
    assert b2.unit == duq.unit("s")


def test_search_and_index_functions_return_plain() -> None:
    q = qv(np.array([1.0, 3.0, 2.0]), "m")
    assert not isinstance(np.argmin(q), Quantity)
    assert not isinstance(np.argsort(q), Quantity)
    assert not isinstance(np.count_nonzero(q), Quantity)
    # searchsorted / digitize convert the query to the array unit
    sorted_q = qv(V, "m")
    idx = np.searchsorted(sorted_q, Quantity(250.0, "cm"))
    assert int(idx) == np.searchsorted(V, 2.5)
    bins = np.digitize(qv(np.array([2.5]), "m"), Quantity(V, "m"))
    np.testing.assert_array_equal(bins, np.digitize([2.5], V))


def test_interp_converts_x_preserves_fp_unit() -> None:
    x = Quantity(np.array([150.0]), "cm")  # 1.5 m
    xp = qv(V, "m")
    fp = Quantity(W, "s")
    result = np.interp(x, xp, fp)
    assert result.unit == duq.unit("s")
    np.testing.assert_allclose(np.asarray(result.value), np.interp([1.5], V, W))


def test_meshgrid_preserves_each_unit() -> None:
    gx, gy = np.meshgrid(qv(V, "m"), Quantity(W, "s"))
    assert gx.unit == duq.unit("m")
    assert gy.unit == duq.unit("s")


def test_histogram_returns_plain_counts_and_edge_quantity() -> None:
    counts, edges = np.histogram(qv(V, "m"), bins=Quantity(np.array([0.0, 2.0, 4.0]), "m"))
    assert not isinstance(counts, Quantity)
    assert isinstance(edges, Quantity)
    assert edges.unit == duq.unit("m")


def test_gradient_divides_units() -> None:
    f = Quantity(V**2, "m")
    x = Quantity(V, "s")
    grad = np.gradient(f, x)
    assert grad.unit == duq.unit("m/s")
    np.testing.assert_allclose(np.asarray(grad.value), np.gradient(V**2, V))
    # No spacing -> unit preserved.
    assert np.gradient(f).unit == duq.unit("m")


def test_trapezoid_multiplies_units() -> None:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 vs 1.26
    y = Quantity(V, "m")
    x = Quantity(W, "s")
    result = trapezoid(y, x=x)
    assert result.unit == duq.unit("m*s")
    result_dx = trapezoid(y, dx=Quantity(2.0, "s"))
    assert result_dx.unit == duq.unit("m*s")


def test_allclose_and_isclose_unit_aware() -> None:
    a = qv(V, "m")
    b = Quantity(V * 100, "cm")
    assert np.allclose(a, b, atol=Quantity(1e-9, "m"))
    close = np.isclose(a, b, atol=Quantity(1e-9, "m"))
    assert close.dtype == np.bool_
    assert bool(close.all())
    # dimensionless arrays may use the default tolerances
    assert np.allclose(Quantity(V, "1"), Quantity(V, "1"))


def test_array_equal_and_equiv() -> None:
    assert np.array_equal(qv(V, "m"), Quantity(V * 100, "cm"))
    assert not np.array_equal(qv(V, "m"), Quantity(V, "s"))  # incompatible -> False
    assert np.array_equiv(qv(V, "m"), Quantity(V * 100, "cm"))
