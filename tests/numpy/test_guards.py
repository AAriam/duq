"""Fail-loud guards: silent-strip holes, unsupported ops, and the affine policy."""

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


def test_array_and_asarray_never_strip_units() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="ustrip"):
        np.array(q)
    with pytest.raises(duq.UnsupportedOperationError, match="ustrip"):
        np.asarray(q)
    with pytest.raises(duq.UnsupportedOperationError):
        q.__array__()


def test_scalar_coercions_fail_loud_for_dimensional() -> None:
    q = Quantity(2.0, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="ustrip"):
        float(q)
    with pytest.raises(duq.UnsupportedOperationError):
        int(q)
    with pytest.raises(duq.UnsupportedOperationError):
        complex(q)
    with pytest.raises(duq.UnsupportedOperationError):
        bool(q)
    # Dimensionless quantities coerce, applying the unit scale (pint-consistent).
    assert float(Quantity(50.0, "%")) == 0.5


def test_unsupported_ufunc_names_the_function() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="heaviside"):
        np.heaviside(q, 0.5)


def test_unsupported_function_names_the_function() -> None:
    q = Quantity(np.eye(2), "m")
    with pytest.raises(duq.UnsupportedOperationError, match="inv"):
        np.linalg.inv(q)
    with pytest.raises(duq.UnsupportedOperationError):
        np.linalg.det(q)


def test_out_argument_is_rejected() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="out="):
        np.add(q, q, out=np.empty(3))


def test_multiply_reduce_is_rejected_with_guidance() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="prod"):
        np.multiply.reduce(q)


def test_unsupported_ufunc_method_is_rejected() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError):
        np.add.outer(q, q)


def test_pad_unsupported_mode_is_rejected() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="mode"):
        np.pad(q, 1, mode="linear_ramp")


def test_power_rejects_nonuniform_array_exponent() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="exponent"):
        np.power(q, np.array([1.0, 2.0, 3.0]))


def test_allclose_requires_quantity_atol_for_dimensional() -> None:
    a = Quantity(V, "m")
    b = Quantity(V, "m")
    # The default atol=1e-8 would silently assume the base-unit scale (wrong
    # for e.g. km), so a dimensional comparison demands an explicit Quantity atol.
    with pytest.raises(duq.DimensionalityError, match="atol"):
        np.allclose(a, b)
    with pytest.raises(duq.DimensionalityError, match="atol"):
        np.allclose(a, b, atol=1e-9)  # bare float atol rejected too
    with pytest.raises(duq.DimensionalityError, match="atol"):
        np.isclose(a, b)
    with pytest.raises(duq.DimensionalityError, match="atol"):
        np.isclose(a, b, atol=1e-9)
    # A Quantity atol converts and works, whatever its (compatible) unit.
    km = Quantity(V / 1000.0, "km")
    assert np.allclose(a, km, atol=Quantity(1e-6, "mm"))
    assert bool(np.isclose(a, km, atol=Quantity(1e-6, "mm")).all())
    with pytest.raises(duq.DimensionalityError):
        np.allclose(a, b, atol=Quantity(1e-9, "s"))  # wrong-dimension atol


def test_incompatible_dimensions_raise() -> None:
    a = Quantity(V, "m")
    b = Quantity(V, "s")
    with pytest.raises(duq.DimensionalityError):
        np.add(a, b)
    with pytest.raises(duq.DimensionalityError):
        np.concatenate([a, b])
    with pytest.raises(duq.DimensionalityError):
        _ = a < b


def test_equal_on_incompatible_dims_is_all_false() -> None:
    a = Quantity(V, "m")
    b = Quantity(V, "s")
    eq = np.equal(a, b)
    assert isinstance(eq, np.ndarray)
    assert not eq.any()
    ne = np.not_equal(a, b)
    assert ne.all()


def test_bare_number_with_dimensional_array_rejected() -> None:
    a = Quantity(V, "m")
    with pytest.raises(duq.DimensionalityError):
        np.add(a, 2.0)
    with pytest.raises(duq.DimensionalityError):
        _ = a + np.array([1.0, 2.0, 3.0])


def test_celsius_array_converts_but_arithmetic_follows_affine_policy() -> None:
    celsius = Quantity(np.array([0.0, 100.0]), "degC")
    np.testing.assert_allclose(np.asarray(celsius.to("K").value), [273.15, 373.15])
    with pytest.raises(duq.AffineUnitError):
        celsius + celsius
    with pytest.raises(duq.AffineUnitError):
        np.multiply(celsius, 2.0)
    with pytest.raises(duq.AffineUnitError):
        np.negative(celsius)
    # °C - °C is a difference in kelvin (per the scalar affine policy).
    diff = celsius - Quantity(np.array([0.0, 50.0]), "degC")
    assert diff.unit == duq.unit("K")


def test_dimensionless_function_requires_dimensionless() -> None:
    with pytest.raises(duq.DimensionalityError):
        np.exp(Quantity(V, "m"))
    with pytest.raises(duq.DimensionalityError):
        np.sin(Quantity(V, "m"))


def test_compact_and_allclose_reject_array_magnitudes() -> None:
    q = Quantity(V, "m")
    with pytest.raises(duq.UnsupportedOperationError):
        q.compact()
    with pytest.raises(duq.UnsupportedOperationError):
        q.allclose(q)
