"""Fail-loud guards: uncovered primitives, dimension mismatches, affine policy."""

# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, operator"

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax

import duq
import duq.jax
import duq.jax.numpy as djnp
from duq.jax import Quantity

V = [1.0, 2.0, 3.0]


def test_uncovered_primitive_fails_loud_naming_it() -> None:
    spd = Quantity(jnp.eye(2) + 1.0, "m")
    with pytest.raises(duq.UnsupportedOperationError, match="cholesky"):
        djnp.linalg.cholesky(spd)


def test_materialise_refuses() -> None:
    with pytest.raises(duq.UnsupportedOperationError, match="materialise"):
        Quantity(V, "m").materialise()


def test_incompatible_dimensions_raise() -> None:
    a, b = Quantity(V, "m"), Quantity(V, "s")
    with pytest.raises(duq.DimensionalityError):
        _ = a + b
    with pytest.raises(duq.DimensionalityError):
        djnp.add(a, b)
    with pytest.raises(duq.DimensionalityError):
        djnp.concatenate([a, b])
    with pytest.raises(duq.DimensionalityError):
        djnp.maximum(a, b)


def test_bare_operand_with_dimensional_array_rejected() -> None:
    a = Quantity(V, "m")
    with pytest.raises(duq.DimensionalityError):
        _ = a + 2.0
    with pytest.raises(duq.DimensionalityError):
        _ = a + jnp.asarray(V)
    with pytest.raises(duq.DimensionalityError):
        djnp.maximum(a, 1.0)


def test_dimensionless_functions_require_dimensionless() -> None:
    a = Quantity(V, "m")
    with pytest.raises(duq.DimensionalityError):
        djnp.exp(a)
    with pytest.raises(duq.DimensionalityError):
        djnp.sin(a)
    with pytest.raises(duq.DimensionalityError):
        djnp.arcsin(a)
    with pytest.raises(duq.DimensionalityError):
        djnp.deg2rad(a)


def test_ordering_against_bare_nonzero_raises_but_zero_passes() -> None:
    a = Quantity(V, "m")
    with pytest.raises(duq.DimensionalityError):
        _ = a < 1.0
    # 0/±inf/nan are scale-invariant: comparisons against them are meaningful
    # in any linear unit (jnp compositions rely on this).
    np.testing.assert_array_equal(np.asarray(a > 0.0), [True, True, True])
    np.testing.assert_array_equal(np.asarray(a < np.inf), [True, True, True])
    eq_zero = Quantity([0.0, 1.0], "m") == 0.0
    np.testing.assert_array_equal(np.asarray(eq_zero), [True, False])


def test_zero_comparison_on_affine_still_rejected() -> None:
    celsius = Quantity([0.0], "degC")
    with pytest.raises(duq.DimensionalityError):
        _ = celsius < 0.0


def test_traced_pow_exponent_on_dimensional_base_raises() -> None:
    def f(base: Quantity, exponent: jax.Array) -> Quantity:
        return base**exponent

    with pytest.raises(duq.UnsupportedOperationError, match="traced"):
        duq.jax.jit(f)(Quantity(2.0, "m"), jnp.asarray(2.0))
    # ... but a scale-1 dimensionless base accepts a traced exponent
    result = duq.jax.jit(f)(Quantity(2.0, "1"), jnp.asarray(3.0))
    np.testing.assert_allclose(float(result.value), 8.0)
    assert result.unit == duq.unit("1")


def test_nonuniform_array_exponent_rejected() -> None:
    with pytest.raises(duq.UnsupportedOperationError, match="exponent"):
        djnp.power(Quantity(V, "m"), jnp.array([1.0, 2.0, 3.0]))
    uniform = djnp.power(Quantity(V, "m"), jnp.array([2.0, 2.0, 2.0]))
    assert uniform.unit == duq.unit("m^2")


def test_quantity_exponent_must_be_dimensionless() -> None:
    with pytest.raises(duq.DimensionalityError):
        djnp.power(Quantity(V, "m"), Quantity(2.0, "s"))
    result = djnp.power(Quantity(V, "m"), Quantity(200.0, "%"))
    assert result.unit == duq.unit("m^2")


def test_cumprod_rejected_for_dimensional() -> None:
    with pytest.raises(duq.UnsupportedOperationError, match="cumprod"):
        djnp.cumprod(Quantity(V, "m"))
    ok = djnp.cumprod(Quantity(V, "1"))
    np.testing.assert_allclose(np.asarray(ok.value), [1.0, 2.0, 6.0])


def test_affine_policy_matches_numpy_layer() -> None:
    celsius = Quantity([0.0, 100.0], "degC")
    with pytest.raises(duq.AffineUnitError):
        _ = celsius + celsius
    with pytest.raises(duq.AffineUnitError):
        djnp.multiply(celsius, 2.0)
    with pytest.raises(duq.AffineUnitError):
        _ = -celsius
    with pytest.raises(duq.AffineUnitError):
        abs(celsius)
    diff = celsius - Quantity([0.0, 50.0], "degC")
    assert diff.unit == duq.unit("K")
    np.testing.assert_allclose(np.asarray(diff.value), [0.0, 50.0])
    with pytest.raises(duq.AffineUnitError):
        djnp.dot(celsius, celsius)


def test_affine_plus_delta_matches_core() -> None:
    celsius = Quantity([20.0], "degC")
    nudged = celsius + Quantity([5.0], "K")
    assert nudged.unit == duq.unit("degC")
    np.testing.assert_allclose(np.asarray(nudged.value), [25.0])
    core = duq.Quantity(20.0, "degC") + duq.Quantity(5.0, "K")
    np.testing.assert_allclose(float(core.value), 25.0)


def test_relative_plus_affine_raises() -> None:
    with pytest.raises(duq.AffineUnitError):
        _ = Quantity([5.0], "K") + Quantity([20.0], "degC")


def test_scatter_add_with_incompatible_updates_raises() -> None:
    a = Quantity(V, "m")
    with pytest.raises(duq.DimensionalityError):
        a.at[0].add(Quantity(1.0, "s"))


def test_quaxify_grad_scalar_check_sharp_edge_documented() -> None:
    # quax's known sharp edge (quax#5): quaxify(jax.grad(f)) breaks when f
    # returns a Quantity built from internally-created quantities, because the
    # nested trace levels split the unit bookkeeping.  duq.jax.grad is the
    # supported route; this test pins the failure mode so a future quax fix
    # is noticed.
    def kinetic(v: object) -> object:
        return 0.5 * Quantity(2.0, "kg") * v * v

    with pytest.raises(TypeError, match="scalar"):
        quax.quaxify(jax.grad(kinetic))(Quantity(3.0, "m/s"))
