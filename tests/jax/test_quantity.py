"""duq.jax.Quantity: construction, conversion, operators and coercion."""

# JAX's stubs cannot describe quax Value operands; the deliberately dynamic
# dispatch these tests exercise trips the same false positives as the NumPy
# conformance suite.
# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, operator"

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import duq
import duq.jax
from duq.jax import Quantity

V = [1.0, 2.0, 3.0]


def q(values: object = None, unit: str = "m") -> Quantity:
    return Quantity(V if values is None else values, unit)


# -- construction ---------------------------------------------------------------


def test_construct_from_list_scalar_numpy_and_jax() -> None:
    assert q().shape == (3,)
    assert Quantity(2.0, "m").shape == ()
    assert Quantity(np.array(V), "m").shape == (3,)
    assert Quantity(jnp.array(V), "m").shape == (3,)
    assert Quantity(2, "m").dtype == jnp.asarray(2).dtype


def test_construct_fraction_and_decimal_coerce_to_float() -> None:
    assert float(Quantity(Fraction(1, 2), "1").value) == 0.5
    assert float(Quantity(Decimal("0.25"), "1").value) == 0.25


def test_construct_with_unit_object_and_bad_unit_type() -> None:
    assert Quantity(1.0, duq.unit("kJ/mol")).unit == duq.unit("kJ/mol")
    with pytest.raises(TypeError, match="unit must be"):
        Quantity(1.0, 42)


def test_construct_rejects_core_quantity() -> None:
    with pytest.raises(TypeError, match="from_core"):
        Quantity(duq.Quantity(1.0, "m"), "m")


def test_core_quantity_rejects_jax_magnitude() -> None:
    with pytest.raises(TypeError, match=r"duq\.jax\.Quantity"):
        duq.Quantity(jnp.array(V), "m")


def test_from_core_and_to_core_roundtrip() -> None:
    core = duq.Quantity(np.array(V), "kJ/mol")
    jq = Quantity.from_core(core)
    assert jq.unit == core.unit
    back = jq.to_core()
    assert isinstance(back, duq.Quantity)
    assert back.is_array
    np.testing.assert_allclose(np.asarray(back.value), V)


def test_properties() -> None:
    a = q()
    assert a.dimension == duq.dimension("length")
    assert a.value is a.magnitude
    assert a.ndim == 1
    assert a.size == 3
    assert a.shape == (3,)


# -- conversion -------------------------------------------------------------------


def test_to_and_value_in() -> None:
    a = Quantity(1.0, "km")
    np.testing.assert_allclose(float(a.to("m").value), 1000.0)
    np.testing.assert_allclose(float(a.value_in("cm")), 100_000.0)
    assert a.to("m").unit == duq.unit("m")


def test_to_same_scale_is_noop_relabel() -> None:
    a = Quantity(V, "J")
    b = a.to("kg.m^2/s^2")
    assert b.unit == duq.unit("J")
    np.testing.assert_allclose(np.asarray(b.value), V)


def test_to_affine_celsius() -> None:
    c = Quantity([0.0, 100.0], "degC")
    np.testing.assert_allclose(np.asarray(c.to("K").value), [273.15, 373.15])
    back = c.to("K").to("degC")
    np.testing.assert_allclose(np.asarray(back.value), [0.0, 100.0], atol=1e-12)


def test_to_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        q().to("s")
    with pytest.raises(ValueError, match="equivalence"):
        q().to("m", equivalence="nope")


def test_to_molar_equivalence() -> None:
    a = Quantity(1.0, "kJ/mol")
    j = a.to("J", equivalence="molar")
    assert j.unit == duq.unit("J")
    core = duq.Quantity(1.0, "kJ/mol").to("J", equivalence="molar")
    np.testing.assert_allclose(float(j.value), float(core.value))
    with pytest.raises(duq.DimensionalityError):
        a.to("m", equivalence="molar")
    with pytest.raises(duq.AffineUnitError):
        Quantity(1.0, "degC").to("K", equivalence="molar")


# -- scalar coercion ----------------------------------------------------------------


def test_dimensionless_coercion_matches_core_policy() -> None:
    assert float(Quantity(50.0, "%")) == pytest.approx(0.5)
    assert int(Quantity(200.0, "%")) == 2
    assert complex(Quantity(50.0, "%")) == pytest.approx(0.5 + 0j)
    assert bool(Quantity(50.0, "%")) is True
    assert bool(Quantity(0.0, "1")) is False
    assert float(Quantity(1.5, "rad")) == pytest.approx(1.5)


def test_dimensional_coercion_fails_loud() -> None:
    a = Quantity(2.0, "m")
    for coerce in (float, int, complex, bool):
        with pytest.raises(duq.UnsupportedOperationError, match="ustrip"):
            coerce(a)


def test_array_conversion_fails_loud() -> None:
    with pytest.raises(duq.UnsupportedOperationError, match="ustrip"):
        np.asarray(q())
    with pytest.raises(duq.UnsupportedOperationError):
        q().__array__()


# -- operators -------------------------------------------------------------------------


def test_add_sub_convert_units() -> None:
    a = q()
    b = Quantity([100.0, 200.0, 300.0], "cm")
    np.testing.assert_allclose(np.asarray((a + b).value), [2.0, 4.0, 6.0])
    assert (a + b).unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray((a - b).value), [0.0, 0.0, 0.0])


def test_reflected_add_sub_with_dimensionless() -> None:
    a = Quantity(3.0, "1")
    assert float((2.0 + a).value) == 5.0
    assert float((2.0 - a).value) == -1.0


def test_mul_div_operators() -> None:
    a, b = q(), Quantity([2.0, 5.0, 7.0], "s")
    assert (a * b).unit == duq.unit("m*s")
    assert (a / b).unit == duq.unit("m/s")
    assert (2.0 * a).unit == duq.unit("m")
    assert (a / 2.0).unit == duq.unit("m")
    assert (1.0 / a).unit == duq.unit("m^-1")


def test_floordiv_and_mod_operators() -> None:
    a = Quantity([7.0], "m")
    b = Quantity([300.0], "cm")
    np.testing.assert_allclose(np.asarray((a % b).value), [1.0])
    assert (a % b).unit == duq.unit("m")
    result = Quantity([7.0], "1") // Quantity([3.0], "1")
    np.testing.assert_allclose(np.asarray(result.value), [2.0])
    assert float((8.0 % Quantity(3.0, "1")).value) == 2.0
    assert float((8.0 // Quantity(3.0, "1")).value) == 2.0


def test_pow_operator() -> None:
    a = q()
    assert (a**2).unit == duq.unit("m^2")
    assert (a**0.5).unit == duq.unit("m") ** Fraction(1, 2)
    np.testing.assert_allclose(np.asarray((a**2).value), np.array(V) ** 2)


def test_matmul_operator() -> None:
    a = Quantity(jnp.eye(3), "kg")
    b = q()
    assert (a @ b).unit == duq.unit("kg*m")
    assert (np.asarray(np.eye(3)) @ b).unit == duq.unit("m")


def test_unary_operators() -> None:
    a = q()
    np.testing.assert_allclose(np.asarray((-a).value), [-1.0, -2.0, -3.0])
    assert (+a) is a
    np.testing.assert_allclose(np.asarray(abs(-a).value), V)


def test_comparison_operators_convert() -> None:
    a = q()
    b = Quantity([100.0, 200.0, 300.0], "cm")
    assert bool(jnp.all(a == b))
    assert not bool(jnp.any(a != b))
    assert bool(jnp.all(a <= b))
    assert bool(jnp.all(a >= b))
    assert not bool(jnp.any(a < b))
    assert not bool(jnp.any(a > b))


def test_eq_on_incompatible_dims_matches_numpy_layer() -> None:
    a, b = q(), Quantity(V, "s")
    jax_eq = np.asarray(a == b)
    jax_ne = np.asarray(a != b)
    core_a = duq.Quantity(np.array(V), "m")
    core_b = duq.Quantity(np.array(V), "s")
    np.testing.assert_array_equal(jax_eq, np.asarray(core_a == core_b))
    np.testing.assert_array_equal(jax_ne, np.asarray(core_a != core_b))
    assert not jax_eq.any()
    assert jax_ne.all()


def test_order_compare_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        _ = q() < Quantity(V, "s")


def test_operator_with_unsupported_type_returns_notimplemented() -> None:
    with pytest.raises(TypeError):
        _ = q() + "banana"


def test_mixing_core_quantity_raises_with_guidance() -> None:
    with pytest.raises(TypeError, match="from_core"):
        _ = q() * duq.Quantity(2.0, "s")
    with pytest.raises(TypeError, match="from_core"):
        _ = duq.Quantity(2.0, "s") * q()


def test_unhashable_like_ndarray() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        hash(q())


# -- array ergonomics ----------------------------------------------------------------------


def test_getitem_len_and_at() -> None:
    a = q()
    assert float(a[0].value) == 1.0
    assert a[0].unit == duq.unit("m")
    assert len(a) == 3
    with pytest.raises(TypeError, match="unsized"):
        len(Quantity(1.0, "m"))
    updated = a.at[0].set(Quantity(900.0, "cm"))
    np.testing.assert_allclose(np.asarray(updated.value), [9.0, 2.0, 3.0])
    added = a.at[0].add(Quantity(100.0, "cm"))
    np.testing.assert_allclose(np.asarray(added.value), [2.0, 2.0, 3.0])
    subtracted = a.at[0].subtract(Quantity(100.0, "cm"))
    np.testing.assert_allclose(np.asarray(subtracted.value), [0.0, 2.0, 3.0])
    clipped_lo = a.at[0].max(Quantity(250.0, "cm"))
    np.testing.assert_allclose(np.asarray(clipped_lo.value), [2.5, 2.0, 3.0])
    clipped_hi = a.at[2].min(Quantity(250.0, "cm"))
    np.testing.assert_allclose(np.asarray(clipped_hi.value), [1.0, 2.0, 2.5])
    got = a.at[1].get()
    assert float(got.value) == 2.0
    assert got.unit == duq.unit("m")


def test_at_set_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        q().at[0].set(Quantity(1.0, "s"))


# -- rendering / pytree -----------------------------------------------------------------------


def test_repr_and_str() -> None:
    a = Quantity(2.0, "kJ/mol")
    assert "Quantity(" in repr(a)
    assert "kJ" in repr(a)
    assert str(a).endswith("kJ·mol⁻¹")


def test_pytree_flatten_keeps_unit_static() -> None:
    a = q()
    leaves, treedef = jax.tree_util.tree_flatten(a)
    assert len(leaves) == 1
    assert leaves[0].shape == (3,)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.unit == duq.unit("m")
    other_treedef = jax.tree_util.tree_flatten(Quantity(V, "km"))[1]
    assert treedef != other_treedef  # distinct units -> distinct treedefs (jit keys)


def test_uconvert_ustrip_dispatch_on_jax_quantity() -> None:
    a = Quantity(1.0, "m")
    converted = duq.uconvert("cm", a)
    assert isinstance(converted, Quantity)
    assert converted.unit == duq.unit("cm")
    stripped = duq.ustrip("cm", a)
    assert isinstance(stripped, jax.Array)
    np.testing.assert_allclose(float(stripped), 100.0)
