"""jit / grad / vmap / scan / cond / while behaviour with unit-carrying arrays."""

# mypy: disable-error-code="arg-type, call-overload, attr-defined, union-attr"
# mypy: disable-error-code="type-var, return-value, no-untyped-call, operator, no-any-return"

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


def kinetic(v: Quantity) -> Quantity:
    """Return the kinetic energy of a 2 kg mass (a quantity created inside)."""
    return 0.5 * Quantity(2.0, "kg") * v * v


# -- jit ---------------------------------------------------------------------------


def test_jit_pytree_style_unit_correct() -> None:
    result = jax.jit(lambda q: q * q)(Quantity([3.0], "m"))
    assert isinstance(result, Quantity)
    assert result.unit == duq.unit("m^2")
    np.testing.assert_allclose(np.asarray(result.value), [9.0])


def test_jit_wrapper_unit_correct() -> None:
    result = duq.jax.jit(kinetic)(Quantity(3.0, "m/s"))
    assert result.unit == duq.unit("kg") * duq.unit("m/s") ** 2
    np.testing.assert_allclose(float(result.value), 9.0)


def test_jit_retraces_per_unit_but_not_per_call() -> None:
    traces = []

    @duq.jax.jit
    def f(q: Quantity) -> Quantity:
        traces.append(q.unit)  # Python side effect: runs only when tracing
        return q * q

    a = jnp.array([1.0, 2.0])
    f(Quantity(a, "m"))
    f(Quantity(a * 2, "m"))  # same unit -> cached, no retrace
    assert len(traces) == 1
    result = f(Quantity(a, "km"))  # new unit -> retrace
    assert len(traces) == 2
    assert result.unit == duq.unit("km^2")
    f(Quantity(a, "km"))
    assert len(traces) == 2


def test_jit_with_ustrip_hot_kernel_pattern() -> None:
    @duq.jax.jit
    def hot(mag: jax.Array) -> jax.Array:
        return mag * 2.0

    q = Quantity([1.0, 2.0], "km")
    out = Quantity(hot(duq.ustrip("m", q)), "m")
    np.testing.assert_allclose(np.asarray(out.value), [2000.0, 4000.0])


# -- grad --------------------------------------------------------------------------


def test_grad_unit_is_exactly_out_over_in() -> None:
    g = duq.jax.grad(kinetic)(Quantity(3.0, "m/s"))
    assert isinstance(g, Quantity)
    assert g.unit == duq.unit("J") / duq.unit("m/s")
    np.testing.assert_allclose(float(g.value), 6.0)  # d(1/2 m v^2)/dv = m v


def test_grad_of_ustripped_scalar_returns_plain() -> None:
    g = duq.jax.grad(lambda q: duq.ustrip("J", kinetic(q)))(Quantity(3.0, "m/s"))
    assert not isinstance(g, Quantity)
    np.testing.assert_allclose(float(g), 6.0)


def test_grad_multiple_argnums() -> None:
    def energy(m: Quantity, v: Quantity) -> Quantity:
        return 0.5 * m * v * v

    gm, gv = duq.jax.grad(energy, argnums=(0, 1))(Quantity(2.0, "kg"), Quantity(3.0, "m/s"))
    assert gm.unit == duq.unit("J") / duq.unit("kg")
    assert gv.unit == duq.unit("J") / duq.unit("m/s")
    np.testing.assert_allclose(float(gm.value), 4.5)
    np.testing.assert_allclose(float(gv.value), 6.0)


def test_grad_through_unit_conversion() -> None:
    g = duq.jax.grad(lambda q: (q * q).to("cm^2"))(Quantity(3.0, "m"))
    assert g.unit == duq.unit("cm^2") / duq.unit("m")
    np.testing.assert_allclose(float(g.value), 60_000.0)  # d(x^2 cm^2)/dx m


def test_jacfwd_jacrev_and_hessian_smoke() -> None:
    square = lambda q: q * q  # noqa: E731
    x = Quantity(jnp.array([1.0, 2.0]), "m")
    jf = duq.jax.jacfwd(square)(x)
    assert isinstance(jf, Quantity)
    assert jf.unit == duq.unit("m^2") / duq.unit("m")
    np.testing.assert_allclose(np.asarray(jf.value), np.diag([2.0, 4.0]))
    jr = duq.jax.jacrev(square)(x)
    assert jr.unit == jf.unit
    np.testing.assert_allclose(np.asarray(jr.value), np.asarray(jf.value))
    h = duq.jax.hessian(kinetic)(Quantity(3.0, "m/s"))
    assert h.unit == duq.unit("J") / duq.unit("m/s") ** 2
    np.testing.assert_allclose(float(h.value), 2.0)  # d2(1/2 m v^2)/dv2 = m


# -- vmap ---------------------------------------------------------------------------


def test_vmap_over_magnitude_axis_unit_shared() -> None:
    batched = duq.jax.vmap(kinetic)(Quantity(jnp.array([1.0, 2.0, 3.0]), "m/s"))
    assert isinstance(batched, Quantity)
    assert batched.unit == duq.unit("kg") * duq.unit("m/s") ** 2
    np.testing.assert_allclose(np.asarray(batched.value), [1.0, 4.0, 9.0])


def test_plain_jax_vmap_works_on_quantity_pytrees() -> None:
    batched = jax.vmap(lambda q: q + q)(Quantity(jnp.array([1.0, 2.0]), "m"))
    assert batched.unit == duq.unit("m")


# -- control flow ---------------------------------------------------------------------


def test_scan_carry_keeps_unit_across_steps() -> None:
    def step(carry: Quantity, x: Quantity) -> tuple[Quantity, Quantity]:
        new = carry + x
        return new, new

    xs = Quantity(jnp.array([1.0, 2.0, 3.0]), "m")
    final, ys = jax.lax.scan(step, Quantity(0.0, "m"), xs)
    assert final.unit == duq.unit("m")
    assert ys.unit == duq.unit("m")
    np.testing.assert_allclose(float(final.value), 6.0)
    np.testing.assert_allclose(np.asarray(ys.value), [1.0, 3.0, 6.0])


def test_scan_carry_converts_mixed_units_per_rule() -> None:
    def step(carry: Quantity, x: Quantity) -> tuple[Quantity, Quantity]:
        new = carry + x.to("m")
        return new, new

    xs = Quantity(jnp.array([100.0, 200.0]), "cm")
    final, _ = jax.lax.scan(step, Quantity(0.0, "m"), xs)
    np.testing.assert_allclose(float(final.value), 3.0)


def test_while_loop_with_quantity_carry() -> None:
    result = jax.lax.while_loop(
        lambda c: c < Quantity(10.0, "m"),
        lambda c: c * 2.0,
        Quantity(1.0, "m"),
    )
    assert result.unit == duq.unit("m")
    np.testing.assert_allclose(float(result.value), 16.0)


def test_cond_with_consistent_units() -> None:
    def branchy(pred: object, q: Quantity) -> Quantity:
        return jax.lax.cond(pred, lambda x: x * 2.0, lambda x: x * 3.0, q)

    a = Quantity(1.0, "m")
    np.testing.assert_allclose(float(branchy(True, a).value), 2.0)
    np.testing.assert_allclose(float(branchy(False, a).value), 3.0)
    assert branchy(True, a).unit == duq.unit("m")


def test_cond_with_different_units_raises_clearly() -> None:
    with pytest.raises(TypeError, match="pytree"):
        jax.lax.cond(True, lambda x: x.to("cm"), lambda x: x, Quantity(1.0, "m"))


def test_where_converts_compatible_branch_units() -> None:
    # jnp.where is the select_n route: branches converge to the first unit.
    result = djnp.where(
        jnp.array([True, False]),
        Quantity([1.0, 2.0], "m"),
        Quantity([300.0, 400.0], "cm"),
    )
    assert result.unit == duq.unit("m")
    np.testing.assert_allclose(np.asarray(result.value), [1.0, 4.0])


# -- affine °C under jit -----------------------------------------------------------------


def test_celsius_to_kelvin_under_jit() -> None:
    convert = duq.jax.jit(lambda q: q.to("K"))
    result = convert(Quantity(jnp.array([0.0, 100.0]), "degC"))
    assert result.unit == duq.unit("K")
    np.testing.assert_allclose(np.asarray(result.value), [273.15, 373.15])


def test_celsius_arithmetic_raises_at_trace_time() -> None:
    celsius = Quantity(jnp.array([25.0]), "degC")
    with pytest.raises(duq.AffineUnitError):
        duq.jax.jit(lambda q: q + q)(celsius)
    with pytest.raises(duq.AffineUnitError):
        duq.jax.jit(lambda q: q * 2.0)(celsius)
    diff = duq.jax.jit(lambda q: q - Quantity(jnp.array([20.0]), "degC"))(celsius)
    assert diff.unit == duq.unit("K")  # degC - degC is a difference in kelvin
    np.testing.assert_allclose(np.asarray(diff.value), [5.0])


# -- quaxify interoperability -----------------------------------------------------------


def test_raw_quaxify_style_with_arguments_only() -> None:
    # The quaxed-style composition works when quantities enter as arguments.
    def f(x: object, y: object) -> object:
        return jnp.sqrt(x * x + y * y)

    result = quax.quaxify(f)(Quantity(3.0, "m"), Quantity(400.0, "cm"))
    assert isinstance(result, Quantity)
    assert result.unit == duq.unit("m")
    np.testing.assert_allclose(float(result.value), 5.0)
