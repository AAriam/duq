"""The duq.jax.numpy quaxified namespace: memoisation, passthrough, errors."""

# mypy: disable-error-code="arg-type, call-overload, attr-defined, no-untyped-call"

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import duq
import duq.jax
import duq.jax.numpy as djnp


def test_functions_are_quaxified_and_memoised() -> None:
    first = djnp.sqrt
    assert first is djnp.sqrt  # second access hits the memoised module global
    result = first(duq.jax.Quantity([4.0], "m^2"))
    assert result.unit == duq.unit("m")


def test_plain_arrays_behave_like_jax_numpy() -> None:
    np.testing.assert_allclose(np.asarray(djnp.sqrt(jnp.array([4.0]))), [2.0])
    np.testing.assert_allclose(np.asarray(djnp.linspace(0.0, 1.0, 3)), np.linspace(0.0, 1.0, 3))


def test_non_callables_pass_through() -> None:
    assert djnp.pi == jnp.pi
    assert djnp.float32 is jnp.float32  # a type: not wrapped


def test_missing_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = djnp.definitely_not_a_numpy_function


def test_dir_lists_jnp_names() -> None:
    listing = dir(djnp)
    assert "sqrt" in listing
    assert "concatenate" in listing


def test_submodules_are_quaxified_proxies() -> None:
    # jnp.linalg is proxied so its covered functions carry units...
    result = djnp.linalg.norm(duq.jax.Quantity([3.0, 4.0], "m"))
    assert result.unit == duq.unit("m")
    np.testing.assert_allclose(float(result.value), 5.0)
    assert "linalg" in repr(djnp.linalg)
    assert "norm" in dir(djnp.linalg)
    assert djnp.linalg.norm is djnp.linalg.norm  # memoised
    # ...and plain arrays still behave like jax.numpy.linalg
    np.testing.assert_allclose(float(djnp.linalg.norm(jnp.array([3.0, 4.0]))), 5.0)


def test_angle_conversion_overrides_are_unit_aware() -> None:
    q = duq.jax.Quantity([90.0], "deg")
    result = djnp.deg2rad(q)
    assert result.unit == duq.unit("rad")
    np.testing.assert_allclose(np.asarray(result.value), [np.pi / 2])
    # plain operands pass straight through to jax.numpy
    np.testing.assert_allclose(float(djnp.rad2deg(np.pi)), 180.0)
