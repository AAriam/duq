"""Unit-aware JAX transformation wrappers.

A :class:`duq.jax.Quantity` is a pytree (magnitude = traced leaf, unit =
static), so the plain JAX transformations already accept quantities directly;
:func:`jit` and :func:`vmap` below are documented thin aliases.  The
derivative wrappers (:func:`grad`, :func:`jacfwd`, :func:`jacrev`,
:func:`hessian`) additionally label their outputs with the exact
``unit_out / unit_in`` derivative unit, which raw :func:`jax.grad` cannot do
(it returns the gradient with the *input* pytree structure, hence the input
unit label).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax

from ._quantity import Quantity

if TYPE_CHECKING:
    from collections.abc import Callable

    from duq._unit import Unit

__all__ = ("grad", "hessian", "jacfwd", "jacrev", "jit", "vmap")


def _is_quantity(x: object) -> bool:
    return isinstance(x, Quantity)


def _relabel_gradient(grads: Any, unit_out: Unit | None) -> Any:
    """Relabel a gradient pytree: each Quantity leaf gets ``unit_out / unit_in``.

    ``jax.grad`` returns the gradient with the *input* tree structure, so a
    quantity gradient arrives labelled with the input unit.  When the
    differentiated function returned a plain (already-stripped) scalar,
    ``unit_out`` is ``None`` and the bare magnitudes are returned instead --
    duq never guesses a unit it cannot know.
    """

    def relabel(leaf: Any) -> Any:
        if isinstance(leaf, Quantity):
            if unit_out is None:
                return leaf.magnitude
            return Quantity(leaf.magnitude, unit_out / leaf.unit)
        return leaf

    return jax.tree_util.tree_map(relabel, grads, is_leaf=_is_quantity)


def grad(
    fun: Callable[..., Any], argnums: int | tuple[int, ...] = 0, **kwargs: Any
) -> Callable[..., Any]:
    """Return a unit-aware gradient function (see :func:`jax.grad`).

    ``fun`` may accept and return quantities; quantities may also be created
    freely inside it.  When ``fun`` returns a :class:`~duq.jax.Quantity`, the
    gradient of every quantity argument is labelled with the exact derivative
    unit ``unit_out / unit_in``.  When ``fun`` returns a plain scalar (e.g.
    after :func:`duq.ustrip`), plain gradient magnitudes are returned.

    Parameters
    ----------
    fun : callable
        A function returning a scalar :class:`~duq.jax.Quantity` or a plain
        scalar array.
    argnums : int or tuple of int, optional
        Which positional argument(s) to differentiate with respect to.
    **kwargs
        Forwarded to :func:`jax.grad` (e.g. ``has_aux`` is not supported).

    Returns
    -------
    callable
        The gradient function.

    Examples
    --------
    >>> import duq, duq.jax
    >>> def kinetic(v):
    ...     return 0.5 * duq.jax.Quantity(2.0, "kg") * v * v
    >>> g = duq.jax.grad(kinetic)(duq.jax.Quantity(3.0, "m/s"))
    >>> g.unit == duq.unit("J") / duq.unit("m/s")
    True
    """

    @functools.wraps(fun)
    def wrapper(*args: Any, **fkwargs: Any) -> Any:
        out_struct = jax.eval_shape(fun, *args, **fkwargs)
        if isinstance(out_struct, Quantity):
            unit_out: Unit | None = out_struct.unit

            def scalar_fun(*a: Any, **k: Any) -> Any:
                return fun(*a, **k).magnitude

        else:
            unit_out = None
            scalar_fun = fun
        grads = jax.grad(scalar_fun, argnums=argnums, **kwargs)(*args, **fkwargs)
        return _relabel_gradient(grads, unit_out)

    return wrapper


def _collapse_nested(out: Quantity) -> Quantity:
    """Collapse a nested quantity (out-tree over in-tree) into one unit.

    ``jax.jacfwd(f)(q)`` returns the *output* pytree with each leaf replaced
    by the *input* pytree, so a quantity Jacobian arrives as
    ``Quantity(Quantity(jac, unit_in), unit_out)``; the derivative unit is the
    quotient of the layers.
    """
    unit = out.unit
    magnitude: Any = out.magnitude
    while isinstance(magnitude, Quantity):
        unit = unit / magnitude.unit
        magnitude = magnitude.magnitude
    return Quantity(magnitude, unit)


def _is_nested_quantity(x: object) -> bool:
    return isinstance(x, Quantity) and isinstance(x.magnitude, Quantity)


def _jacobian_like(transform: Callable[..., Any]) -> Callable[..., Any]:
    """Build a unit-collapsing wrapper factory around a JAX Jacobian transform."""

    def factory(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        transformed = transform(fun, **kwargs)

        @functools.wraps(fun)
        def wrapper(*args: Any, **fkwargs: Any) -> Any:
            out = transformed(*args, **fkwargs)
            return jax.tree_util.tree_map(
                lambda x: _collapse_nested(x) if isinstance(x, Quantity) else x,
                out,
                is_leaf=_is_nested_quantity,
            )

        return wrapper

    return factory


def jacfwd(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return a unit-aware forward-mode Jacobian function (see :func:`jax.jacfwd`).

    The Jacobian of a quantity-valued function of a quantity argument is a
    :class:`~duq.jax.Quantity` with unit ``unit_out / unit_in``.

    Parameters
    ----------
    fun : callable
        The function to differentiate.
    **kwargs
        Forwarded to :func:`jax.jacfwd`.

    Returns
    -------
    callable
        The Jacobian function.

    Examples
    --------
    >>> import duq, duq.jax
    >>> import jax.numpy as jnp
    >>> f = lambda q: q * q
    >>> j = duq.jax.jacfwd(f)(duq.jax.Quantity(jnp.array([1.0, 2.0]), "m"))
    >>> j.unit == duq.unit("m")
    True
    """
    return _jacobian_like(jax.jacfwd)(fun, **kwargs)


def jacrev(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return a unit-aware reverse-mode Jacobian function (see :func:`jax.jacrev`).

    Parameters
    ----------
    fun : callable
        The function to differentiate.
    **kwargs
        Forwarded to :func:`jax.jacrev`.

    Returns
    -------
    callable
        The Jacobian function.
    """
    return _jacobian_like(jax.jacrev)(fun, **kwargs)


def hessian(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return a unit-aware Hessian function (see :func:`jax.hessian`).

    The Hessian of a quantity-valued function of a quantity argument is a
    :class:`~duq.jax.Quantity` with unit ``unit_out / unit_in**2``.

    Parameters
    ----------
    fun : callable
        The function to differentiate twice.
    **kwargs
        Forwarded to :func:`jax.hessian`.

    Returns
    -------
    callable
        The Hessian function.
    """
    return _jacobian_like(jax.hessian)(fun, **kwargs)


def jit(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return ``fun`` compiled with :func:`jax.jit` (quantities pass as pytrees).

    A quantity's unit is static pytree metadata and therefore part of the
    compilation cache key: call sites with a fixed unit compile once; a new
    unit retraces.  Strip to a canonical unit with :func:`duq.ustrip` before a
    hot kernel to avoid per-op unit bookkeeping entirely.

    Parameters
    ----------
    fun : callable
        The function to compile; it may accept and return quantities.
    **kwargs
        Forwarded to :func:`jax.jit` (``static_argnums``, ``donate_argnums``, ...).

    Returns
    -------
    callable
        The compiled function.

    Examples
    --------
    >>> import duq.jax
    >>> f = duq.jax.jit(lambda q: q * q)
    >>> str(f(duq.jax.Quantity(3.0, "m")).unit)
    'm²'
    """
    return jax.jit(fun, **kwargs)


def vmap(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return ``fun`` vectorised with :func:`jax.vmap` (quantities pass as pytrees).

    Batching maps over the magnitude leaf; the unit is pytree structure and is
    shared across the whole batch (one unit per array, by design).

    Parameters
    ----------
    fun : callable
        The function to vectorise; it may accept and return quantities.
    **kwargs
        Forwarded to :func:`jax.vmap` (``in_axes``, ``out_axes``, ...).

    Returns
    -------
    callable
        The vectorised function.

    Examples
    --------
    >>> import duq.jax
    >>> import jax.numpy as jnp
    >>> f = duq.jax.vmap(lambda q: q * q)
    >>> str(f(duq.jax.Quantity(jnp.array([1.0, 2.0]), "m")).unit)
    'm²'
    """
    return jax.vmap(fun, **kwargs)
