"""Unit-aware mirror of :mod:`jax.numpy` (quaxified on first access).

Every function looked up on this module is ``quax.quaxify(jax.numpy.<name>)``,
memoised, so :class:`duq.jax.Quantity` operands dispatch through the duq
primitive rules while plain arrays behave exactly as in :mod:`jax.numpy`.
Non-callable attributes (``pi``, ``inf``, dtypes, ...) are passed through
unchanged.

Examples
--------
>>> import duq.jax
>>> import duq.jax.numpy as djnp
>>> q = duq.jax.Quantity([1.0, 4.0], "m^2")
>>> str(djnp.sqrt(q).unit)
'm'
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax.numpy as jnp
import quax

from duq._errors import DimensionalityError

from ._quantity import Quantity

__all__ = ()  # populated dynamically; see __getattr__/__dir__


def _make_angle_convert(name: str, out_unit: str) -> Any:
    """Build a unit-aware ``deg2rad``-family function (NumPy layer parity).

    ``jax.numpy.deg2rad`` lowers to a bare multiplication, which would leave
    the *label* of an angle quantity unchanged while rescaling its magnitude;
    these curated overrides instead require a dimensionless operand (the raw
    magnitude is passed through, exactly like the NumPy layer) and relabel
    the result with the output angle unit.
    """
    jnp_func = getattr(jnp, name)

    def convert(x: Any) -> Any:
        if isinstance(x, Quantity):
            if not x.unit.is_dimensionless:
                raise DimensionalityError(
                    f"expected a dimensionless operand, got dimension {x.unit.dimension}"
                )
            return Quantity(jnp_func(x.magnitude), x.unit.registry.unit(out_unit))
        return jnp_func(x)

    convert.__name__ = name
    convert.__qualname__ = name
    return convert


#: Functions whose jax.numpy implementation is unit-unsafe at the primitive
#: level and therefore gets a curated, unit-aware override.
_OVERRIDES: dict[str, Any] = {
    "deg2rad": _make_angle_convert("deg2rad", "rad"),
    "radians": _make_angle_convert("radians", "rad"),
    "rad2deg": _make_angle_convert("rad2deg", "deg"),
    "degrees": _make_angle_convert("degrees", "deg"),
}


class _QuaxifiedModule:
    """A quaxified view of a ``jax.numpy`` submodule (``linalg``, ``fft``, ...)."""

    __slots__ = ("_cache", "_module")

    def __init__(self, module: Any) -> None:
        self._module = module
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        attribute = getattr(self._module, name)
        attribute = _wrap(attribute)
        self._cache[name] = attribute
        return attribute

    def __dir__(self) -> list[str]:
        return sorted(set(self._cache) | set(dir(self._module)))

    def __repr__(self) -> str:
        return f"<duq.jax quaxified view of {self._module.__name__!r}>"


def _wrap(attribute: Any) -> Any:
    """Quaxify a callable, proxy a module, and pass anything else through."""
    if isinstance(attribute, ModuleType):
        return _QuaxifiedModule(attribute)
    if callable(attribute) and not isinstance(attribute, type):
        return quax.quaxify(attribute)
    return attribute


def __getattr__(name: str) -> Any:
    """Return the quaxified ``jax.numpy`` attribute (memoised in the module)."""
    override = _OVERRIDES.get(name)
    if override is not None:
        globals()[name] = override
        return override
    attribute = _wrap(getattr(jnp, name))  # AttributeError keeps jnp's message
    globals()[name] = attribute  # memoise: next access skips __getattr__
    return attribute


def __dir__() -> list[str]:
    """List the mirrored ``jax.numpy`` namespace."""
    return sorted(set(globals()) | set(dir(jnp)))
