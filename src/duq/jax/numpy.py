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

from typing import Any

import jax.numpy as jnp
import quax

__all__ = ()  # populated dynamically; see __getattr__/__dir__


def __getattr__(name: str) -> Any:
    """Return the quaxified ``jax.numpy`` attribute (memoised in the module)."""
    attribute = getattr(jnp, name)  # AttributeError propagates with jnp's message
    if callable(attribute) and not isinstance(attribute, type):
        attribute = quax.quaxify(attribute)
    globals()[name] = attribute  # memoise: next access skips __getattr__
    return attribute


def __dir__() -> list[str]:
    """List the mirrored ``jax.numpy`` namespace."""
    return sorted(set(globals()) | set(dir(jnp)))
