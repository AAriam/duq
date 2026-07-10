"""JAX back-end for duq: jit/grad/vmap-safe unit-carrying arrays.

:class:`duq.jax.Quantity` couples a JAX array magnitude with a static
:class:`duq.Unit`.  Units are checked and propagated at *trace time* through
quax primitive interception, so a compiled kernel pays zero runtime cost for
its units.  ``import duq`` never imports JAX; this subpackage requires the
``duq[jax]`` extra (``pip install duq[jax]``).

Construction
------------
Build quantities from anything :func:`jax.numpy.asarray` accepts, convert
with :meth:`~duq.jax.Quantity.to`, and cross over to the NumPy-backed core
with :meth:`~duq.jax.Quantity.from_core` / :meth:`~duq.jax.Quantity.to_core`:

>>> import duq.jax
>>> q = duq.jax.Quantity([1.0, 2.0, 3.0], "m")
>>> q.to("cm").value
Array([100., 200., 300.], dtype=float32)

Operators work eagerly and under every transformation; ``duq.jax.numpy``
mirrors :mod:`jax.numpy` with unit awareness:

>>> import duq.jax.numpy as djnp
>>> str(djnp.sqrt(q * q).unit)
'm'

Transformations
---------------
A quantity is a pytree whose magnitude is the traced leaf and whose unit is
static metadata, so plain :func:`jax.jit`, :func:`jax.vmap`,
:func:`jax.lax.scan` and friends accept quantity arguments directly.
:func:`duq.jax.grad` (and ``jacfwd``/``jacrev``/``hessian``) additionally
label derivative outputs with the exact ``unit_out / unit_in`` unit:

>>> import jax.numpy as jnp
>>> def kinetic(v):
...     return 0.5 * duq.jax.Quantity(2.0, "kg") * v * v
>>> g = duq.jax.grad(kinetic)(duq.jax.Quantity(3.0, "m/s"))
>>> g.unit == duq.unit("J") / duq.unit("m/s")
True

Retracing
---------
The unit is part of the pytree structure, hence part of ``jit``'s cache key:
calling a jitted function with metres and then with kilometres compiles
twice.  Call sites with a fixed unit are compiled once and reused.  For hot
kernels, strip to a canonical unit *before* entering the kernel::

    mag = duq.ustrip("m", q)          # plain jax.Array, no unit bookkeeping
    out = duq.jax.Quantity(hot_kernel(mag), "m/s")

Known limitations
-----------------
- ``jax.custom_vjp`` is not supported by quax (``custom_jvp`` is); strip
  units around a ``custom_vjp`` boundary with :func:`duq.ustrip`.
- Uncovered primitives fail loud with
  :class:`~duq.UnsupportedOperationError` naming the primitive; see
  ``docs/dev/jax_coverage.md`` for the covered set.
- Mixing quantities *created inside* a ``quax.quaxify``-transformed function
  with that function's traced arguments splits the unit bookkeeping across
  trace levels; prefer the :func:`duq.jax.grad`-family wrappers, or pass all
  quantities as arguments when using raw ``quax.quaxify``.
"""

from __future__ import annotations

try:
    import equinox  # noqa: F401  (import order: lightest first)
    import jax  # noqa: F401
    import quax  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised via subprocess test
    raise ImportError(
        "duq.jax requires the optional JAX dependencies (jax, quax, equinox); "
        "install them with:  pip install 'duq[jax]'"
    ) from exc

from duq._quantity import uconvert, ustrip

from . import numpy
from ._primitives import PRIMITIVE_COVERAGE
from ._quantity import Quantity
from ._transforms import grad, hessian, jacfwd, jacrev, jit, vmap

__all__ = (
    "PRIMITIVE_COVERAGE",
    "Quantity",
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "jit",
    "numpy",
    "uconvert",
    "ustrip",
    "vmap",
)
