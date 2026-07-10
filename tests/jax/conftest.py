"""JAX test configuration: enable float64 for parity with the NumPy layer."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
