"""Suite-wide collection config: skip the JAX tests when JAX is absent."""

from __future__ import annotations

import importlib.util

collect_ignore: list[str] = []
if importlib.util.find_spec("jax") is None:
    collect_ignore.append("jax")
