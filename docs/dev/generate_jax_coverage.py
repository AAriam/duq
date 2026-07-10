"""Regenerate ``docs/dev/jax_coverage.md`` from ``duq.jax.PRIMITIVE_COVERAGE``.

Run from the repository root inside the test-jax environment::

    pixi run -e test-jax python docs/dev/generate_jax_coverage.py

``tests/jax/test_meta_jax.py`` asserts the generated table stays in sync with
the registered primitive rules.
"""

from __future__ import annotations

from pathlib import Path

import duq.jax

_HEADER = """\
# duq.jax primitive coverage

Generated from `duq.jax.PRIMITIVE_COVERAGE` (the module-level registry in
`src/duq/jax/_primitives.py`) by `docs/dev/generate_jax_coverage.py`;
`tests/jax/test_meta_jax.py` asserts this table stays in sync with the
registered rules.  Any `lax` primitive **not** listed here fails loud with
`UnsupportedOperationError` naming the primitive (via
`duq.jax.Quantity.default`); units are never silently dropped.

The control-flow primitives (`cond`, `while`, `scan`, `pjit`) are handled
by quax itself and work with quantity carries/branches out of the box.
"""

_FOOTER = """\
## Known exclusions

- `jax.numpy.interp` compares its inputs against a hard-coded epsilon
  literal internally (dimensionally unsound at the primitive level); strip
  units with `duq.ustrip` around it.
- `jnp.deg2rad`/`rad2deg`/`radians`/`degrees` lower to a bare
  multiplication, so `duq.jax.numpy` ships curated unit-aware overrides
  for them instead of the quaxified originals.
- `lax.linalg` decompositions (`cholesky`, `svd`, `eig`, ...),
  `cumlogsumexp`, `scatter_mul`, FFTs and convolutions are uncovered and
  fail loud; convert to a canonical unit and strip before calling them.
- `cumprod`/`reduce_prod` differ: `reduce_prod` maps to `unit ** n`
  (static reduced size), while `cumprod` would give every element a
  different unit and is rejected for dimensional operands.
"""


def main() -> None:
    """Write the coverage table next to this script."""
    rows = duq.jax.PRIMITIVE_COVERAGE
    name_width = max(len(name) for name, _ in rows) + 2
    desc_width = max(len(desc) for _, desc in rows)
    lines = [_HEADER]
    lines.append(f"| {'Primitive'.ljust(name_width)} | Unit rule |")
    lines.append(f"|{'-' * (name_width + 2)}|{'-' * (desc_width + 2)}|")
    lines.extend(f"| {('`' + name + '`').ljust(name_width)} | {desc} |" for name, desc in rows)
    lines.append("")
    lines.append(_FOOTER)
    target = Path(__file__).with_name("jax_coverage.md")
    target.write_text("\n".join(lines))
    print(f"wrote {target} ({len(rows)} primitives)")


if __name__ == "__main__":
    main()
