"""Hypothesis strategies for duq dimensions, units and quantities."""

from __future__ import annotations

import hypothesis.strategies as st

import duq
from duq import Dimension, Quantity, Unit

_BASE_SYMBOLS = ("T", "L", "M", "I", "Θ", "N", "J")

# Atoms with exact (Fraction) scales and no affine offset, so composed units
# keep exact scales and equality stays reliable.
_LINEAR_ATOMS = (
    "m",
    "s",
    "kg",
    "g",
    "A",
    "K",
    "mol",
    "cd",
    "J",
    "N",
    "Pa",
    "W",
    "C",
    "V",
    "Hz",
    "eV",
    "L",
    "bar",
    "min",
    "h",
)

_nonzero_exponents = st.integers(min_value=-3, max_value=3).filter(lambda x: x != 0)


@st.composite
def dimensions(draw: st.DrawFn) -> Dimension:
    """Draw a Dimension with small integer base exponents."""
    mapping = draw(
        st.dictionaries(
            st.sampled_from(_BASE_SYMBOLS),
            st.integers(min_value=-3, max_value=3),
            max_size=4,
        )
    )
    return Dimension(mapping)


@st.composite
def units(draw: st.DrawFn) -> Unit:
    """Draw a Unit composed of one to three exact-scale atoms."""
    count = draw(st.integers(min_value=1, max_value=3))
    result: Unit | None = None
    for _ in range(count):
        symbol = draw(st.sampled_from(_LINEAR_ATOMS))
        exponent = draw(_nonzero_exponents)
        part = duq.unit(symbol) ** exponent
        result = part if result is None else result * part
    assert result is not None
    return result


@st.composite
def finite_values(draw: st.DrawFn) -> float:
    """Draw a finite, non-tiny float value."""
    value = draw(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return value if abs(value) > 1e-6 else 1.0


@st.composite
def quantities(draw: st.DrawFn) -> Quantity:
    """Draw a scalar Quantity with an exact-scale unit."""
    return Quantity(draw(finite_values()), draw(units()))
