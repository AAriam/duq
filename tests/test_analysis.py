"""Tests for the dimensional-analysis module."""

from __future__ import annotations

from fractions import Fraction

import pytest

import duq
from duq import Dimension
from duq.analysis import compositions, decompose, name_of, shortest_composition


def test_name_of() -> None:
    assert name_of(duq.dimension("M.L^2.T^-2")) == "energy"
    assert name_of(duq.dimension("mass")) == "mass"
    assert name_of(Dimension({})) == "dimensionless"
    assert name_of(duq.dimension("M^2.L")) is None


def test_decompose() -> None:
    assert decompose(duq.dimension("velocity")) == {"T": Fraction(-1), "L": Fraction(1)}


def test_compositions_finds_force_length() -> None:
    comps = compositions(duq.dimension("energy"), max_terms=2)
    assert {"force": Fraction(1), "length": Fraction(1)} in comps
    # Sorted: shortest / smallest first.
    assert all(len(comps[i]) <= len(comps[i + 1]) for i in range(len(comps) - 1))


def test_compositions_respects_bounds() -> None:
    comps = compositions(duq.dimension("energy"), max_terms=1)
    assert all(len(c) == 1 for c in comps)


def test_shortest_named() -> None:
    assert shortest_composition(duq.dimension("energy")) == {"energy": Fraction(1)}


def test_shortest_dimensionless_is_empty() -> None:
    assert shortest_composition(Dimension({})) == {}


def test_shortest_falls_back_to_base() -> None:
    # A dimension with no single name but an integer base decomposition.
    weird = duq.dimension("M^2.L")
    result = shortest_composition(weird)
    assert result == {"M": Fraction(2), "L": Fraction(1)}


def test_shortest_raises_for_fractional() -> None:
    with pytest.raises(duq.AnalysisError):
        shortest_composition(Dimension({"L": Fraction(1, 2)}))
