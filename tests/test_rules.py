"""Tests for the shared unit-rule engine."""

from __future__ import annotations

from fractions import Fraction

import pytest

from duq import DimensionalityError
from duq._dimension import Dimension
from duq._rules import Rule, apply, result_dimension

LENGTH = Dimension({"L": 1})
AREA = Dimension({"L": 2})
DIMENSIONLESS = Dimension({})


def test_multiply_divide_power() -> None:
    assert apply(Rule.MULTIPLY, LENGTH, LENGTH) == AREA
    assert apply(Rule.DIVIDE, AREA, LENGTH) == LENGTH
    assert apply(Rule.POWER, LENGTH, Fraction(2)) == AREA


def test_same_dim() -> None:
    assert apply(Rule.SAME_DIM, LENGTH, LENGTH) == LENGTH
    with pytest.raises(DimensionalityError):
        apply(Rule.SAME_DIM, LENGTH, AREA)


def test_dimensionless_and_angle_in() -> None:
    assert apply(Rule.DIMENSIONLESS_IN_DIMENSIONLESS_OUT, DIMENSIONLESS) == DIMENSIONLESS
    assert apply(Rule.ANGLE_IN, DIMENSIONLESS) == DIMENSIONLESS
    with pytest.raises(DimensionalityError):
        apply(Rule.DIMENSIONLESS_IN_DIMENSIONLESS_OUT, LENGTH)


def test_preserve_and_roots() -> None:
    assert apply(Rule.PRESERVE, AREA) == AREA
    assert apply(Rule.SQRT, AREA) == LENGTH
    assert apply(Rule.CBRT, Dimension({"L": 3})) == LENGTH


def test_dimensionless_out() -> None:
    assert apply(Rule.DIMENSIONLESS_OUT, LENGTH) == DIMENSIONLESS


def test_result_dimension_alias() -> None:
    assert result_dimension(Rule.PRESERVE, LENGTH) == LENGTH
