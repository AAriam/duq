"""Targeted tests for less-common branches."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest

import duq
from duq import Quantity
from duq._parse import tokenize


def test_decimal_conversion_with_float_scale() -> None:
    result = Quantity(Decimal("2"), "deg").to("rad").value
    assert isinstance(result, Decimal)
    assert result == pytest.approx(Decimal(str(2 * math.pi / 180)))


def test_complex_conversion() -> None:
    result = Quantity(1 + 1j, "m").to("cm").value
    assert result == 100 + 100j


def test_complex_addition_with_fraction_scale() -> None:
    total = Quantity(1 + 0j, "km") + Quantity(500 + 0j, "m")
    assert total.value == pytest.approx(1.5)


def test_compact_returns_self_for_extreme_value() -> None:
    huge = Quantity(1e40, "Qm")  # already quetta-metre; no larger prefix exists
    assert huge.compact() is huge


def test_compact_handles_infinity() -> None:
    inf = Quantity(math.inf, "m")
    assert inf.compact() is inf


def test_rtruediv_and_pow_reject_non_numbers() -> None:
    assert Quantity(1.0, "m").__rtruediv__("x") is NotImplemented
    assert Quantity(1.0, "m").__pow__("x") is NotImplemented
    assert Quantity(1.0, "m").__mul__("x") is NotImplemented
    assert Quantity(1.0, "m").__truediv__("x") is NotImplemented


def test_molar_reverse_direction() -> None:
    joules = Quantity(1.66053906892e-21, "J")
    per_mol = joules.to("kJ/mol", equivalence="molar")
    assert per_mol.value == pytest.approx(1.0, rel=1e-6)


def test_invalid_superscript_exponent() -> None:
    with pytest.raises(duq.UnitParseError, match="invalid superscript"):
        tokenize("m⁻")


def test_leading_one_times_operator() -> None:
    assert tokenize("1*s") == [("s", Fraction(1))]


def test_constant_set_year_and_private() -> None:
    catalog = duq.constants.codata(2022)
    assert catalog.year == 2022
    with pytest.raises(AttributeError):
        _ = catalog._hidden


def test_ne_returns_notimplemented_for_foreign() -> None:
    assert Quantity(1.0, "m").__ne__(3) is NotImplemented
