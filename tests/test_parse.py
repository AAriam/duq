"""Tests for the unit/dimension tokeniser and formatter."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given

import duq
from duq._format import format_composition, order_terms, superscript
from duq._parse import tokenize

from . import strategies as sd


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kg.m^2.s^-2", [("kg", Fraction(1)), ("m", Fraction(2)), ("s", Fraction(-2))]),
        ("kg*m**2/s**2", [("kg", Fraction(1)), ("m", Fraction(2)), ("s", Fraction(-2))]),
        ("kJ/mol", [("kJ", Fraction(1)), ("mol", Fraction(-1))]),
        ("1/s", [("s", Fraction(-1))]),
        ("m^3/2", [("m", Fraction(3, 2))]),
        ("m**(3/2)", [("m", Fraction(3, 2))]),
        ("m²·s⁻¹", [("m", Fraction(2)), ("s", Fraction(-1))]),
        ("kg/(m*s^2)", [("kg", Fraction(1)), ("m", Fraction(-1)), ("s", Fraction(-2))]),
        ("  m /  s ", [("m", Fraction(1)), ("s", Fraction(-1))]),
    ],
)
def test_tokenize_cases(text: str, expected: list[tuple[str, Fraction]]) -> None:
    assert tokenize(text) == expected


def test_tokenize_leading_one_alone() -> None:
    assert tokenize("1") == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("kg..m", "expected a unit or dimension name"),
        ("2*m", "expected a unit or dimension name"),
        ("1m", "unexpected token after leading '1'"),
        ("m^", "expected a digit"),
        ("kg m", "expected"),
        ("(m", "unbalanced"),
        ("m**(3/2", "close exponent"),
    ],
)
def test_tokenize_errors(text: str, message: str) -> None:
    with pytest.raises(duq.UnitParseError, match=message):
        tokenize(text)


def test_group_exponent_applies() -> None:
    assert tokenize("(m.s)^2") == [("m", Fraction(2)), ("s", Fraction(2))]


def test_superscript_roundtrip() -> None:
    assert superscript(-12) == "⁻¹²"


def test_format_styles() -> None:
    terms = [("kJ", Fraction(1)), ("mol", Fraction(-1))]
    assert format_composition(terms, style="unicode") == "kJ·mol⁻¹"
    assert format_composition(terms, style="plain") == "kJ.mol^-1"
    assert format_composition(terms, style="latex") == r"\mathrm{kJ}\,\mathrm{mol}^{-1}"


def test_format_fractional_exponent() -> None:
    assert format_composition([("m", Fraction(3, 2))]) == "m^3/2"
    assert format_composition([("m", Fraction(3, 2))], style="latex") == r"\mathrm{m}^{3/2}"


def test_format_empty_and_unknown_style() -> None:
    assert format_composition([], empty="dimensionless") == "dimensionless"
    with pytest.raises(ValueError, match="unknown style"):
        format_composition([("m", Fraction(1))], style="klingon")


def test_order_terms_positive_first() -> None:
    ordered = order_terms([("s", Fraction(-2)), ("m", Fraction(1))])
    assert ordered == [("m", Fraction(1)), ("s", Fraction(-2))]


@given(unit=sd.units())
def test_parse_format_roundtrip_plain(unit: duq.Unit) -> None:
    assert duq.unit(unit.format("plain")) == unit


@given(unit=sd.units())
def test_parse_format_roundtrip_unicode(unit: duq.Unit) -> None:
    assert duq.unit(str(unit)) == unit
