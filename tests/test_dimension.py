"""Tests for :class:`duq.Dimension`."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given

import duq
from duq import Dimension

from . import strategies as sd

DIMENSIONLESS = Dimension({})


def test_parse_dot_and_star_agree() -> None:
    assert Dimension.parse("M.L^2.T^-2") == Dimension.parse("mass*length**2/time**2")


def test_parse_named_derived() -> None:
    assert Dimension.parse("energy") == Dimension({"M": 1, "L": 2, "T": -2})
    assert Dimension.parse("velocity") == Dimension({"L": 1, "T": -1})


def test_parse_fractional_exponent_is_exact() -> None:
    dim = Dimension.parse("L^3/2")
    assert dim.exponents["L"] == Fraction(3, 2)


def test_construct_from_names_and_symbols() -> None:
    assert Dimension({"mass": 1}) == Dimension({"M": 1})


def test_unknown_base_rejected() -> None:
    with pytest.raises(ValueError, match="unknown base dimension"):
        Dimension({"Q": 1})


def test_parse_unknown_token_rejected() -> None:
    with pytest.raises(duq.UnitParseError, match="unknown dimension"):
        Dimension.parse("wibble")


def test_is_dimensionless() -> None:
    assert DIMENSIONLESS.is_dimensionless
    assert not Dimension({"L": 1}).is_dimensionless


def test_str_and_repr_roundtrip() -> None:
    energy = Dimension({"M": 1, "L": 2, "T": -2})
    assert str(energy) == "M·L²·T⁻²"
    assert eval(repr(energy)) == energy


def test_format_styles() -> None:
    energy = Dimension({"M": 1, "L": 2, "T": -2})
    assert energy.format("plain") == "M.L^2.T^-2"
    assert "\\mathrm" in energy.format("latex")


def test_pow_accepts_exact_float() -> None:
    assert Dimension({"L": 2}) ** 0.5 == Dimension({"L": 1})


def test_pow_rejects_inexact_float() -> None:
    with pytest.raises(ValueError, match="not an exact simple fraction"):
        Dimension({"L": 1}) ** 0.333


def test_pow_rejects_bool_exponent() -> None:
    with pytest.raises(TypeError):
        Dimension({"L": 1}) ** True


def test_dunder_returns_notimplemented_for_foreign() -> None:
    assert Dimension({"L": 1}).__mul__(3) is NotImplemented
    assert Dimension({"L": 1}).__truediv__("x") is NotImplemented
    assert Dimension({"L": 1}).__eq__(3) is NotImplemented
    assert Dimension({"L": 1}).__pow__("x") is NotImplemented


def test_exponents_mapping_is_readonly() -> None:
    exps = Dimension({"L": 1}).exponents
    with pytest.raises(TypeError):
        exps["L"] = Fraction(2)  # type: ignore[index]


@given(a=sd.dimensions(), b=sd.dimensions(), c=sd.dimensions())
def test_multiplication_is_associative(a: Dimension, b: Dimension, c: Dimension) -> None:
    assert (a * b) * c == a * (b * c)


@given(a=sd.dimensions(), b=sd.dimensions())
def test_multiplication_is_commutative(a: Dimension, b: Dimension) -> None:
    assert a * b == b * a


@given(a=sd.dimensions())
def test_identity_and_inverse(a: Dimension) -> None:
    assert a * DIMENSIONLESS == a
    assert a * a**-1 == DIMENSIONLESS
    assert a / a == DIMENSIONLESS


@given(a=sd.dimensions())
def test_hash_consistent_with_equality(a: Dimension) -> None:
    b = Dimension(dict(a.exponents))
    assert a == b
    assert hash(a) == hash(b)
