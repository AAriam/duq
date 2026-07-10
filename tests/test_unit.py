"""Tests for :class:`duq.Unit`."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given

import duq

from . import strategies as sd


def test_equal_units_across_expressions() -> None:
    assert duq.unit("J") == duq.unit("kg.m^2/s^2")
    assert duq.unit("N.m") == duq.unit("J")


def test_km_and_m_not_equal_but_same_dimension() -> None:
    assert duq.unit("km") != duq.unit("m")
    assert duq.unit("km").dimension == duq.unit("m").dimension


def test_scale_is_exact_for_exact_atoms() -> None:
    assert duq.unit("km").scale == Fraction(1000)
    assert duq.unit("eV").scale == Fraction(1602176634, 10**28)


def test_same_expression() -> None:
    assert not duq.unit("J").same_expression(duq.unit("kg.m^2/s^2"))
    assert duq.unit("kJ/mol").same_expression(duq.unit("kJ/mol"))


def test_multiplication_cancels_like_factors() -> None:
    assert str(duq.unit("kJ/mol") * duq.unit("mol")) == "kJ"


def test_reciprocal() -> None:
    assert str(1 / duq.unit("s")) == "s⁻¹"
    assert (1 / duq.unit("s")) == duq.unit("Hz")


def test_reciprocal_only_one() -> None:
    assert duq.unit("s").__rtruediv__(2) is NotImplemented


def test_power_and_pow_type_guard() -> None:
    assert str(duq.unit("m") ** 3) == "m³"
    assert duq.unit("m").__pow__(1.5) is NotImplemented
    assert duq.unit("m").__pow__(True) is NotImplemented


def test_to_coherent_si() -> None:
    assert str(duq.unit("kJ/mol").to_coherent_si()) == "kg·m²·s⁻²·mol⁻¹"


def test_is_dimensionless_and_affine() -> None:
    assert duq.unit("%").is_dimensionless
    assert duq.unit("degC").is_affine
    assert not duq.unit("K").is_affine


def test_format_plain() -> None:
    assert duq.unit("kJ/mol").format("plain") == "kJ.mol^-1"


def test_repr() -> None:
    assert repr(duq.unit("kJ/mol")) == "Unit('kJ.mol^-1')"


def test_dunder_notimplemented_for_foreign() -> None:
    assert duq.unit("m").__mul__(3) is NotImplemented
    assert duq.unit("m").__truediv__("x") is NotImplemented
    assert duq.unit("m").__eq__(3) is NotImplemented


@pytest.mark.parametrize("expression", ["degC^2", "degC.s", "degC^-1", "°C.mol"])
def test_affine_policy_rejects_compounds(expression: str) -> None:
    with pytest.raises(duq.AffineUnitError):
        duq.unit(expression)


def test_affine_bare_is_allowed() -> None:
    assert duq.unit("degC").is_affine
    assert duq.unit("°C").is_affine


def test_prefix_exact_atom_wins() -> None:
    # "T" is tesla (exact atom), not tera + nothing; "Tm" is tera*metre.
    assert duq.unit("T").dimension == duq.dimension("magnetic_flux_density")
    assert duq.unit("Tm").scale == Fraction(10) ** 12


def test_hidden_unit_constructor_rejected() -> None:
    with pytest.raises(TypeError):
        duq.Unit()


@given(a=sd.units(), b=sd.units(), c=sd.units())
def test_unit_multiplication_associative(a: duq.Unit, b: duq.Unit, c: duq.Unit) -> None:
    assert (a * b) * c == a * (b * c)


@given(a=sd.units(), b=sd.units())
def test_unit_multiplication_commutative(a: duq.Unit, b: duq.Unit) -> None:
    assert a * b == b * a


@given(a=sd.units())
def test_unit_inverse(a: duq.Unit) -> None:
    assert a / a == duq.unit("1")


def test_hash_matches_equality_across_expressions() -> None:
    assert hash(duq.unit("J")) == hash(duq.unit("kg.m^2/s^2"))


@given(a=sd.units())
def test_unit_hash_eq_contract(a: duq.Unit) -> None:
    reparsed = duq.unit(a.format("plain"))
    assert a == reparsed
    assert hash(a) == hash(reparsed)
