"""Tests for CODATA constants; values verified against NIST CODATA 2022."""

from __future__ import annotations

import pytest

import duq

# (slug, value, unit-string) transcribed from NIST CODATA 2022.
NIST_2022 = [
    ("c", 299792458.0, "m/s"),
    ("h", 6.62607015e-34, "J.s"),
    ("hbar", 1.054571817e-34, "J.s"),
    ("e", 1.602176634e-19, "C"),
    ("k_B", 1.380649e-23, "J/K"),
    ("N_A", 6.02214076e23, "1/mol"),
    ("R", 8.31446261815324, "J/(mol.K)"),
    ("alpha", 7.2973525643e-3, "1"),
    ("m_e", 9.1093837139e-31, "kg"),
    ("m_p", 1.67262192595e-27, "kg"),
    ("m_n", 1.67492750056e-27, "kg"),
    ("u", 1.66053906892e-27, "kg"),
    ("eps_0", 8.8541878188e-12, "F/m"),
    ("mu_0", 1.25663706127e-6, "N/A^2"),
    ("G", 6.67430e-11, "m^3/(kg.s^2)"),
    ("g_n", 9.80665, "m/s^2"),
    ("sigma_SB", 5.670374419e-8, "W/(m^2.K^4)"),
    ("R_inf", 10973731.568157, "1/m"),
    ("a_0", 5.29177210544e-11, "m"),
    ("E_h", 4.3597447222060e-18, "J"),
]


@pytest.mark.parametrize(("slug", "value", "unit"), NIST_2022)
def test_codata_2022_values(slug: str, value: float, unit: str) -> None:
    q = duq.constants.codata(2022).quantity(slug)
    assert q.value == value
    assert q.dimension == duq.unit(unit).dimension


def test_module_attribute_access() -> None:
    assert duq.constants.k_B.value == 1.380649e-23
    assert duq.constants.N_A.unit == duq.unit("1/mol")


def test_default_year_matches_codata_2022() -> None:
    assert duq.constants.c.value == duq.constants.codata(2022).c.value


def test_boltzmann_exponent_is_minus_23() -> None:
    # Regression: the prototype once had a -32 exponent typo.
    assert duq.constants.k_B.value == pytest.approx(1.380649e-23, rel=1e-12)


def test_derived_relationship_r_equals_na_kb() -> None:
    product = duq.constants.N_A * duq.constants.k_B
    assert product.to("J/(mol.K)").value == pytest.approx(8.31446261815324, rel=1e-12)


def test_r_stores_full_exact_decimal() -> None:
    # R = N_A * k_B is an exact finite decimal: 8.31446261815324 J/(mol.K).
    assert duq.constants.R.value == 8.31446261815324
    assert duq.constants.R.value == pytest.approx(
        duq.constants.N_A.value * duq.constants.k_B.value, rel=1e-15
    )


def test_info_metadata() -> None:
    info = duq.constants.codata(2022).info("h")
    assert info["uncertainty"] == "exact"
    assert info["symbol"] == "h"


def test_unknown_constant_and_year() -> None:
    with pytest.raises(AttributeError):
        _ = duq.constants.nonexistent
    with pytest.raises(ValueError, match="no bundled CODATA data"):
        duq.constants.codata(1999)


def test_dir_lists_constants() -> None:
    assert "k_B" in dir(duq.constants)
    assert "N_A" in dir(duq.constants.codata(2022))
