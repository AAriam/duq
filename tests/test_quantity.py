"""Tests for :class:`duq.Quantity`."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given

import duq
from duq import Quantity

from . import strategies as sd


def test_construction_and_properties() -> None:
    q = Quantity(1.5, "kJ/mol")
    assert q.value == 1.5
    assert str(q.unit) == "kJ·mol⁻¹"
    assert q.dimension == duq.dimension("molar_energy")


def test_construct_from_unit_object() -> None:
    q = Quantity(2.0, duq.unit("m"))
    assert q.value == 2.0


@pytest.mark.parametrize("value", [1, 1.5, 2 + 3j, Fraction(1, 2), Decimal("1.5")])
def test_accepts_numeric_types(value: object) -> None:
    assert Quantity(value, "m").value == value  # type: ignore[arg-type]


def test_rejects_bad_value_and_unit() -> None:
    with pytest.raises(TypeError, match="value must be"):
        Quantity("x", "m")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be"):
        Quantity(True, "m")
    with pytest.raises(TypeError, match="unit must be"):
        Quantity(1.0, 3)  # type: ignore[arg-type]


def test_immutable() -> None:
    q = Quantity(1.0, "m")
    with pytest.raises(AttributeError):
        q.value = 2.0  # type: ignore[misc]


def test_to_linear() -> None:
    assert Quantity(1.0, "eV").to("J").value == 1.602176634e-19
    assert Quantity(2.0, "m").value_in("cm") == 200.0


def test_to_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        Quantity(1.0, "m").to("s")


def test_to_unknown_equivalence() -> None:
    with pytest.raises(ValueError, match="unknown equivalence"):
        Quantity(1.0, "kJ/mol").to("J", equivalence="spectral")


def test_to_si() -> None:
    q = Quantity(1.0, "kJ/mol").to_si()
    assert str(q.unit) == "kg·m²·s⁻²·mol⁻¹"
    assert q.value == 1000.0


def test_affine_conversion_values() -> None:
    assert Quantity(25.0, "degC").to("K").value == pytest.approx(298.15)
    assert Quantity(0.0, "degC").to("degF").value == pytest.approx(32.0)
    assert Quantity(212.0, "degF").to("degC").value == pytest.approx(100.0)


def test_addition_converts_to_left_unit() -> None:
    result = Quantity(2.0, "m") + Quantity(30.0, "cm")
    assert result.value == pytest.approx(2.3)
    assert str(result.unit) == "m"


def test_addition_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        Quantity(1.0, "m") + Quantity(1.0, "s")


def test_plain_number_only_with_dimensionless() -> None:
    assert (Quantity(0.5, "1") + 0.25).value == 0.75
    assert (1.0 + Quantity(0.5, "1")).value == 1.5
    assert Quantity(1.0, "m").__add__(2.0) is NotImplemented


def test_affine_matrix() -> None:
    # °C + Δ°C is allowed and stays °C.
    warmer = Quantity(20.0, "degC") + Quantity(5.0, "delta_degC")
    assert warmer.value == pytest.approx(25.0)
    assert str(warmer.unit) == "°C"
    # °C - °C is a temperature difference in kelvin.
    diff = Quantity(20.0, "degC") - Quantity(15.0, "degC")
    assert diff.value == pytest.approx(5.0)
    assert str(diff.unit) == "K"
    # °C + °C is ambiguous and rejected.
    with pytest.raises(duq.AffineUnitError):
        Quantity(20.0, "degC") + Quantity(20.0, "degC")
    # relative + absolute is rejected.
    with pytest.raises(duq.AffineUnitError):
        Quantity(5.0, "K") + Quantity(20.0, "degC")


def test_molar_equivalence_roundtrip() -> None:
    per_particle = Quantity(1.0, "kJ/mol").to("J", equivalence="molar")
    assert per_particle.value == pytest.approx(1000 / 6.02214076e23)
    back = per_particle.to("kJ/mol", equivalence="molar")
    assert back.value == pytest.approx(1.0)


def test_molar_equivalence_refuses_non_amount_difference() -> None:
    with pytest.raises(duq.DimensionalityError):
        Quantity(1.0, "J").to("m", equivalence="molar")


def test_molar_equivalence_rejects_affine_units() -> None:
    # Regression: the affine offset was silently dropped (25 degC "became" 25 K).
    with pytest.raises(duq.AffineUnitError, match="molar equivalence is undefined"):
        Quantity(25.0, "degC").to("K", equivalence="molar")
    with pytest.raises(duq.AffineUnitError, match="molar equivalence is undefined"):
        Quantity(298.15, "K").to("degC", equivalence="molar")


def test_molar_equivalence_with_zero_k_is_plain_conversion() -> None:
    converted = Quantity(1.0, "km").to("m", equivalence="molar")
    assert converted.value == 1000.0
    assert converted == Quantity(1.0, "km").to("m")


def test_multiplication_and_division() -> None:
    energy = Quantity(2.0, "kJ/mol") * Quantity(3.0, "mol")
    assert energy.value == 6.0
    assert energy.dimension == duq.dimension("energy")
    assert (Quantity(6.0, "m") / Quantity(2.0, "s")).value == 3.0


def test_scalar_multiplication_and_reflected() -> None:
    assert (3 * Quantity(2.0, "m")).value == 6.0
    assert (Quantity(6.0, "m") / 2).value == 3.0
    assert (2.0 / Quantity(4.0, "s")).value == 0.5
    assert str((2.0 / Quantity(4.0, "s")).unit) == "s⁻¹"


def test_bool_operands_rejected() -> None:
    assert Quantity(1.0, "m").__mul__(True) is NotImplemented
    assert Quantity(1.0, "m").__truediv__(True) is NotImplemented
    assert Quantity(1.0, "m").__rtruediv__(True) is NotImplemented
    assert Quantity(1.0, "m").__pow__(True) is NotImplemented


def test_power() -> None:
    q = Quantity(2.0, "m") ** 2
    assert q.value == 4.0
    assert str(q.unit) == "m²"


def test_negation_and_abs() -> None:
    assert (-Quantity(2.0, "m")).value == -2.0
    assert (+Quantity(2.0, "m")).value == 2.0
    assert abs(Quantity(-2.0, "m")).value == 2.0


def test_scaling_affine_rejected() -> None:
    with pytest.raises(duq.AffineUnitError):
        2 * Quantity(20.0, "degC")
    with pytest.raises(duq.AffineUnitError):
        abs(Quantity(-5.0, "degC"))
    with pytest.raises(duq.AffineUnitError):
        1 / Quantity(20.0, "degC")


def test_compact() -> None:
    assert str(Quantity(1500.0, "m").compact()) == "1.5 km"
    assert str(Quantity(0.0025, "m").compact().unit) == "mm"
    # zero, dimensionless, and non-prefixable are unchanged.
    assert Quantity(0.0, "m").compact().value == 0.0
    assert Quantity(5.0, "1").compact().value == 5.0
    assert str(Quantity(3.0, "min").compact().unit) == "min"


def test_equality_exact_and_hash() -> None:
    assert Quantity(1.0, "km") == Quantity(1000.0, "m")
    assert hash(Quantity(1.0, "km")) == hash(Quantity(1000.0, "m"))
    assert Quantity(1.0, "m") != Quantity(2.0, "m")


def test_equality_incompatible_is_false() -> None:
    assert Quantity(1.0, "m") != Quantity(1.0, "s")
    assert Quantity(1.0, "m").__eq__(3) is NotImplemented


def test_ordering() -> None:
    assert Quantity(1.0, "m") < Quantity(200.0, "cm")
    assert Quantity(1.0, "m") <= Quantity(100.0, "cm")
    assert Quantity(2.0, "m") > Quantity(100.0, "cm")
    assert Quantity(1.0, "m") >= Quantity(100.0, "cm")
    assert Quantity(1.0, "m").__lt__(3) is NotImplemented


def test_ordering_incompatible_raises() -> None:
    with pytest.raises(duq.DimensionalityError):
        _ = Quantity(1.0, "m") < Quantity(1.0, "s")


def test_allclose() -> None:
    assert Quantity(1.0, "J").allclose(Quantity(1.0000000001, "J"))
    assert not Quantity(1.0, "J").allclose(Quantity(2.0, "J"))
    with pytest.raises(duq.DimensionalityError):
        Quantity(1.0, "J").allclose(Quantity(1.0, "s"))


def test_decimal_and_complex_values() -> None:
    assert Quantity(Decimal("1"), "m").to("cm").value == Decimal("100")
    q = Quantity(2 + 0j, "m") * Quantity(3 + 0j, "s")
    assert q.value == 6 + 0j


def test_repr_and_str() -> None:
    q = Quantity(1.5, "kJ/mol")
    assert repr(q) == "Quantity(1.5, Unit('kJ.mol^-1'))"
    assert str(q) == "1.5 kJ·mol⁻¹"


def test_uconvert_ustrip() -> None:
    assert duq.uconvert("cm", Quantity(1.0, "m")).value == 100.0
    assert duq.ustrip("cm", Quantity(1.0, "m")) == 100.0


def test_array_hooks_raise() -> None:
    q = Quantity(1.0, "m")
    with pytest.raises(duq.UnsupportedOperationError):
        q.__array_ufunc__(None, "__call__")
    with pytest.raises(duq.UnsupportedOperationError):
        q.__array_function__(None, (), (), {})


@given(q=sd.quantities(), unit=sd.units())
def test_conversion_roundtrip(q: Quantity, unit: duq.Unit) -> None:
    if q.dimension != unit.dimension:
        return
    back = q.to(unit).to(q.unit)
    assert back.allclose(q, rel_tol=1e-7, abs_tol=1e-12)
