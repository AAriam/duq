"""Tests for :class:`duq.UnitRegistry`."""

from __future__ import annotations

from fractions import Fraction

import pytest

import duq
from duq import Dimension, UnitRegistry


def test_prefix_longest_match() -> None:
    assert duq.unit("dam").scale == Fraction(10)  # deca + metre, not deci + "am"
    assert duq.unit("dm").scale == Fraction(1, 10)  # deci + metre


def test_micro_greek_normalised() -> None:
    assert duq.unit("μm") == duq.unit("µm")


def test_non_prefixable_atom_rejects_prefix() -> None:
    with pytest.raises(duq.UndefinedUnitError):
        duq.unit("katm")


def test_undefined_unit() -> None:
    with pytest.raises(duq.UndefinedUnitError, match="unknown unit"):
        duq.unit("smoot")


def test_registry_mismatch() -> None:
    reg_a = UnitRegistry()
    reg_b = UnitRegistry()
    with pytest.raises(duq.RegistryMismatchError):
        reg_a.unit("m") * reg_b.unit("s")


def test_unit_of_unit_checks_registry() -> None:
    reg_a = UnitRegistry()
    reg_b = UnitRegistry()
    metre = reg_a.unit("m")
    assert reg_a.unit(metre) is metre
    with pytest.raises(duq.UndefinedUnitError, match="different registry"):
        reg_b.unit(metre)


def test_define_custom_atom() -> None:
    reg = UnitRegistry()
    reg.define("smoot", "smoot", "length", 1.702, aliases=("Smoot",))
    assert reg.unit("smoot").dimension == Dimension({"L": 1})
    assert reg.unit("Smoot").scale == 1.702


def test_define_with_dimension_object() -> None:
    reg = UnitRegistry()
    reg.define("widget", "wdg", Dimension({"N": 1}), Fraction(2))
    assert reg.unit("wdg").scale == Fraction(2)


def test_atoms_iteration() -> None:
    slugs = {atom.slug for atom in duq.default_registry.atoms}
    assert {"metre", "joule", "electronvolt", "celsius"} <= slugs


def test_si_prefix_lookup() -> None:
    reg = UnitRegistry()
    assert reg.si_prefix(3).symbol == "k"  # type: ignore[union-attr]
    assert reg.si_prefix(0) is None
    assert reg.si_prefix(7) is None


def test_coherent_unit() -> None:
    reg = UnitRegistry()
    coherent = reg.coherent_unit(Dimension({"M": 1, "L": 2, "T": -2}))
    assert str(coherent) == "kg·m²·s⁻²"
    assert coherent.scale == Fraction(1)


def test_empty_registry_without_autoload() -> None:
    reg = UnitRegistry(autoload=False)
    reg.define("foo", "foo", Dimension({"L": 1}), 1)
    assert reg.unit("foo").scale == Fraction(1)
    with pytest.raises(duq.UndefinedUnitError):
        reg.unit("m")
