"""Tests for the public package surface."""

from __future__ import annotations

import pytest

import duq


def test_version_is_a_string() -> None:
    assert isinstance(duq.__version__, str)


def test_all_names_are_resolvable() -> None:
    for name in duq.__all__:
        assert hasattr(duq, name)


def test_unit_and_dimension_helpers() -> None:
    assert duq.unit("kg.m/s^2") == duq.unit("N")
    assert duq.dimension("force") == duq.dimension("M.L.T^-2")


def test_units_namespace() -> None:
    assert str(duq.units.kJ) == "kJ"
    assert duq.units.metre == duq.unit("m")
    assert duq.units.joule == duq.unit("J")  # slug fallback


def test_units_namespace_unknown() -> None:
    with pytest.raises(AttributeError):
        _ = duq.units.definitely_not_a_unit
    with pytest.raises(AttributeError):
        _ = duq.units._private


def test_dims_namespace() -> None:
    assert duq.dims.energy == duq.dimension("M.L^2.T^-2")
    assert duq.dims.mass == duq.dimension("mass")


def test_dims_namespace_unknown() -> None:
    with pytest.raises(AttributeError):
        _ = duq.dims.not_a_dimension
    with pytest.raises(AttributeError):
        _ = duq.dims._private


def test_dir_of_units_lists_slugs() -> None:
    assert "joule" in dir(duq.units)
