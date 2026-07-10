"""Tests for the exception hierarchy."""

from __future__ import annotations

import duq


def test_hierarchy() -> None:
    assert issubclass(duq.UnitParseError, duq.DuqError)
    assert issubclass(duq.UnitParseError, ValueError)
    assert issubclass(duq.UndefinedUnitError, KeyError)
    assert issubclass(duq.DimensionalityError, ValueError)
    assert issubclass(duq.AffineUnitError, duq.DimensionalityError)
    assert issubclass(duq.RegistryMismatchError, ValueError)
    assert issubclass(duq.UnsupportedOperationError, TypeError)
    assert issubclass(duq.AnalysisError, RuntimeError)


def test_all_derive_from_duq_error() -> None:
    for error in (
        duq.UnitParseError,
        duq.UndefinedUnitError,
        duq.DimensionalityError,
        duq.AffineUnitError,
        duq.RegistryMismatchError,
        duq.UnsupportedOperationError,
        duq.AnalysisError,
    ):
        assert issubclass(error, duq.DuqError)
