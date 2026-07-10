"""Exception hierarchy for :mod:`duq`.

All exceptions raised by ``duq`` derive from :class:`DuqError`, so a single
``except duq.DuqError`` catches every library-specific failure.  Each concrete
error also inherits from a matching built-in (``ValueError``, ``KeyError``,
``TypeError``, ``RuntimeError``) so that code written against the built-ins keeps
working.
"""

from __future__ import annotations

__all__ = (
    "AffineUnitError",
    "AnalysisError",
    "DimensionalityError",
    "DuqError",
    "RegistryMismatchError",
    "UndefinedUnitError",
    "UnitParseError",
    "UnsupportedOperationError",
)


class DuqError(Exception):
    """Base class for every exception raised by :mod:`duq`.

    Examples
    --------
    >>> import duq
    >>> try:
    ...     duq.unit("not-a-unit")
    ... except duq.DuqError as exc:
    ...     print(type(exc).__name__)
    UnitParseError
    """


class UnitParseError(DuqError, ValueError):
    """Raised when a unit or dimension string cannot be parsed.

    The message includes the character position at which parsing failed.

    Examples
    --------
    >>> import duq
    >>> try:
    ...     duq.unit("kg..m")
    ... except duq.UnitParseError as exc:
    ...     print("failed")
    failed
    """


class UndefinedUnitError(DuqError, KeyError):
    """Raised when a referenced unit atom or prefix is not in the registry.

    Examples
    --------
    >>> import duq
    >>> try:
    ...     duq.unit("smoot")
    ... except duq.UndefinedUnitError:
    ...     print("undefined")
    undefined
    """


class DimensionalityError(DuqError, ValueError):
    """Raised when an operation combines incompatible dimensions.

    Examples
    --------
    >>> import duq
    >>> metre = duq.Quantity(1.0, "m")
    >>> second = duq.Quantity(1.0, "s")
    >>> try:
    ...     metre + second
    ... except duq.DimensionalityError:
    ...     print("incompatible")
    incompatible
    """


class AffineUnitError(DimensionalityError):
    """Raised for illegal use of an affine unit such as ``degC``.

    Affine units may only appear alone with exponent one, and may not be
    freely combined in arithmetic.

    Examples
    --------
    >>> import duq
    >>> try:
    ...     duq.unit("degC^2")
    ... except duq.AffineUnitError:
    ...     print("affine")
    affine
    """


class RegistryMismatchError(DuqError, ValueError):
    """Raised when units from different registries are combined.

    Examples
    --------
    >>> import duq
    >>> reg_a = duq.UnitRegistry()
    >>> reg_b = duq.UnitRegistry()
    >>> try:
    ...     reg_a.unit("m") * reg_b.unit("s")
    ... except duq.RegistryMismatchError:
    ...     print("mismatch")
    mismatch
    """


class UnsupportedOperationError(DuqError, TypeError):
    """Raised when an operation is not supported by the scalar core.

    NumPy array support arrives in a later release; until then the array
    protocol hooks raise this error.

    Examples
    --------
    >>> import duq
    >>> UnsupportedOperationError = duq.UnsupportedOperationError
    >>> issubclass(UnsupportedOperationError, TypeError)
    True
    """


class AnalysisError(DuqError, RuntimeError):
    """Raised when a dimensional-analysis search cannot return an exact result.

    Examples
    --------
    >>> import duq
    >>> issubclass(duq.AnalysisError, RuntimeError)
    True
    """
