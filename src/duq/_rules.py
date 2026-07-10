"""Declarative unit-rule engine shared across the scalar, NumPy and JAX layers.

Each abstract operation maps to a rule that, given the input :class:`Dimension`
objects, returns the output :class:`Dimension` (or raises
:class:`DimensionalityError`).  The scalar :class:`~duq._quantity.Quantity`
consumes these rules today; the NumPy and JAX front-ends added in later releases
reuse the very same table, so the *physics* of dimensional analysis is written
once.

The engine is pure data plus small functions -- it never imports NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from fractions import Fraction

from ._dimension import Dimension
from ._errors import DimensionalityError

__all__ = ("Rule", "apply", "result_dimension")

_DIMENSIONLESS = Dimension({})


class Rule(Enum):
    """Abstract dimensional rules keyed by operation family."""

    SAME_DIM = "same_dim"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    DIMENSIONLESS_IN_DIMENSIONLESS_OUT = "dimensionless_in_dimensionless_out"
    ANGLE_IN = "angle_in"
    PRESERVE = "preserve"
    SQRT = "sqrt"
    CBRT = "cbrt"
    DIMENSIONLESS_OUT = "dimensionless_out"


def _same_dim(a: Dimension, b: Dimension) -> Dimension:
    if a != b:
        raise DimensionalityError(f"operands must share a dimension: {a!r} vs {b!r}")
    return a


def _multiply(a: Dimension, b: Dimension) -> Dimension:
    return a * b


def _divide(a: Dimension, b: Dimension) -> Dimension:
    return a / b


def _power(a: Dimension, exponent: Fraction) -> Dimension:
    return a**exponent


def _require_dimensionless(a: Dimension) -> Dimension:
    if not a.is_dimensionless:
        raise DimensionalityError(f"expected a dimensionless operand, got {a!r}")
    return _DIMENSIONLESS


def _preserve(a: Dimension) -> Dimension:
    return a


def _sqrt(a: Dimension) -> Dimension:
    return a ** Fraction(1, 2)


def _cbrt(a: Dimension) -> Dimension:
    return a ** Fraction(1, 3)


def _dimensionless_out(a: Dimension) -> Dimension:
    return _DIMENSIONLESS


_TABLE: dict[Rule, Callable[..., Dimension]] = {
    Rule.SAME_DIM: _same_dim,
    Rule.MULTIPLY: _multiply,
    Rule.DIVIDE: _divide,
    Rule.POWER: _power,
    Rule.DIMENSIONLESS_IN_DIMENSIONLESS_OUT: _require_dimensionless,
    Rule.ANGLE_IN: _require_dimensionless,
    Rule.PRESERVE: _preserve,
    Rule.SQRT: _sqrt,
    Rule.CBRT: _cbrt,
    Rule.DIMENSIONLESS_OUT: _dimensionless_out,
}


def apply(rule: Rule, *operands: Dimension | Fraction) -> Dimension:
    """Apply a rule to its operands and return the output dimension.

    Parameters
    ----------
    rule : Rule
        The abstract operation to apply.
    *operands : Dimension or fractions.Fraction
        The operands.  Most rules take :class:`Dimension` operands; ``POWER``
        additionally takes a :class:`~fractions.Fraction` exponent.

    Returns
    -------
    Dimension
        The resulting dimension.

    Raises
    ------
    DimensionalityError
        If the operands are dimensionally incompatible with the rule.

    Examples
    --------
    >>> from duq._dimension import Dimension
    >>> length = Dimension({"L": 1})
    >>> apply(Rule.MULTIPLY, length, length) == Dimension({"L": 2})
    True
    """
    return _TABLE[rule](*operands)


def result_dimension(rule: Rule, *operands: Dimension | Fraction) -> Dimension:
    """Alias for :func:`apply`, spelled for the array front-ends.

    Parameters
    ----------
    rule : Rule
        The abstract operation to apply.
    *operands : Dimension or fractions.Fraction
        The operands.

    Returns
    -------
    Dimension
        The resulting dimension.

    Examples
    --------
    >>> from duq._dimension import Dimension
    >>> result_dimension(Rule.PRESERVE, Dimension({"M": 1})) == Dimension({"M": 1})
    True
    """
    return apply(rule, *operands)
