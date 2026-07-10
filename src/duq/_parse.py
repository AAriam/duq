"""Tokeniser for unit and dimension composition strings.

A single hand-written parser understands both the *dot* style
(``kg.m^2.s^-2``) and the *star* style (``kg*m**2/s**2``, ``kJ/mol``).  It
returns a flat, order-preserving list of ``(name, exponent)`` terms in which
division has been folded into the sign of the exponent.  The caller
(:class:`~duq._dimension.Dimension` or :class:`~duq._registry.UnitRegistry`)
maps each ``name`` onto a base dimension or unit atom.

Grammar (informally)
--------------------
* Factors are separated by multiplication (``.``, ``·`` or ``*``) or division
  (``/``).  Division negates the exponent of the factor (or parenthesised
  group) that immediately follows it.
* An exponent is introduced by ``^`` or ``**`` and is either a parenthesised
  signed fraction (``(3/2)``) or a signed integer optionally followed by
  ``/<digits>`` (``3/2``).  Thus ``m^3/2`` is ``m`` to the power ``3/2`` while
  ``m^2/s^2`` is ``m²·s⁻²`` -- the ``/`` is only part of the exponent when a
  digit follows it.
* Unicode superscripts (``m²``, ``s⁻¹``) are accepted for integer exponents.
* A leading ``1`` is the dimensionless identity; ``1/s`` is ``s⁻¹``.
* Any other numeric literal is illegal.
"""

from __future__ import annotations

from fractions import Fraction

from ._errors import UnitParseError

__all__ = ("Term", "tokenize")

Term = tuple[str, Fraction]
"""A single parsed factor: its ``name`` and signed exponent."""

_SUPERSCRIPT_TO_ASCII = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
}

# Characters (beyond ``str.isalpha``) that may appear inside a unit/dimension
# name: the degree sign, percent sign, and underscore (e.g. ``Δ°C``, ``%``).
_NAME_EXTRA = frozenset("°%_")

_MULTIPLY = frozenset(".·*")
_WHITESPACE = frozenset(" \t\n\r")


class _Cursor:
    """A position-tracking cursor over the source string."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.i = 0
        self.n = len(source)

    def eof(self) -> bool:
        """Return whether the cursor is at or past the end of the input."""
        return self.i >= self.n

    def peek(self) -> str:
        """Return the current character, or ``""`` at end of input."""
        return "" if self.i >= self.n else self.source[self.i]

    def peek2(self) -> str:
        """Return the next-but-one character, or ``""`` if unavailable."""
        j = self.i + 1
        return "" if j >= self.n else self.source[j]

    def skip_ws(self) -> None:
        """Advance past any run of whitespace."""
        while self.i < self.n and self.source[self.i] in _WHITESPACE:
            self.i += 1

    def fail(self, message: str) -> UnitParseError:
        """Build a :class:`UnitParseError` annotated with the position."""
        return UnitParseError(f"{message} at position {self.i} in {self.source!r}")


def _read_digits(cur: _Cursor) -> int:
    """Read a run of ASCII digits, or raise if none are present."""
    start = cur.i
    while cur.i < cur.n and cur.source[cur.i].isdigit():
        cur.i += 1
    if cur.i == start:
        raise cur.fail("expected a digit")
    return int(cur.source[start : cur.i])


def _read_signed_int_or_fraction(cur: _Cursor) -> Fraction:
    """Read ``[+-]?digits(/digits)?`` as an exact :class:`~fractions.Fraction`."""
    sign = 1
    if cur.peek() == "-":
        sign = -1
        cur.i += 1
    elif cur.peek() == "+":
        cur.i += 1
    numerator = _read_digits(cur)
    if cur.peek() == "/" and cur.peek2().isdigit():
        cur.i += 1
        denominator = _read_digits(cur)
        return Fraction(sign * numerator, denominator)
    return Fraction(sign * numerator)


def _read_exponent(cur: _Cursor) -> Fraction:
    """Read an optional exponent following a factor; default is ``1``."""
    cur.skip_ws()
    ch = cur.peek()
    if ch in _SUPERSCRIPT_TO_ASCII:
        start = cur.i
        while cur.i < cur.n and cur.source[cur.i] in _SUPERSCRIPT_TO_ASCII:
            cur.i += 1
        ascii_exp = "".join(_SUPERSCRIPT_TO_ASCII[c] for c in cur.source[start : cur.i])
        try:
            return Fraction(int(ascii_exp))
        except ValueError:
            raise cur.fail("invalid superscript exponent") from None
    if ch == "^" or (ch == "*" and cur.peek2() == "*"):
        cur.i += 2 if ch == "*" else 1
        cur.skip_ws()
        if cur.peek() == "(":
            cur.i += 1
            cur.skip_ws()
            exp = _read_signed_int_or_fraction(cur)
            cur.skip_ws()
            if cur.peek() != ")":
                raise cur.fail("expected ')' to close exponent")
            cur.i += 1
            return exp
        return _read_signed_int_or_fraction(cur)
    return Fraction(1)


def _read_name(cur: _Cursor) -> str:
    """Read a unit/dimension name token."""
    start = cur.i
    while cur.i < cur.n:
        ch = cur.source[cur.i]
        if ch.isalpha() or ch in _NAME_EXTRA:
            cur.i += 1
        else:
            break
    if cur.i == start:
        raise cur.fail("expected a unit or dimension name")
    return cur.source[start : cur.i]


def _read_operand(cur: _Cursor, sign: int) -> list[Term]:
    """Read a single factor or a parenthesised group, applying ``sign``."""
    cur.skip_ws()
    if cur.peek() == "(":
        cur.i += 1
        depth = 1
        start = cur.i
        while cur.i < cur.n and depth > 0:
            if cur.source[cur.i] == "(":
                depth += 1
            elif cur.source[cur.i] == ")":
                depth -= 1
                if depth == 0:
                    break
            cur.i += 1
        if depth != 0:
            raise cur.fail("unbalanced '('")
        inner = cur.source[start : cur.i]
        cur.i += 1  # consume ')'
        group_exp = _read_exponent(cur)
        factor = sign * group_exp
        return [(name, exp * factor) for name, exp in tokenize(inner)]
    name = _read_name(cur)
    exp = _read_exponent(cur)
    return [(name, sign * exp)]


def tokenize(source: str) -> list[Term]:
    """Parse a unit or dimension string into ordered ``(name, exponent)`` terms.

    Parameters
    ----------
    source : str
        The composition string, e.g. ``"kg.m^2.s^-2"`` or ``"kJ/mol"``.

    Returns
    -------
    list of (str, fractions.Fraction)
        The parsed terms, in as-entered order, with division folded into the
        exponent sign.  An empty list denotes the dimensionless identity.

    Raises
    ------
    UnitParseError
        If the string is empty or malformed.

    Examples
    --------
    >>> tokenize("kg.m^2.s^-2")
    [('kg', Fraction(1, 1)), ('m', Fraction(2, 1)), ('s', Fraction(-2, 1))]
    >>> tokenize("kJ/mol")
    [('kJ', Fraction(1, 1)), ('mol', Fraction(-1, 1))]
    >>> tokenize("1/s")
    [('s', Fraction(-1, 1))]
    >>> tokenize("m^3/2")
    [('m', Fraction(3, 2))]
    """
    cur = _Cursor(source)
    cur.skip_ws()
    if cur.eof():
        raise cur.fail("empty unit/dimension string")

    terms: list[Term] = []
    expect_operator: bool

    if cur.peek() == "1" and not cur.peek2().isdigit():
        cur.i += 1
        cur.skip_ws()
        if cur.eof():
            return []
        if cur.peek() not in ("/", *_MULTIPLY):
            raise cur.fail("unexpected token after leading '1'")
        expect_operator = True
    else:
        terms.extend(_read_operand(cur, 1))
        expect_operator = True

    while True:
        cur.skip_ws()
        if cur.eof():
            break
        if not expect_operator:  # pragma: no cover - defensive
            raise cur.fail("expected an operator")
        ch = cur.peek()
        if ch in _MULTIPLY:
            sign = 1
        elif ch == "/":
            sign = -1
        else:
            raise cur.fail("expected '.', '*', '/' or end of input")
        cur.i += 1
        terms.extend(_read_operand(cur, sign))

    return terms
