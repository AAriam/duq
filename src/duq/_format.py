r"""Rendering of unit and dimension compositions to text.

Three output styles are supported:

``"unicode"``
    Uses a middle dot (``·``) as the multiplication separator and Unicode
    superscripts for exponents (``kJ·mol⁻¹``).  Fractional exponents fall back
    to ``^p/q``.
``"plain"``
    ASCII only, using ``.`` and ``^`` (``kJ.mol^-1``).
``"latex"``
    A minimal LaTeX rendering (``\\mathrm{kJ}\\,\\mathrm{mol}^{-1}``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction

__all__ = ("format_composition", "superscript")

_SUPERSCRIPT_DIGITS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
}

Style = str
"""Type alias for the accepted formatting styles."""

_STYLES: tuple[str, ...] = ("unicode", "plain", "latex")


def superscript(value: int) -> str:
    """Render an integer as a Unicode superscript string.

    Parameters
    ----------
    value : int
        The integer to render.

    Returns
    -------
    str
        The Unicode superscript form, e.g. ``-2`` becomes ``"⁻²"``.

    Examples
    --------
    >>> superscript(-2)
    '⁻²'
    >>> superscript(31)
    '³¹'
    """
    return "".join(_SUPERSCRIPT_DIGITS[ch] for ch in str(value))


def _format_exponent(exp: Fraction, style: Style) -> str:
    """Render a single exponent in the requested style."""
    if exp == 1:
        return ""
    if exp.denominator == 1:
        n = exp.numerator
        if style == "unicode":
            return superscript(n)
        if style == "latex":
            return f"^{{{n}}}"
        return f"^{n}"
    frac = f"{exp.numerator}/{exp.denominator}"
    if style == "latex":
        return f"^{{{frac}}}"
    return f"^{frac}"


def format_composition(
    terms: Iterable[tuple[str, Fraction]],
    *,
    style: Style = "unicode",
    empty: str = "1",
) -> str:
    """Render an ordered composition of ``(symbol, exponent)`` terms.

    Parameters
    ----------
    terms : iterable of (str, fractions.Fraction)
        The composition, in display order.  Each item pairs a display symbol
        with its exponent.
    style : {"unicode", "plain", "latex"}, optional
        The output style (default ``"unicode"``).
    empty : str, optional
        The string to return when ``terms`` is empty (default ``"1"``).

    Returns
    -------
    str
        The rendered composition.

    Examples
    --------
    >>> from fractions import Fraction
    >>> format_composition([("kJ", Fraction(1)), ("mol", Fraction(-1))])
    'kJ·mol⁻¹'
    >>> format_composition([("m", Fraction(1))], style="plain", empty="1")
    'm'
    >>> format_composition([], empty="dimensionless")
    'dimensionless'
    """
    if style not in _STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {_STYLES}")
    parts: list[str] = []
    for symbol, exp in terms:
        sym = f"\\mathrm{{{symbol}}}" if style == "latex" else symbol
        parts.append(f"{sym}{_format_exponent(exp, style)}")
    if not parts:
        return empty
    sep = {"unicode": "·", "plain": ".", "latex": r"\,"}[style]
    return sep.join(parts)


def order_terms(
    terms: Sequence[tuple[str, Fraction]],
) -> list[tuple[str, Fraction]]:
    """Return ``terms`` with positive exponents before negative ones.

    Positive (and zero) exponents keep their relative order and precede the
    negative exponents, which also keep their relative order.  This mirrors the
    conventional ``a·b·c⁻¹`` layout while preserving as-entered ordering within
    each group.

    Parameters
    ----------
    terms : sequence of (str, fractions.Fraction)
        The composition to reorder.

    Returns
    -------
    list of (str, fractions.Fraction)
        The reordered composition.

    Examples
    --------
    >>> from fractions import Fraction
    >>> order_terms([("s", Fraction(-2)), ("m", Fraction(1))])
    [('m', Fraction(1, 1)), ('s', Fraction(-2, 1))]
    """
    positive = [t for t in terms if t[1] >= 0]
    negative = [t for t in terms if t[1] < 0]
    return [*positive, *negative]
