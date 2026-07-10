"""The :class:`UnitRegistry`: the catalogue that turns strings into units.

A registry loads the TOML catalogues lazily on first use, resolves SI prefixes
against prefixable atoms (longest-match, exact-atom-wins), and can be extended
at runtime with :meth:`UnitRegistry.define`.  A single module-level default
registry backs :func:`duq.unit`, the :mod:`duq.units` namespace and all string
parsing; units from different registries refuse to combine.
"""

from __future__ import annotations

import importlib.resources
import tomllib
from collections.abc import Iterator, Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Final

from ._dimension import Dimension
from ._errors import RegistryMismatchError, UndefinedUnitError
from ._parse import tokenize
from ._unit import Factor, Prefix, Scalar, Unit, UnitAtom

__all__ = ("UnitRegistry", "default_registry")

# Display order for coherent-SI reconstruction (mass, length, time first).
_COHERENT_ORDER: Final[tuple[str, ...]] = ("M", "L", "T", "I", "Θ", "N", "J")
_BASE_ATOM_BY_SYMBOL: Final[dict[str, str]] = {
    "T": "s",
    "L": "m",
    "M": "kg",
    "I": "A",
    "Θ": "K",
    "N": "mol",
    "J": "cd",
}
_MICRO_GREEK = "μ"
_MICRO_SIGN = "µ"


def _parse_scalar(text: str, *, exact: bool) -> Scalar:
    """Parse a scale/offset string to an exact Fraction or a float."""
    if not exact:
        return float(text)
    try:
        return Fraction(text)
    except ValueError:
        return Fraction(Decimal(text))


def _load_toml(name: str) -> dict[str, object]:
    """Load a TOML data file shipped inside :mod:`duq.data`."""
    resource = importlib.resources.files("duq.data").joinpath(name)
    with resource.open("rb") as handle:
        return tomllib.load(handle)


class UnitRegistry:
    """A catalogue of unit atoms and prefixes that parses unit strings.

    Parameters
    ----------
    autoload : bool, optional
        When ``True`` (the default), the bundled catalogues are loaded on the
        first lookup.  Pass ``False`` to build an empty registry populated only
        via :meth:`define`.

    Examples
    --------
    >>> reg = UnitRegistry()
    >>> str(reg.unit("kN.m"))
    'kN·m'
    >>> reg.unit("km").scale
    Fraction(1000, 1)
    """

    def __init__(self, *, autoload: bool = True) -> None:
        self._autoload = autoload
        self._loaded = False
        self._atoms: dict[str, UnitAtom] = {}
        self._by_symbol: dict[str, UnitAtom] = {}
        self._by_alias: dict[str, UnitAtom] = {}
        self._prefixes: list[Prefix] = []
        self._prefix_by_exp: dict[int, Prefix] = {}

    # -- loading ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._autoload:
            self._load_catalogs()

    def _load_catalogs(self) -> None:
        prefixes = _load_toml("prefixes.toml")
        for name, entry in prefixes.items():
            assert isinstance(entry, dict)
            prefix = Prefix(
                name=name, symbol=str(entry["symbol"]), exponent10=int(entry["exponent10"])
            )
            self._prefixes.append(prefix)
            self._prefix_by_exp[prefix.exponent10] = prefix
        self._prefixes.sort(key=lambda p: len(p.symbol), reverse=True)

        units = _load_toml("units.toml")
        catalog = units["units"]
        assert isinstance(catalog, dict)
        for slug, entry in catalog.items():
            assert isinstance(entry, dict)
            self._register_atom(slug, entry)

    def _register_atom(self, slug: str, entry: Mapping[str, object]) -> None:
        if "dimension" in entry:
            dim = Dimension.parse(str(entry["dimension"]))
        else:
            raw = entry["exponents"]
            assert isinstance(raw, dict)
            dim = Dimension({str(k): v for k, v in raw.items()})
        exact = bool(entry.get("exact", False))
        scale = _parse_scalar(str(entry["scale"]), exact=exact)
        offset = _parse_scalar(str(entry.get("offset", "0")), exact=exact)
        raw_aliases = entry.get("aliases", ())
        assert isinstance(raw_aliases, list | tuple)
        aliases = tuple(str(a) for a in raw_aliases)
        atom = UnitAtom(
            slug=slug,
            symbol=str(entry["symbol"]),
            aliases=aliases,
            dimension=dim,
            scale=scale,
            offset=offset,
            prefixable=bool(entry.get("prefixable", False)),
            kind=str(entry.get("kind", "linear")),
        )
        self._add_atom(atom)

    def _add_atom(self, atom: UnitAtom) -> None:
        self._atoms[atom.slug] = atom
        self._by_symbol[atom.symbol] = atom
        for alias in atom.aliases:
            self._by_alias[alias] = atom

    # -- public API ---------------------------------------------------------

    def define(  # noqa: PLR0913 - a unit atom legitimately has many attributes
        self,
        slug: str,
        symbol: str,
        dimension: Dimension | str,
        scale: Scalar | int,
        *,
        aliases: tuple[str, ...] = (),
        offset: Scalar | int = 0,
        prefixable: bool = False,
        kind: str = "linear",
    ) -> UnitAtom:
        """Define a new unit atom in this registry.

        Parameters
        ----------
        slug : str
            A unique identifier for the atom.
        symbol : str
            The display symbol used when rendering the unit.
        dimension : Dimension or str
            The atom's physical dimension (parsed if a string).
        scale : fractions.Fraction, float or int
            The factor to the coherent SI unit.
        aliases : tuple of str, optional
            Alternative names accepted by the parser.
        offset : fractions.Fraction, float or int, optional
            Additive offset to coherent SI (affine units only).
        prefixable : bool, optional
            Whether SI prefixes may be attached (default ``False``).
        kind : {"linear", "affine", "delta"}, optional
            The atom kind (default ``"linear"``).

        Returns
        -------
        UnitAtom
            The newly registered atom.

        Examples
        --------
        >>> reg = UnitRegistry()
        >>> _ = reg.define("smoot", "smoot", "length", 1.702)
        >>> reg.unit("smoot").dimension == Dimension({"L": 1})
        True
        """
        self._ensure_loaded()
        dim = dimension if isinstance(dimension, Dimension) else Dimension.parse(dimension)
        atom = UnitAtom(
            slug=slug,
            symbol=symbol,
            aliases=aliases,
            dimension=dim,
            scale=Fraction(scale) if isinstance(scale, int) else scale,
            offset=Fraction(offset) if isinstance(offset, int) else offset,
            prefixable=prefixable,
            kind=kind,
        )
        self._add_atom(atom)
        return atom

    @property
    def atoms(self) -> Iterator[UnitAtom]:
        """Iterate over every atom registered in this registry."""
        self._ensure_loaded()
        return iter(self._atoms.values())

    def unit(self, expression: str | Unit) -> Unit:
        """Parse a unit expression (or return a unit unchanged).

        Parameters
        ----------
        expression : str or Unit
            The unit expression to parse, e.g. ``"kJ/mol"``.  A :class:`Unit`
            is returned unchanged (after a registry check).

        Returns
        -------
        Unit
            The parsed unit.

        Raises
        ------
        UndefinedUnitError
            If a token resolves to no known atom or prefix.
        UnitParseError
            If the expression is malformed.
        RegistryMismatchError
            If a :class:`Unit` from a different registry is passed.

        Examples
        --------
        >>> reg = UnitRegistry()
        >>> str(reg.unit("mmol/L"))
        'mmol·L⁻¹'
        """
        if isinstance(expression, Unit):
            if expression.registry is not self:
                raise RegistryMismatchError("unit belongs to a different registry")
            return expression
        self._ensure_loaded()
        factors = tuple(self._resolve(name, exp) for name, exp in tokenize(expression))
        return Unit._create(self, factors)

    def coherent_unit(self, dimension: Dimension) -> Unit:
        """Return the coherent SI unit for a dimension.

        Parameters
        ----------
        dimension : Dimension
            The target dimension.

        Returns
        -------
        Unit
            The unit built from base SI atoms with unit scale.

        Examples
        --------
        >>> reg = UnitRegistry()
        >>> str(reg.coherent_unit(Dimension({"M": 1, "L": 2, "T": -2})))
        'kg·m²·s⁻²'
        """
        self._ensure_loaded()
        exps = dimension.exponents
        factors: list[Factor] = []
        for sym in _COHERENT_ORDER:
            exp = exps[sym]
            if exp != 0:
                atom = self._by_symbol[_BASE_ATOM_BY_SYMBOL[sym]]
                factors.append(Factor(None, atom, exp))
        return Unit._create(self, tuple(factors))

    def si_prefix(self, exponent10: int) -> Prefix | None:
        """Return the prefix for a power-of-ten exponent, or ``None`` for zero.

        Parameters
        ----------
        exponent10 : int
            The power-of-ten exponent (e.g. ``3`` for kilo).

        Returns
        -------
        Prefix or None
            The matching prefix, ``None`` when ``exponent10`` is ``0``, and
            ``None`` when no prefix has that exponent.

        Examples
        --------
        >>> reg = UnitRegistry()
        >>> reg.si_prefix(3).symbol
        'k'
        >>> reg.si_prefix(0) is None
        True
        """
        self._ensure_loaded()
        if exponent10 == 0:
            return None
        return self._prefix_by_exp.get(exponent10)

    # -- resolution ---------------------------------------------------------

    def _lookup_atom(self, token: str) -> UnitAtom | None:
        return self._by_symbol.get(token) or self._by_alias.get(token)

    def _resolve(self, token: str, exponent: Fraction) -> Factor:
        norm = token.replace(_MICRO_GREEK, _MICRO_SIGN)
        atom = self._lookup_atom(norm)
        if atom is not None:
            return Factor(None, atom, exponent)
        for prefix in self._prefixes:
            symbol = prefix.symbol
            if norm.startswith(symbol) and len(norm) > len(symbol):
                rest = norm[len(symbol) :]
                candidate = self._lookup_atom(rest)
                if candidate is not None and candidate.prefixable:
                    return Factor(prefix, candidate, exponent)
        raise UndefinedUnitError(f"unknown unit {token!r}")


#: The process-wide default registry backing :func:`duq.unit` and :mod:`duq.units`.
default_registry: Final[UnitRegistry] = UnitRegistry()
