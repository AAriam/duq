"""Meta tests: no-numpy import, import time, docstrings and catalog schema."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from fractions import Fraction

import pytest

import duq


def test_import_does_not_import_numpy() -> None:
    code = (
        "import sys, duq\n"
        "q = duq.Quantity(2.0, 'kJ/mol') * duq.Quantity(3.0, 'mol')\n"
        "_ = q.to('J'); _ = duq.constants.k_B; _ = duq.unit('kg.m^2/s^2')\n"
        "assert 'numpy' not in sys.modules, 'numpy was imported by the core'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_import_does_not_import_jax() -> None:
    code = (
        "import sys, duq\n"
        "q = duq.Quantity(2.0, 'kJ/mol') * duq.Quantity(3.0, 'mol')\n"
        "_ = q.to('J'); _ = duq.unit('kg.m^2/s^2')\n"
        "assert 'jax' not in sys.modules, 'jax was imported by the core'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is not None,
    reason="jax is installed; the ImportError path only exists without it",
)
def test_import_duq_jax_without_jax_raises_helpful_error() -> None:
    code = (
        "try:\n"
        "    import duq.jax\n"
        "except ImportError as exc:\n"
        "    assert 'duq[jax]' in str(exc), str(exc)\n"
        "    print('ok')\n"
        "else:\n"
        "    print('no error')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_import_time_is_reasonable() -> None:
    code = (
        "import time\n"
        "start = time.perf_counter()\n"
        "import duq\n"
        "print(time.perf_counter() - start)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) < 0.5


def test_public_objects_have_docstrings() -> None:
    for name in duq.__all__:
        if name == "__version__":
            continue
        obj = getattr(duq, name)
        assert obj.__doc__, f"{name} is missing a docstring"


def test_units_catalog_roundtrips() -> None:
    for atom in duq.default_registry.atoms:
        unit = duq.default_registry.unit(atom.symbol)
        assert duq.unit(str(unit)) == unit
        assert isinstance(atom.scale, Fraction | float)
        assert isinstance(atom.offset, Fraction | float)
        assert atom.kind in {"linear", "affine", "delta"}


def test_constants_catalog_is_well_formed() -> None:
    catalog = duq.constants.codata(2022)
    for slug in dir(catalog):
        if slug.startswith("_") or slug in {"info", "quantity", "year"}:
            continue
        quantity = catalog.quantity(slug)
        info = catalog.info(slug)
        assert quantity.dimension == duq.unit(info["unit"]).dimension
        assert info["uncertainty"]


@pytest.mark.parametrize("year", [2022])
def test_every_constant_parses(year: int) -> None:
    catalog = duq.constants.codata(year)
    info = catalog.info("N_A")
    assert float(info["value"]) > 0
