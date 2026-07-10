"""Meta tests for the JAX layer: dependency pins and coverage-doc sync."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import equinox
import quax
from packaging.requirements import Requirement
from packaging.version import Version

import duq.jax

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _jax_extra_requirements() -> dict[str, Requirement]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    requirements = [
        Requirement(spec) for spec in pyproject["project"]["optional-dependencies"]["jax"]
    ]
    return {requirement.name: requirement for requirement in requirements}


def test_installed_quax_satisfies_the_pyproject_pin() -> None:
    requirements = _jax_extra_requirements()
    assert "quax" in requirements, "the jax extra must pin quax"
    specifier = requirements["quax"].specifier
    assert specifier.contains(Version(quax.__version__)), (
        f"installed quax {quax.__version__} violates the pyproject pin {specifier}"
    )
    # the pin must carry an upper bound (single-maintainer risk, design doc)
    assert any(spec.operator in ("<", "<=", "==", "~=") for spec in specifier), (
        "quax must be pinned with an upper bound"
    )


def test_installed_equinox_satisfies_the_pyproject_pin() -> None:
    requirements = _jax_extra_requirements()
    assert "equinox" in requirements
    assert requirements["equinox"].specifier.contains(Version(equinox.__version__))


def test_primitive_coverage_is_nonempty_and_sorted() -> None:
    names = [name for name, _ in duq.jax.PRIMITIVE_COVERAGE]
    assert len(names) >= 60
    assert names == sorted(names)
    assert len(set(names)) == len(names)


def test_coverage_doc_lists_every_registered_primitive() -> None:
    doc = (_REPO_ROOT / "docs" / "dev" / "jax_coverage.md").read_text()
    documented = set(re.findall(r"^\| `([a-z0-9_-]+)` +\|", doc, flags=re.MULTILINE))
    registered = {name for name, _ in duq.jax.PRIMITIVE_COVERAGE}
    missing = registered - documented
    stale = documented - registered
    assert not missing, f"primitives missing from docs/dev/jax_coverage.md: {sorted(missing)}"
    assert not stale, f"stale primitives in docs/dev/jax_coverage.md: {sorted(stale)}"
