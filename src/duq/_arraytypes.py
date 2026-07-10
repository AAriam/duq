"""Lazy NumPy type detection that never imports NumPy itself.

The helpers here answer "is this object a NumPy array or scalar?" by consulting
:data:`sys.modules`.  If NumPy has never been imported then nothing in the
process can be a NumPy object, so the answer is trivially ``False`` and
``import duq`` -- together with all scalar arithmetic -- stays NumPy-free.  The
moment a genuine array appears, NumPy is guaranteed to already be in
``sys.modules`` (the caller had to import it to build the array), so the check
is both correct and free of an eager import.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = (
    "is_numpy_array",
    "is_numpy_magnitude",
    "numpy_if_loaded",
)


def numpy_if_loaded() -> ModuleType | None:
    """Return the already-imported ``numpy`` module, or ``None``.

    Returns
    -------
    module or None
        The :mod:`numpy` module if it has been imported, else ``None``.
    """
    return sys.modules.get("numpy")


def is_numpy_array(value: object) -> bool:
    """Return whether ``value`` is a NumPy ``ndarray`` without importing NumPy.

    Parameters
    ----------
    value : object
        The object to test.

    Returns
    -------
    bool
        ``True`` only if NumPy is imported and ``value`` is an ``ndarray``.
    """
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.ndarray)


def is_numpy_magnitude(value: object) -> bool:
    """Return whether ``value`` is a NumPy array or scalar without importing NumPy.

    Parameters
    ----------
    value : object
        The object to test.

    Returns
    -------
    bool
        ``True`` only if NumPy is imported and ``value`` is an ``ndarray`` or a
        NumPy scalar (``numpy.generic``).
    """
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.ndarray | np.generic)
