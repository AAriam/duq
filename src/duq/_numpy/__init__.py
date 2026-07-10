"""NumPy back-end for :class:`duq.Quantity` (NEP 13 / NEP 18).

This package is imported *lazily* -- only from ``Quantity``'s array hooks and
array helpers -- so ``import duq`` and all scalar arithmetic stay NumPy-free.
It carries units through the curated ufunc and function tables, driven by the
core :mod:`duq._rules` engine, and fails loud on anything uncovered.
"""

from __future__ import annotations

from ._dispatch import (
    dispatch_function,
    dispatch_ufunc,
    get_item,
    iter_quantity,
    magnitude_dtype,
    magnitude_len,
    magnitude_ndim,
    magnitude_shape,
    magnitude_size,
    matmul,
    q_astype,
    q_ravel,
    q_reduce,
    q_reshape,
    q_transpose,
    richcompare,
    scalar_item,
)

__all__ = (
    "dispatch_function",
    "dispatch_ufunc",
    "get_item",
    "iter_quantity",
    "magnitude_dtype",
    "magnitude_len",
    "magnitude_ndim",
    "magnitude_shape",
    "magnitude_size",
    "matmul",
    "q_astype",
    "q_ravel",
    "q_reduce",
    "q_reshape",
    "q_transpose",
    "richcompare",
    "scalar_item",
)
