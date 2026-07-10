"""Entry points wired into :class:`duq.Quantity`'s NumPy hooks.

These functions are imported lazily from the ``Quantity.__array_*`` methods and
its array introspection/reduction helpers; they route each call to the curated
:data:`~duq._numpy._ufuncs.UFUNC_TABLE` / :data:`~duq._numpy._functions.FUNCTION_TABLE`
and fail loud (``UnsupportedOperationError``) on anything uncovered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from duq._errors import UnsupportedOperationError

from ._common import Quantity, box, compare_core, magnitude, unit_of
from ._functions import FUNCTION_TABLE
from ._ufuncs import REDUCE_PRESERVE, UFUNC_TABLE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

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


# -- protocol dispatch -------------------------------------------------------


def _name(ufunc: Any) -> str:
    return str(getattr(ufunc, "__name__", ufunc))


def dispatch_ufunc(
    ufunc: Any, method: str, inputs: tuple[object, ...], kwargs: dict[str, Any]
) -> object:
    """Route a NumPy ufunc call through the curated unit-rule table."""
    if kwargs.get("out") is not None:
        raise UnsupportedOperationError(
            "a Quantity is immutable, so the ufunc out= argument is not supported"
        )
    call_kwargs = {key: value for key, value in kwargs.items() if key != "out"}
    if method == "__call__":
        handler = UFUNC_TABLE.get(ufunc)
        if handler is None:
            raise UnsupportedOperationError(
                f"the numpy ufunc {_name(ufunc)!r} has no duq unit rule and is unsupported"
            )
        return handler(ufunc, inputs, call_kwargs)
    if method in ("reduce", "accumulate"):
        return _reduce(ufunc, method, inputs, call_kwargs)
    raise UnsupportedOperationError(
        f"the ufunc method {_name(ufunc)}.{method} is not supported by duq"
    )


def _reduce(ufunc: Any, method: str, inputs: tuple[object, ...], kwargs: dict[str, Any]) -> object:
    if ufunc in REDUCE_PRESERVE:
        operand = inputs[0]
        result = getattr(ufunc, method)(magnitude(operand), **kwargs)
        unit = unit_of(operand)
        return box(result, unit) if unit is not None else result
    if ufunc is np.multiply:
        raise UnsupportedOperationError(
            "multiply.reduce is unsupported: it would raise the unit to the array "
            "length (an ambiguous unit**n); duq ships no .prod() equivalent in v1"
        )
    raise UnsupportedOperationError(
        f"the ufunc reduction {_name(ufunc)}.{method} is not supported by duq"
    )


def dispatch_function(
    func: Callable[..., object],
    types: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Route a NumPy function call through the curated unit-rule table."""
    handler = FUNCTION_TABLE.get(func)
    if handler is None:
        module = getattr(func, "__module__", "numpy")
        name = getattr(func, "__name__", repr(func))
        raise UnsupportedOperationError(
            f"the numpy function {module}.{name} has no duq unit rule and is "
            "unsupported; call duq.ustrip(unit, q) to drop units explicitly first"
        )
    return handler(args, dict(kwargs))


def richcompare(op_name: str, left: object, right: object) -> npt.NDArray[Any]:
    """Compare two operands (at least one array), returning a plain bool ndarray."""
    return cast("npt.NDArray[Any]", compare_core(op_name, left, right))


def matmul(a: Any, b: Any) -> object:
    """Matrix-multiply two operands (``@``), multiplying their units."""
    return np.matmul(a, b)


# -- array introspection -----------------------------------------------------


def magnitude_shape(mag: Any) -> tuple[int, ...]:
    """Return the shape of a magnitude (``()`` for a scalar)."""
    return np.shape(mag)


def magnitude_ndim(mag: Any) -> int:
    """Return the number of dimensions of a magnitude."""
    return int(np.ndim(mag))


def magnitude_size(mag: Any) -> int:
    """Return the number of elements in a magnitude."""
    return int(np.size(mag))


def magnitude_dtype(mag: Any) -> np.dtype[Any]:
    """Return the NumPy dtype of a magnitude."""
    return np.asarray(mag).dtype


def magnitude_len(mag: Any) -> int:
    """Return ``len`` of an array magnitude (raises for a scalar magnitude)."""
    return len(mag)


def iter_quantity(q: Quantity) -> Iterator[Quantity]:
    """Iterate an array quantity, yielding element quantities (unit preserved)."""
    mag: Any = q.value
    return (Quantity(element, q.unit) for element in mag)


def get_item(q: Quantity, key: object) -> Quantity:
    """Index an array quantity, returning a quantity (unit preserved)."""
    mag: Any = q.value
    return Quantity(mag[key], q.unit)


def scalar_item(q: Quantity, args: tuple[Any, ...]) -> Quantity:
    """Return a single element as a scalar quantity (unit preserved)."""
    mag = np.asarray(q.value)
    return Quantity(mag.item(*args), q.unit)


# -- array reductions / reshaping --------------------------------------------


def q_transpose(q: Quantity) -> Quantity:
    """Transpose an array quantity (unit preserved)."""
    return Quantity(np.asarray(q.value).T, q.unit)


def q_reduce(q: Quantity, name: str, kwargs: dict[str, Any], unit_power: int = 1) -> Quantity:
    """Reduce a quantity with ``numpy.<name>`` (unit raised to ``unit_power``)."""
    func = getattr(np, name)
    result = func(q.value, **kwargs)
    unit = q.unit if unit_power == 1 else q.unit**unit_power
    return Quantity(result, unit)


def q_reshape(q: Quantity, shape: tuple[Any, ...], kwargs: dict[str, Any]) -> Quantity:
    """Reshape an array quantity (unit preserved)."""
    mag = np.asarray(q.value)
    return Quantity(mag.reshape(*shape, **kwargs), q.unit)


def q_ravel(q: Quantity, kwargs: dict[str, Any]) -> Quantity:
    """Flatten an array quantity (unit preserved)."""
    mag = np.asarray(q.value)
    return Quantity(mag.ravel(**kwargs), q.unit)


def q_astype(q: Quantity, dtype: Any, kwargs: dict[str, Any]) -> Quantity:
    """Cast an array quantity's dtype (unit preserved)."""
    mag = np.asarray(q.value)
    return Quantity(mag.astype(dtype, **kwargs), q.unit)
