"""
Helper functions used in other modules.
"""

# Standard library
from typing import Tuple, Union, Sequence, Type
from fractions import Fraction

# 3rd-party
import numpy as np

# Self
from .data.unicode_chars import superscript_chars


def parse_base_with_exp_string(string: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a string representation of a collection of bases and exponents,
    seperated from each other by a '.' symbol. Each base may be followed
    by a `^` symbol, separating it from its exponent. The exponent should
    either be an integer, or a fraction where the nominator and denominator
    are separated by a `/` symbol.

    Parameters
    ----------
    string : str
        The string representation of the expression.

    Returns
    -------
        tuple[numpy.ndarray, numpy.ndarray]

    Examples
    --------
    "kg.m^2.s^-2" -> (["kg", "m", "s"], [1, 2, -2])
    "m^3/2" -> (["m"], [1.5])
    """

    bases = []
    exps = []
    terms_list = string.split(".")
    for term in terms_list:
        base_and_exp = term.split("^")
        len_term = len(base_and_exp)
        if len_term == 0:
            raise ValueError("Consecutive '.' symbols are not allowed.")
        elif len_term == 1:
            exps.append(1)
        elif len_term == 2:
            exps.append(float(Fraction(base_and_exp[1])))
        else:
            raise ValueError(
                "Only one '^' symbol may appear for each term (i.e. between two '.' symbols)."
            )
        bases.append(base_and_exp[0])
    bases = np.array(bases)
    exps = np.array(exps)
    return bases, exps


def generate_symbol_for_base_exp_series(
    base_array: np.ndarray, exp_array: np.ndarray, seperator: str = ""
) -> str:
    """
    Take two arrays representing the bases and exponents of an expression,
    and create a string representation of the expression where the bases with
    an exponent of 0 are ignored, bases with an exponent of 1 are written without the exponent,
    and all other exponents are superscripted, and each base-exponent pair is separated from the
    others by a given `separator`.

    Parameters
    ----------
    base_array : numpy.ndarray
        Array representing the bases.
    exp_array : numpy.ndarray
        Array representing the exponents
    seperator : str
        Separator to be printed in between base-exponent pairs.

    Returns
    -------
        str
        String representation of the expression.
    """
    nonzero_exp_mask = exp_array != 0
    nonzero_exps = exp_array[nonzero_exp_mask]
    bases_of_nonzero_exps = base_array[nonzero_exp_mask]
    pretty_string_representation = pretty_print_base_with_exp_series(
        bases_of_nonzero_exps, nonzero_exps, seperator
    )
    return pretty_string_representation


def pretty_print_base_with_exp_series(
    base_array: Sequence, exp_array: Sequence, seperator: str = ""
) -> str:
    """
    Create a string representation of a collection of bases and their corresponding exponents,
    where exponents are in superscript and the terms may be separated by a seperator character.

    Parameters
    ----------
    base_array : Sequence
        Collection of bases.
    exp_array : Sequence
        Collection of exponents
    seperator : str
        A character or a string to separate base-exponent pairs from each other.

    Returns
    -------
        str
        The string representation of the expression.

    Examples
    --------
    (base_array=["L", "M", "T"], exp_array=[1, 2, -2]) -> "LM²T⁻²"
    (base_array=["length", "mass", "time"], exp_array=[1, 2, -2], seperator=".") -> "length.mass².time⁻²"
    """
    base_array = np.array(base_array)

    for idx, base in enumerate(base_array):
        if "." in base or "^" in base:
            base_array_, exp_array_ = parse_base_with_exp_string(base)
            base_array[
                idx
            ] = f"({pretty_print_base_with_exp_series(base_array_, exp_array_, '.')})"

    exp_array = np.array(exp_array)
    if base_array.size == 0:
        return "empty"
    else:
        seperator_array = np.full(exp_array.size, seperator)
        seperator_array[-1] = ""
        superscripted_exps = np.array(list(map(superscript_map_func, exp_array)))
        superscripted_exps_with_seperator = np.core.defchararray.add(
            superscripted_exps, seperator_array
        )
        full_expression_array = np.core.defchararray.add(
            base_array, superscripted_exps_with_seperator
        )
        return "".join(full_expression_array)


def superscript_map_func(exp: Union[int, float]) -> str:
    """
    Turn a number into its superscript string form.
    """
    if exp == 1:
        exp_reduced_str = ""
    elif int(exp) == exp:
        exp_reduced_str = str(int(exp))
    else:
        exp_reduced_str = str(Fraction(exp).limit_denominator())
    return "".join([superscript_chars[char] for char in exp_reduced_str])


def raise_for_type(
    obj: object,
    obj_type: Union[Type, Tuple[Type]],
    msg: str,
    error_type: type = NotImplementedError,
) -> None:
    """
    Check an object's type and raise a specific error with a given message
    if the type does not match with the given expected type.

    Parameters
    ----------
    obj : Object
        The object whose type is to be checked.
    obj_type : Type
        The expected type of the object.
    msg : str
        Error message to be shown when the type does not match.
    error_type : Exception (optional; default: NotImplementedError)
        Type of error to be raised.

    Returns
    -------
        None
    """
    if not isinstance(obj, obj_type):
        raise error_type(msg)
    return


def order_for_repr(
    arrays: Union[Sequence[Union[Sequence, np.ndarray]], np.ndarray], cut_idx: int
) -> list:
    """
    Re-order each sub-array (not in-place) in an array of arrays.
    Each subarray is reordered based on priority of dimensions/units for
    generating name and symbol representations.
    The priority is set as follows:
        1. Derived dimensions/units before primary dimensions/units
        2. More complex derived dimensions/units before simpler ones
        3. Primary dimensions/units in the conventional order
    All arrays containing names, symbols etc. are already ordered in a way
    that the primary dimensions/units are first (in the conventional order),
    followed by derived dimensions/units, going from simple ones to more complex ones.
    Thus, for reordering, only the index of the first derived dimension/unit is needed.

    Parameters
    ----------
    arrays : Union[Sequence[Union[Sequence, np.ndarray]], np.ndarray]
        An array containing the arrays to be ordered.
    cut_idx : int
        Index used to slice the array into two.
        This should be the index of the first derived dimension/unit in the array.

    Returns
    -------
    ordered_arrays : list
        List of ordered arrays.

    Examples
    --------
    ([[a,b,c,d,e,f]], 3) -> [[f,e,d,a,b,c]]
    """

    # Calculate indices of the reordered array
    indices = np.arange(len(arrays[0]))
    ordered_indices = np.concatenate(
        (  # Take the derived dimensions and flip them
            # since in the original database, derived dimensions
            # go from simple to complex
            indices[cut_idx:][::-1],
            # Put the primary dimensions afterwards, but don't change
            # the order within them, since it's already in the conventional order
            indices[:cut_idx],
        )
    )
    # Reorder arrays using the calculated index array.
    ordered_arrays = []
    for array in arrays:
        ordered_arrays.append(array[ordered_indices])
    return ordered_arrays
