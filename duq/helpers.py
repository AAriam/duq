from fractions import Fraction
import numpy as np
from .data.unicode_chars import superscript_chars
from typing import Tuple, Union, Sequence


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
            raise ValueError("Only one '^' symbol may appear for each term (i.e. between two '.' symbols).")
        bases.append(base_and_exp[0])
    bases = np.array(bases)
    exps = np.array(exps)
    return bases, exps


def pretty_print_base_with_exp_series(base_array: Sequence, exp_array: Sequence, seperator: str = "") -> str:
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
            base_array[idx] = f"({pretty_print_base_with_exp_series(base_array_, exp_array_, '.')})"

    exp_array = np.array(exp_array)
    if base_array.size == 0:
        return "empty"
    else:
        seperator_array = np.full(exp_array.size, seperator)
        seperator_array[-1] = ""
        superscripted_exps = np.array(list(map(superscript_map_func, exp_array)))
        superscripted_exps_with_seperator = np.core.defchararray.add(superscripted_exps, seperator_array)
        full_expression_array = np.core.defchararray.add(base_array, superscripted_exps_with_seperator)
        return "".join(full_expression_array)


def generate_symbol_for_base_exp_series(base_array, exp_array, seperator=""):
    nonzero_exp_mask = exp_array != 0
    nonzero_exps = exp_array[nonzero_exp_mask]
    bases_of_nonzero_exps = base_array[nonzero_exp_mask]
    pretty_string_representation = pretty_print_base_with_exp_series(bases_of_nonzero_exps, nonzero_exps, seperator)
    return pretty_string_representation


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


def raise_for_type(obj, obj_type, msg, error_type=NotImplementedError):
    if not isinstance(obj, obj_type):
        raise error_type(msg)
    return
