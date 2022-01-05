from __future__ import annotations
from typing import Union, Sequence, Tuple
from functools import partial
from itertools import combinations

import numpy as np

from .data.dimensions_units import primary, derived
from .helpers import parse_base_with_exp_string as parse_base_exp
from .helpers import generate_symbol_for_base_exp_series as gen_symbol
from .helpers import raise_for_type


class Dimension:
    """
    Object representing a primary or derived physical dimension, allowing for dimensional analysis.

    Parameters
    ----------
    dimension : Union[str, Sequence, numpy.ndarray]
        The representation of the dimension.
        For string inputs, the string may be composed of a number of
        single dimensions, separated by dots. Each single dimension
        may have an exponent, denoted after a '^' symbol. The exponent
        may be written as an integer, or a fraction, where the nominator
        and denominator are separated by a '/' symbol.
        A list of supported input dimensions can be viewed by calling the
        class method `supported_input_dimensions`.
        Examples of supported string arguments:
            Using dimension names: "length", "length^2", "mass.length^2.time^-2"
            or dimension symbols: "L", "L^2", "M.L^2.T^-2", "L^3/2"

        Another acceptable input is a 1-dimensional sequence of numbers with a length
        equal to the number of supported input dimensions `len(supported_input_dimensions())`
        where each number in the sequence corresponds to the exponent of that particular dimension.
        However, this is less convenient and is primarily for usage inside the class.

        Another option is to use the alternative constructor method `from_prim_dim_decomposition`,
        which accepts a sequence of 7 numbers, corresponding to the exponents of the
        primary-dimension-decomposition of the desired dimension, in the order:
        [mass, length, time, electric current, temperature, amount of substance, luminous intensity]
    """

    # Initialize class attributes
    _prim_dim_count: int = len(primary.keys())  # = 7, unless there is fundamental changes in physics.
    _db_all: dict = primary | derived
    _db_names: np.ndarray = np.array([dim["name"] for dim in _db_all.values()])
    _db_symbols: np.ndarray = np.array([dim["symbol"] for dim in _db_all.values()])
    _db_si_units: np.ndarray = np.array([list(dim["units"].values())[0]["symbol"] for dim in _db_all.values()])
    _db_prim_exps: np.ndarray = np.array([dim["prim_dim_exps"] for dim in _db_all.values()])

    @classmethod
    def supported_input_dimensions(cls) -> Tuple:
        """
        Returns a tuple of names and symbols of supported dimensions
        available for constructing a Dimension object.
        """
        return tuple([(name, symbol) for name, symbol in zip(Dimension._db_names, Dimension._db_symbols)])

    @classmethod
    def from_prim_dim_decomposition(cls, prim_dims_exps) -> Dimension:
        """
        Alternative factory method to construct a Dimension object
        from a list of primary dimensions exponents, in the order:
        [mass, length, time, electric current, temperature, amount of substance, luminous intensity]

        Parameters
        ----------
        prim_dims_exps : array-like
            A sequence of 7 numbers, representing the exponents of each
            primary dimension composing the dimension.

        Returns
        -------
            Dimension
        """
        all_dims_exps = np.zeros(cls._db_names.size)
        all_dims_exps[:cls._prim_dim_count] = prim_dims_exps
        return cls(all_dims_exps)

    def __init__(self, dimension):
        self._all_dims_exps = np.zeros(self._db_names.size)
        if isinstance(dimension, str):
            if dimension == "":
                raise ValueError("Dimension input is an empty string.")
            else:
                dims, exps = parse_base_exp(dimension)
                for dim, exp in zip(dims, exps):
                    mask = (self._db_symbols == dim) + (self._db_names == dim)
                    if np.any(mask):
                        self._all_dims_exps[mask] += exp
                    else:
                        raise ValueError(f"Dimension {dim} not recognized.")
        elif isinstance(dimension, (list, np.ndarray)):
            all_dims_exps = np.array(dimension)
            if all_dims_exps.shape != self._all_dims_exps.shape:
                raise ValueError(f"`dimension` array should have a shape of {self._all_dims_exps.shape}.")
            elif all_dims_exps.dtype.kind not in (np.typecodes["AllInteger"] + np.typecodes["AllFloat"]):
                raise ValueError("All elements of the `dimension` array must be numbers.")
            else:
                self._all_dims_exps[...] = all_dims_exps
        else:
            raise ValueError("Argument of `dimension` should either be a string or array-like of numbers.")

    def __repr__(self):
        return f"Dimension({repr(list(self._all_dims_exps))})"

    def __str__(self):
        str_repr = (
            f"As is:    {self.symbol_as_is} = {self.name_as_is} [{self.si_unit_as_is}]\n"
            f"Shortest: {self.symbol_shortest_composition} = {self.name_shortest_composition} "
            f"[{self.si_unit_shortest_composition}]\n"
            f"Primary:  {self.symbol_primary_decomposition} = {self.name_primary_decomposition} "
            f"[{self.si_unit_primary_decomposition}]"
        )
        return str_repr

    def __eq__(self, other):
        raise_for_type(other, Dimension, "Equality can only be assessed between two Dimension objects.")
        return np.all(self.exponents_primary_decomposition == other.exponents_primary_decomposition)

    def __mul_common__(self, other):
        raise_for_type(other, Dimension, "Multiplication is only defined between two Dimension objects.")
        return self._all_dims_exps + other._all_dims_exps

    def __mul__(self, other):
        return Dimension(self.__mul_common__(other))

    def __imul__(self, other):
        self._all_dims_exps = self.__mul_common__(other)
        return self

    def __truediv_common__(self, other):
        raise_for_type(other, Dimension, "Division is only defined between two Dimension objects.")
        return self._all_dims_exps - other._all_dims_exps

    def __truediv__(self, other):
        return Dimension(self.__truediv_common__(other))

    def __itruediv__(self, other):
        self._all_dims_exps = self.__truediv_common__(other)
        return self

    def __pow_common__(self, power):
        raise_for_type(power, (int, float), "Exponentiation is only defined for a number.")
        return self._all_dims_exps * power

    def __pow__(self, power):
        return Dimension(self.__pow_common__(power))

    def __ipow__(self, power):
        self._all_dims_exps = self.__pow_common__(power)
        return self

    @property
    def name_as_is(self) -> str:
        """
        Name representation of the current dimension, with not simplification applied.
        """
        names_ordered, exps_ordered = self._order_for_repr([self._db_names, self._all_dims_exps])
        return gen_symbol(names_ordered, exps_ordered, " . ").replace("empty", "dimensionless")

    @property
    def name_shortest_composition(self) -> str:
        """
        Name representation of the shortest equivalent composition for the current dimension.
        """
        return self.equiv_dim_shortest_composition.name_as_is

    @property
    def name_primary_decomposition(self) -> str:
        """
        Name representation of the equivalent primary dimension decomposition of the current dimension.
        """
        return self.equiv_dim_primary_decomposition.name_as_is

    @property
    def symbol_as_is(self) -> str:
        """
        Symbol representation of the current dimension, with not simplification applied.
        """
        symbols_ordered, exps_ordered = self._order_for_repr([self._db_symbols, self._all_dims_exps])
        return gen_symbol(symbols_ordered, exps_ordered).replace("empty", "1")

    @property
    def symbol_shortest_composition(self) -> str:
        """
        Symbol representation of the shortest equivalent composition for the current dimension.
        """
        return self.equiv_dim_shortest_composition.symbol_as_is

    @property
    def symbol_primary_decomposition(self) -> str:
        """
        Symbol representation of the equivalent primary dimension decomposition of the current dimension.
        """
        return self.equiv_dim_primary_decomposition.symbol_as_is

    @property
    def si_unit_as_is(self) -> str:
        """
        SI-unit representation of the current dimension, with not simplification applied.
        """
        si_units_ordered, exps_ordered = self._order_for_repr([self._db_si_units, self._all_dims_exps])
        return gen_symbol(si_units_ordered, exps_ordered, ".").replace("empty", "1")

    @property
    def si_unit_shortest_composition(self) -> str:
        """
        SI-unit representation of the shortest equivalent composition for the current dimension.
        """
        return self.equiv_dim_shortest_composition.si_unit_as_is

    @property
    def si_unit_primary_decomposition(self) -> str:
        """
        SI-unit representation of the equivalent primary dimension decomposition of the current dimension.
        """
        return self.equiv_dim_primary_decomposition.si_unit_as_is

    @property
    def exponents_as_is(self) -> np.ndarray:
        """
        Array of exponents for each constituting dimension, with no simplification applied.
        The order of dimensions is the same as returned by `Dimension.supported_input_dimensions()`.
        """
        return self._all_dims_exps

    @property
    def exponents_shortest_composition(self) -> np.ndarray:
        """
        Array of exponents for each constituting dimension of the shortest equivalent composition
        for the current dimension. The order of dimensions is the same as returned by
        `Dimension.supported_input_dimensions()`.
        """
        return self.equiv_dim_shortest_composition.exponents_as_is

    @property
    def exponents_primary_decomposition(self) -> np.ndarray:
        """
        Array of exponents of the primary dimension decomposition of the current dimension, in the order:
        [mass, length, time, electric current, temperature, amount of substance, luminous intensity].
        """
        return self.equiv_dim_primary_decomposition.exponents_as_is[:self._prim_dim_count]

    @property
    def is_primary_dimension(self) -> bool:
        """
        Whether the current dimension is a primary dimension.
        """
        return np.abs(self.exponents_primary_decomposition).sum() == 1

    @property
    def equiv_dim_shortest_composition(self) -> Dimension:
        """
        Derive the shortest equivalent dimension of the current dimension,
        in the sense that is composed of the least amount of known dimensions.

        Returns
        -------
            Dimension
            A new `Dimension` object with the same primary dimension decomposition
            as the current dimension, but composed of the least possible number
            of known dimensions.
        """
        # Create empty array for storing dimension decomposition of the equivalent dimension
        new_dim_dec = np.zeros_like(self._all_dims_exps)
        # Note the current state of the primary dimension decomposition of the dimension
        current_prim_dim_dec = self.exponents_primary_decomposition
        sum_current_prim_dim_exps = np.abs(current_prim_dim_dec).sum()
        # Create index array for use in the While loop, since we are concatenating two arrays
        dim_idxs = np.tile(np.arange(self._all_dims_exps.size), 2)
        # Repeat until all current primary dimensions are consumed
        while not np.isclose(sum_current_prim_dim_exps, 0):
            # Subtract all known dimension decompositions from the current decomposition
            subtraction = current_prim_dim_dec - self._db_prim_exps
            # And also add
            addition = current_prim_dim_dec + self._db_prim_exps
            # Concatenate the two resulting arrays for avoiding repetition of code
            sub_add = np.concatenate((subtraction, addition))
            # Take the absolute value of all dimension exponents, and sum them
            sub_add_result = np.abs(sub_add).sum(axis=1)
            # The best result is the one with the smallest sum; note its index
            best_result_idx = np.argmin(sub_add_result)
            # Also note the actual index of the dimension that gave the best resul
            best_result_dim_idx = dim_idxs[best_result_idx]
            # Add +1/-1 to the exponent of the dimension that gave the best result
            # Note: if the best result was obtained via subtraction (i.e. dividing by another dimension)
            # then +1 should be added as the exponent; otherwise if the result was obtained via addition
            # (i.e. multiplying with another exponent), then -1 should be added as the exponent.
            # This is inferred from the index of the best result, i.e. finding out if it was in the first
            # or second half of the concatenated list.
            new_dim_dec[best_result_dim_idx] += (1 if best_result_idx < self._all_dims_exps.size else -1)
            # Update the current state
            current_prim_dim_dec = sub_add[best_result_idx]
            sum_current_prim_dim_exps = np.abs(current_prim_dim_dec).sum()
        return Dimension(new_dim_dec)

    @property
    def equiv_dim_primary_decomposition(self) -> Dimension:
        """
        Derive the equivalent dimension of the current dimension, composed only
        of primary dimensions, i.e. its primary dimension decomposition.

        Returns
        -------
            Dimension
            A new `Dimension` object with the same primary dimension decomposition
            as the current dimension, but composed only of primary dimensions.
        """
        each_prim_dim_decomposition = (self._all_dims_exps.reshape(-1, 1) * self._db_prim_exps)
        total_prim_dim_decomposition = each_prim_dim_decomposition.sum(axis=0)
        return Dimension.from_prim_dim_decomposition(total_prim_dim_decomposition)

    def equiv_dim_all(self, max_num_dims: int = 5, max_exp: int = 3) -> list[Dimension]:
        """
        Derive all equivalent dimensions of the current dimension, composed of
        at most 7 different known (primary and derived) dimensions, where each
        dimension has an integer exponent.

        Parameters
        ----------
        max_num_dims : int (optional; default:5)
            Maximum allowed number of composing dimensions in an equivalent dimension
        max_exp : int (optional; default: 3)
            Maximum allowed absolute value of an exponent in an equivalent dimension.

        Returns
        -------
            list[Dimension]
            A list of all equivalent dimensions, as `Dimension` objects.
        """

        # Create an index array of all unique combinations of indices of the available dimensions.
        # The array will have a shape of (b, n), where b is the binomial coefficient (s, n), where
        # s is the number of available dimensions in the class database (e.g. size of self._db_names),
        # and n is the number of primary dimensions, i.e. 7.
        # With the current number of available dimensions (19), the shape will be (50388, 7).
        combs = np.array(list(combinations(range(self._db_names.size), self._prim_dim_count)))

        solutions = []
        # Iterate over all combinations
        for dim_idxs in combs:
            try:
                # Create a matrix of primary dimension decomposition exponents of 7 available dimensions,
                # transpose the matrix and solve the system of linear equations to get the exponents of those
                # 7 dimensions that result in the current dimension.
                solution = (
                    np.linalg.solve(
                        self._db_prim_exps[dim_idxs].T, self.exponents_primary_decomposition))
                # Create an empty array for storing exponents of all available dimensions
                all_dims_exps = np.zeros(self._db_names.size)
                # Assign the solution (i.e. exponents) to the indices corresponding to those dimensions
                all_dims_exps[dim_idxs] = solution
                solutions.append(all_dims_exps)
            # The matrix may be singular, in which case there is no unique solution and numpy throws an error.
            except np.linalg.LinAlgError:
                pass

        # Only take those unique solutions where all exponents are integers in the range [-10, 10].
        solutions = np.array(solutions)
        int_sols = np.unique(
            solutions[
                np.all(solutions == solutions.astype(int), axis=1) *
                np.all(np.abs(solutions) < max_exp, axis=1)
            ],
            axis=0
        )
        # Filter out the equivalent dimensions that are exactly the same as the current dimension
        int_sols_not_self = int_sols[
            np.logical_not(
                np.all(
                    (int_sols == self.exponents_as_is), axis=1
                )
            )
        ]
        # Sort the dimensions from the lowest sum of exponents (absolute values) to highest.
        idx_mask = np.argsort(np.abs(int_sols_not_self).sum(axis=1))
        int_sols_not_self_sorted = int_sols_not_self[idx_mask]

        # Filter out the dimensions that have a higher number of composing dimensions than specified
        nonzero_exps_in_each_dim = np.count_nonzero(int_sols_not_self_sorted, axis=1)
        bool_mask = nonzero_exps_in_each_dim <= max_num_dims
        int_sols_not_self_max_dim_sorted = int_sols_not_self_sorted[bool_mask]

        # Create a `Dimension` object for each solution
        equiv_dims = []
        for sol in int_sols_not_self_max_dim_sorted:
            equiv_dims.append(Dimension(sol))
        return equiv_dims

    def _order_for_repr(self, arrays: Union[Sequence[Union[Sequence, np.ndarray]], np.ndarray]) -> list:
        """
        Take an array of arrays, re-order each sub-array (not in-place)
        according to the indices generated by `self._ordered_idx_array`
        and return a list of ordered arrays.

        Parameters
        ----------
        arrays : Union[Sequence[Union[Sequence, np.ndarray]], np.ndarray]
            An array containing the arrays to be ordered.
        """
        ordered_arrays = []
        for array in arrays:
            ordered_arrays.append(array[self._ordered_idx_array])
        return ordered_arrays

    @property
    def _ordered_idx_array(self) -> np.ndarray:
        """
        Calculate an array of indices to apply to other arrays
        (self._db_names, self._db_symbols, self._all_dims_exps)
        to re-order them based on priority of dimensions for
        generating name and symbol representations.
        The priority is set as follows:
            1. Derived dimensions before primary dimensions
            2. More complex derived dimensions before simpler ones
            3. Primary dimensions in the conventional order
        """
        indices = np.arange(self._db_names.size)
        ordered_indices = np.concatenate(
            (   # Take the derived dimensions and flip them
                # since in the original database, derived dimensions
                # go from simple to complex
                indices[self._prim_dim_count:][::-1],
                # Put the primary dimensions afterwards, but don't change
                # the order within them, since it's already in the conventional order
                indices[: self._prim_dim_count]
            )
        )
        return ordered_indices


class PredefinedDimensions:
    """
    Container class for all available known dimensions.
    Each available dimension is defined as a property in
    this class, which returns a new Dimension object for
    that dimension, each time it is called.
    """
    def __init__(self):
        def dim_gen(self, symbol):
            return Dimension(symbol)
        init_dict = primary | derived
        for name, data in init_dict.items():
            setattr(PredefinedDimensions, name, property(partial(dim_gen, symbol=data["symbol"])))


# Instantiate the container class,
# this can now be directly imported whenever needed.
predefined = PredefinedDimensions()
