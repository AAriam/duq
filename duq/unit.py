from __future__ import annotations
from typing import Tuple, Sequence, Union

import numpy as np

from .dimension import Dimension
from .data.dimensions_units import primary, derived
from .data.physical_constants import phys_consts
from .helpers import parse_base_with_exp_string as parse_base_exp
from .helpers import generate_symbol_for_base_exp_series as gen_symbol
from .helpers import raise_for_type


class Unit:
    """
    Object representing a physical unit, allowing for
    analysis, conversion and manipulation of units.

    Parameters
    ----------
    unit : Union[str, Sequence, numpy.ndarray, Dimension]
        If string:
            The string representation of the unit.
            The string may be composed of a number of single units,
            separated by dots. Each single unit may have an exponent,
            denoted after a '^' symbol. The exponent may be written as
            an integer, or a fraction, where nominator and denominator
            are separated by a '/' symbol.
            A list of supported input units can be viewed by calling the
            class method `supported_input_units`.
            Examples of supported string arguments:
                Using unit names: "metre", "metre^3/2", "kilogram.metre^2.second^-2"
                or unit symbols: "m", "m^3/2" "kg.m^2.s^-2"

        If sequence or numpy array:
            1-dimensional sequence of numbers with a length equal to the number of supported
            input units `len(supported_input_units())`, where each number in the sequence
            corresponds to the exponent of that particular unit. However, this is less convenient
            and is primarily for usage inside the class.

        If Dimension:
            Dimension object; the created unit will be in primary SI units.

        Another option is to use the alternative constructor method `from_prim_unit_decomposition`,
        which accepts a sequence of 7 numbers, corresponding to the exponents of the
        primary-unit-decomposition of the desired dimension, in the order:
        [kilogram, metre, second, ampere, kelvin, mole, candela]
    """

    # --- Initialize class attributes ---
    # Number of primary units - to use as reference for slicing
    _prim_unit_count: int = sum([len(dim["units"].keys()) for dim in primary.values()])
    # Dictionary containing all units and dimensions info
    _db_all: dict = primary | derived
    # Name of units
    _db_names = np.array([unit["name"] for dim in _db_all.values() for unit in dim["units"].values()])
    # Symbol of units
    _db_symbols = np.array([unit["symbol"] for dim in _db_all.values() for unit in dim["units"].values()])
    # Conversion factor of units (to SI unit)
    _db_conv_factors = np.array([unit["conv_factor"] for dim in _db_all.values() for unit in dim["units"].values()])
    # Prefix exponent of each unit (e.g. for kg = 3, for g = 0)
    _db_prefix_exp = np.array([unit["prefix_exp"] for dim in _db_all.values() for unit in dim["units"].values()])
    # Number of available dimensions
    _dims_count: int = len(_db_all.keys())
    # Name of the dimension of each available unit
    # (each name is repeated as many times as there are units in that dim)
    _db_dim_names = np.array([dim["name"] for dim in _db_all.values() for unit in dim["units"].values()])
    # Index of the dimension of each available unit
    # (each index is repeated as many times as there are units in that dim)
    _db_dims_idx = np.array([idx for idx, dim in enumerate(_db_all.values()) for unit in dim["units"].values()])
    _, _db_si_units_idx, _db_dim_units_counts = np.unique(_db_dims_idx, return_index=True, return_counts=True)
    # Index of the SI unit for each unit (repeated as many times as there are units with the same SI unit)
    _db_si_units_idx_all = np.repeat(_db_si_units_idx, _db_dim_units_counts)
    del _

    @classmethod
    def supported_input_units(cls) -> Tuple:
        """
        Returns a tuple of names and symbols of supported units
        available for constructing a Unit object.
        """
        return tuple([(name, symbol) for name, symbol in zip(cls._db_names, cls._db_symbols)])

    @classmethod
    def from_prim_unit_decomposition(cls, prim_units_exps) -> Unit:
        """
        Alternative factory method to construct a Unit object
        from a list of primary units exponents, in the order:
        [kilogram, metre, second, ampere, kelvin, mole, candela]

        Parameters
        ----------
        prim_units_exps : array-like
            A sequence of 7 numbers, representing the exponents of each
            primary unit composing the unit.

        Returns
        -------
            Unit
        """
        all_units_exps = np.zeros(cls._db_names.size)
        all_units_exps[cls._db_si_units_idx[:Dimension._prim_dim_count]] = prim_units_exps
        return cls(all_units_exps)

    @classmethod
    def from_dimension_object(cls, dimension: Dimension) -> Unit:
        """
        Alternative factory method to construct a Unit object from a `Dimension` object.
        The created unit will be the SI unit of the corresponding dimension.

        Parameters
        ----------
        dimension : dimension.Dimension
            A `Dimension` object from the `dimension` module.

        Returns
        -------
            Unit
        """
        all_units_exps = np.zeros(cls._db_names.size)
        all_units_exps[cls._db_si_units_idx] = dimension.exponents_as_is
        return cls(all_units_exps)

    def __init__(self, unit):
        self._all_units_exps = np.zeros(self._db_symbols.size)
        if isinstance(unit, str):
            if unit == "":
                raise ValueError("Unit input is an empty string.")
            else:
                units, exps = parse_base_exp(unit)
                for unit, exp in zip(units, exps):
                    mask = (self._db_symbols == unit) + (self._db_names == unit)
                    if np.any(mask):
                        self._all_units_exps[mask] += exp
                    else:
                        raise ValueError(f"Unit {unit} not recognized.")
        elif isinstance(unit, (list, np.ndarray)):
            all_units_exps = np.array(unit)
            if all_units_exps.shape != self._db_symbols.shape:
                raise ValueError(f"`unit` array should have a shape of {self._db_symbols.shape}")
            elif all_units_exps.dtype.kind not in (np.typecodes["AllInteger"] + np.typecodes["AllFloat"]):
                raise ValueError("All elements of the `unit` array must be numbers.")
            else:
                self._all_units_exps[...] = all_units_exps
        elif isinstance(unit, Dimension):
            raise ValueError("For instantiation using a `Dimension` object, use Unit.from_dimension_object.")
        else:
            raise ValueError("Argument of `unit` should either be a string or array-like of numbers.")
        unit_dims = np.zeros(self._dims_count)
        np.add.at(unit_dims, self._db_dims_idx, self._all_units_exps)
        self._dimension = Dimension(unit_dims)

    def __repr__(self):
        return f"Unit({repr(list(self._all_units_exps))})"

    def __str__(self):
        str_repr = (
            f"Unit:\n"
            f"-----\n"
            f"As is:      {self.symbol_as_is} = {self.name_as_is}\n"
            f"SI:         {self.symbol_si} = {self.name_si}\n"
            f"SI primary: {self.symbol_si_primary} = {self.name_si_primary}\n\n"
            "Dimension:\n"
            "----------\n"
            f"{str(self.dimension)}"
        )
        return str_repr

    def __eq__(self, other):
        raise_for_type(other, Unit, "Equality can only be assessed between two Unit objects.")
        return (self.dimension == other.dimension) and (
                np.all(np.isclose(self.conversion_coefficients_to_si, other.conversion_coefficients_to_si))
        )

    def __mul_common__(self, other):
        raise_for_type(other, Unit, "Multiplication is only defined between two Unit objects.")
        return self._all_units_exps + other._all_units_exps

    def __mul__(self, other):
        return Unit(self.__mul_common__(other))

    def __imul__(self, other):
        self._all_units_exps = self.__mul_common__(other)
        self._dimension *= other.dimension
        return self

    def __truediv_common__(self, other):
        raise_for_type(other, Unit, "Division is only defined between two Unit objects.")
        return self._all_units_exps - other._all_units_exps

    def __truediv__(self, other):
        return Unit(self.__truediv_common__(other))

    def __itruediv__(self, other):
        self._all_units_exps = self.__truediv_common__(other)
        self._dimension /= other.dimension
        return self

    def __pow_common__(self, power):
        raise_for_type(power, (int, float), "Exponentiation is only defined for a number.")
        return self._all_units_exps * power

    def __pow__(self, power):
        return Unit(self.__pow_common__(power))

    def __ipow__(self, power):
        self._all_units_exps = self.__pow_common__(power)
        self._dimension **= power
        return self

    @property
    def name_as_is(self) -> str:
        """
        Name representation of the current unit, with not simplification applied.
        """
        names_ordered, exps_ordered = self._order_for_repr([self._db_names, self._all_units_exps])
        return gen_symbol(names_ordered, exps_ordered, " . ").replace("empty", "unitless")

    @property
    def name_si(self) -> str:
        """
        Name representation of the equivalent SI unit of the current unit.
        """
        return self.equiv_unit_si.name_as_is

    @property
    def name_si_primary(self) -> str:
        """
        Name representation of the equivalent primary SI unit of the current unit.
        """
        return self.equiv_unit_si_primary.name_as_is

    @property
    def symbol_as_is(self) -> str:
        """
        Symbol representation of the current unit, with not simplification applied.
        """
        symbols_ordered, exps_ordered = self._order_for_repr([self._db_symbols, self._all_units_exps])
        return gen_symbol(symbols_ordered, exps_ordered, ".").replace("empty", "1")

    @property
    def symbol_si(self) -> str:
        """
        Symbol representation of the equivalent SI unit of the current unit.
        """
        return self.equiv_unit_si.symbol_as_is

    @property
    def symbol_si_primary(self) -> str:
        """
        Symbol representation of the equivalent primary SI unit of the current unit.
        """
        return self.equiv_unit_si_primary.symbol_as_is

    @property
    def exponents_as_is(self) -> np.ndarray:
        """
        Array of exponents for each constituting unit, with no simplification applied.
        The order of units is the same as returned by `Unit.supported_input_units()`.
        """
        return self._all_units_exps

    @property
    def exponents_si(self) -> np.ndarray:
        """
        Array of exponents for each constituting unit of the SI unit equivalent for
        the current unit. The order of units is the same as returned by
        `Unit.supported_input_units()`.
        """
        return self.equiv_unit_si.exponents_as_is

    @property
    def exponents_si_primary(self) -> np.ndarray:
        """
        Array of exponents for each constituting unit of the SI primary unit equivalent
        for the current unit. The order of units is the same as returned by
        `Unit.supported_input_units()`.
        """
        return self.equiv_unit_si_primary.exponents_as_is

    @property
    def dimension(self) -> Dimension:
        """
        `Dimension` object of the unit.
        """
        return self._dimension

    @property
    def is_si_unit(self) -> bool:
        """
        Whether the unit is an SI unit.
        """
        return np.all(self._all_units_exps == self.equiv_unit_si._all_units_exps)

    @property
    def equiv_unit_si(self) -> Unit:
        """
        Unit object representing the equivalent SI unit of the current unit,
        without calculating any conversion coefficients.
        """
        # Initialize empty array for storing the exponents of SI unit
        new_all_units_exps = np.zeros_like(self._all_units_exps)
        # Add the exponents of all units to the indices corresponding to SI units
        np.add.at(new_all_units_exps, self._db_si_units_idx_all, self._all_units_exps)
        return Unit(new_all_units_exps)

    @property
    def equiv_unit_si_primary(self) -> Unit:
        """
        Unit object representing the equivalent primary SI unit of the current unit,
        without calculating any conversion coefficients.
        """
        new_all_units_exps = np.zeros_like(self._all_units_exps)
        new_all_units_exps[
            self._db_si_units_idx[:self.dimension._prim_dim_count]
        ] = self.dimension.exponents_primary_decomposition
        return Unit(new_all_units_exps)

    @property
    def conversion_coefficients_to_si(self) -> Tuple[float, float]:
        """
        Conversion shift and conversion factor needed to transform the unit into SI units.
        The value of this unit should thus be added to the conversion shift, and then multiplied
        with the conversion factor, in order to obtain the value in SI units, i.e.:
        (value [current-unit] + conversion-shift) * conversion-factor = value [SI-unit]

        Returns
        -------
        (conv_shift, conv_factor) : tuple[float, float]
        """
        # Calculate conversion shift
        # Temperature units are the only units with conversion shifts rather than conversion factors;
        # these are found from their dimension and noted to calculate the conversion shift
        temp_bool_mask = self._db_dim_names == "temperature"
        temp_units_idx = np.argwhere(temp_bool_mask).flatten()
        temp_unit_exps = self._all_units_exps[temp_units_idx]
        temp_unit_conv_shift = self._db_conv_factors[temp_units_idx]
        # Here we are only handling Celsius and Kelvin. Since the change in temperature
        # in Kelvin is equal to that of Celsius, having a temperature unit in the denominator
        # means that there is no conversion shift, i.e. 1/K = 1/°C.
        # Thus we are only interested in temperature units with positive exponents
        positive_temp_unit_exps_idx = np.argwhere(temp_unit_exps > 0)
        positive_temp_unit_exps = temp_unit_exps[positive_temp_unit_exps_idx]
        positive_temp_unit_conv_factors = temp_unit_conv_shift[positive_temp_unit_exps_idx]
        conv_shift = (positive_temp_unit_exps * positive_temp_unit_conv_factors).sum()
        # Calculate conversion factor for all other non-temperature units
        non_temp_units_mask = np.logical_not(temp_bool_mask)
        list_conv_factor = self._db_conv_factors[non_temp_units_mask] ** self._all_units_exps[non_temp_units_mask]
        conv_factor = list_conv_factor[list_conv_factor != 0].prod()
        return conv_shift, conv_factor

    def conversion_coefficients_to(self, unit: Union[str, Unit]) -> Tuple[float, float]:
        """
        Conversion shift and conversion factor needed to transform the unit into another given unit.
        The value of the current unit should thus be added to the conversion shift, and then multiplied
        with the conversion factor, in order to obtain the value in SI units, i.e.:
        (value [current-unit] + conversion-shift) * conversion-factor = value [target-unit]

        Parameters
        ----------
        unit : Union[str, Unit]
            The target unit to convert into.

        Returns
        -------
        (conversion shift, conversion factor): tuple[float, float]
        """
        if isinstance(unit, str):
            unit = Unit(unit)
        elif isinstance(unit, Unit):
            pass
        else:
            raise ValueError("`unit` should either be a string or a Unit object.")
        # Check whether the current unit can be converted into the target unit.
        # see `is_convertible_to` for more info.
        is_convertible, n_factor = self.is_convertible_to(unit, return_n_factor=True)
        if not is_convertible:
            raise ValueError("The current unit's dimension does not match with the target unit.")
        else:
            conv_shift_self, conv_factor_self = self.conversion_coefficients_to_si
            conv_factor_self *= phys_consts["avogadro"]["value"] ** -n_factor
            conv_shift_other, conv_factor_other = unit.conversion_coefficients_to_si
            conv_shift_total = conv_shift_self - conv_shift_other
            conv_factor_total = conv_factor_self / conv_factor_other
        return conv_shift_total, conv_factor_total

    @property
    def convert_to_si(self) -> Tuple[float, float, Unit]:
        """
        Convenience method combining the two methods `conversion_coefficients_to_si_unit` and `si_unit`,
        in order to return both conversion coefficients and the SI unit at the same time.

        Returns
        -------
        (conversion shift, conversion factor, SI unit) : tuple[float, float, Unit]
        """
        conv_shift, conv_factor = self.conversion_coefficients_to_si
        si_unit = self.equiv_unit_si
        return conv_shift, conv_factor, si_unit

    @property
    def convert_to_si_primary(self) -> Tuple[float, float, Unit]:
        """
        Convenience method combining the two methods `conversion_coefficients_to_si_unit` and `si_unit`,
        in order to return both conversion coefficients and the SI unit at the same time.

        Returns
        -------
        (conversion shift, conversion factor, SI unit) : tuple[float, float, Unit]
        """
        conv_shift, conv_factor = self.conversion_coefficients_to_si
        si_unit = self.equiv_unit_si_primary
        return conv_shift, conv_factor, si_unit

    def convert_to(self, unit: Union[str, Unit]) -> Tuple[float, float, Unit]:
        """
        Convenience method to convert the current unit into another given unit,
        by returning both conversion coefficients and the target Unit object at the same time.

        Parameters
        ----------
        unit : Union[str, Unit]
            The target unit to convert into.

        Returns
        -------
        (conversion shift, conversion factor, target unit): tuple[float, float, Unit]
        """
        conv_shit, conv_factor = self.conversion_coefficients_to(unit)
        if isinstance(unit, str):
            converted_unit = Unit(unit)
        else:
            converted_unit = unit
        return conv_shit, conv_factor, converted_unit

    def is_convertible_to(self, unit: Union[str, Unit], return_n_factor=False) -> Union[bool, Tuple[bool, float]]:
        """
        Check whether the current unit is convertible to another unit.
        This is the case for two units with the same primary dimension decomposition,
        disregarding the exponent of the dimension for amount of substance (N).
        In other words, any unit or dimension can be (de)molarized.

        Parameters
        ----------
        unit : Union[str, Unit]
            The dimension for which convertibility should be checked.
        return_n_factor : bool (optional; default: False)
            Whether to return the exponent of the amount-of-substance dimension
            with which the current dimension should be multiplied, in order to
            have the same dimension for amount of substance as the second unit.

        Returns
        -------
            Union[bool, tuple[bool, float]]
            The first element is a boolean telling whether the dimension is convertible at all,
            while the second element (only when `return_N_factor` argument is set to True) is
            the exponent of the amount-of-substance dimension with which the current dimension
            should be multiplied, in order to have the same dimension of amount of substance as the second unit.
        """
        if isinstance(unit, str):
            unit = Unit(unit)
        elif isinstance(unit, Unit):
            pass
        else:
            raise ValueError("Argument `unit` should either be a string or a `Unit` object.")
        # Divide dimensions and take the primary dimension decomposition of the result
        dim_diff = (unit.dimension / self.dimension).exponents_primary_decomposition
        # Get the index of 'amount of substance' dimension
        idx_amount_of_subst_dim = np.argwhere(self.dimension._db_symbols == "N")[0, 0]
        # Note the exponent of the 'amount of substance' dimension
        dim_n_diff = dim_diff[idx_amount_of_subst_dim]
        # Now set it to zero and check if all exponents are now zero
        dim_diff[idx_amount_of_subst_dim] = 0
        # The unit is only convertible if all exponents of the division result is zero
        # not considering the 'amount of substance'.
        is_convertible = np.all(dim_diff == 0)
        if return_n_factor:
            return is_convertible, dim_n_diff
        else:
            return is_convertible

    def modify_prefix(self, exp, target_unit=None, inplace=False) -> Union[Unit, None]:
        """
        Add a prefix to the unit.

        Parameters
        ----------
        exp : int
            Exponent of the prefix to be added; i.e. unit will be multiplied with 10^exp
        target_unit : str (optional; default: None)
            A specific constituting unit for the prefix to be added to.
        inplace : bool
            Whether to modify the object in place

        Returns
        -------
            Union[Unit, None]
            Depending on the value of `inplace`.
        """
        nonzero_nontemp_exps_mask = self._all_units_exps != 0 * self._db_dim_names != "temperature"
        no_prefix_mask = self._db_prefix_exp == 0
        raise NotImplementedError("Method `modify_prefix` is not yet implemented.")
        return

    def has_same_dimension(self, other) -> bool:
        """
        Whether the current unit has the same dimension as another unit or dimension.

        Parameters
        ----------
        other : Union[Unit, Dimension, str]
            Either a string representing a unit, or a Unit or Dimension object.

        Returns
        -------
            bool
        """
        if isinstance(other, Unit):
            return self.dimension == other.dimension
        elif isinstance(other, Dimension):
            return self.dimension == other
        elif isinstance(other, str):
            return self.dimension == Unit(other).dimension
        else:
            raise ValueError("Dimension equality can only be assessed for another Unit or Dimension objects.")

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
                indices[self._prim_unit_count:][::-1],
                # Put the primary dimensions afterwards, but don't change
                # the order within them, since it's already in the conventional order
                indices[: self._prim_unit_count]
            )
        )
        return ordered_indices


class PredefinedUnits:
    """
    Container class for all available known units.
    Units are divided into `primary` and `derived`,
    and then further divided into different dimensions.
    Each available unit is defined as a property in
    this class, which returns a new Unit object for
    that unit, each time it is called.
    """
    @property
    def primary(self):
        return PredefinedUnitsCategory(primary)

    @property
    def derived(self):
        return PredefinedUnitsCategory(derived)


class PredefinedUnitsCategory:
    """
    Container class for all available known primary or derived units.
    Units are divided into different dimensions.
    Each available unit is defined as a property in
    this class, which returns a new Unit object for
    that unit, each time it is called.
    """
    def __init__(self, unit_category_dict):
        def dim_gen(dim_units_dict):
            return PredefinedUnitsInDim(dim_units_dict)
        for dim_name, data in unit_category_dict.items():
            property(setattr(self, dim_name, dim_gen(dim_units_dict=data)))


class PredefinedUnitsInDim:
    """
    Container class for all available known units of specific dimension.
    Each available unit is defined as a property in
    this class, which returns a new Unit object for
    that unit, each time it is called.
    """
    def __init__(self, dim_units_dict):
        def unit_gen(unit_symbol):
            return Unit(unit_symbol)
        for unit_name, data in dim_units_dict["units"].items():
            property(setattr(self, unit_name, unit_gen(data["symbol"])))


# Instantiate the container class,
# this can now be directly imported whenever needed.
predefined = PredefinedUnits()
