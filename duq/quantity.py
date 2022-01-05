from __future__ import annotations
from typing import Union, Tuple
from functools import partial

import numpy as np

from .unit import Unit
from .dimension import Dimension
from .helpers import raise_for_type
from .data.physical_constants import phys_consts


class Quantity:
    """
    An object representing a physical quantity, which has a value, a unit,
    and consequently, a dimension.

    Parameters
    ----------
    value : Union[int, float, numpy.number]
        The numerical value of the physical quantity.
    unit : Union[str, unit.Unit]
        The unit of the physical quantity.
        If a string, it may be composed of a number of units, separated
        by a `.`. Each constituting unit may have an exponent, denoted
        by a '^'. The unit may also be written as a single fraction,
        where the nominator and denominator are separated by a '/'.
        Example:
        'kg.m^2/s^2' is equal to 'kg.m^2.s^-2', and both are acceptable.
    normalization : bool (optional; default: False)
        Whether to normalize the value, i.e. bring it to the range [1, 10),
        and add the resulting exponent to the unit.
        Example:
        1234 g will be transformed into 1.234 kg.
    """

    @classmethod
    def supported_input_units(cls) -> Tuple:
        """
        Returns a tuple of names and symbols of supported units
        available for constructing a Quantity object.
        """
        return Unit.supported_input_units()

    def __init__(
            self,
            value: Union[int, float, np.number],
            unit: Union[str, Unit],
            normalization: bool = False
    ):

        # Verify the type of `value` and assign if correct.
        if isinstance(value, (int, float, np.number)):
            self._value = value
        else:
            raise ValueError("Type of `value` should either be int, float, or numpy.number.")

        # Verify the type of `unit` and assign if correct.
        if isinstance(unit, str):
            self._unit = Unit(unit)
        elif isinstance(unit, Unit):
            self._unit = unit
        else:
            raise NotImplementedError("Acceptable types for `unit` are string and unit.Unit.")

        # Verify the type of `normalize` and assign if correct.
        if isinstance(normalization, bool):
            self._normalization = normalization
        else:
            raise ValueError("Type of `normalize` should be boolean.")

        # Run initial normalization if `normalize` is True.
        if self.normalization:
            self.normalize(inplace=True)
        else:
            pass
        return

    def __repr__(self):
        return f"Quantity(value={self.value}, unit={repr(self.unit)}, normalization={self.normalization})"

    def __str__(self):
        # Transform value into string in scientific notation with 10 decimals
        val_sci_not = '{:.10E}'.format(self.value)
        # Separate mantissa and exponent
        mantissa, exp = val_sci_not.split('E')
        # Strip mantissa of trailing zeros
        zero_stripped_mantissa = mantissa.rstrip('0').rstrip('.')
        # Reformat into scientific notation string, by adding 'E' and exponent at the end
        zero_stripped_val_sci_not = zero_stripped_mantissa + 'E' + exp
        return f"{zero_stripped_val_sci_not} {self.unit.symbol_as_is}\n\n{str(self.unit)}"

    def __eq__(self, other: Quantity):
        raise_for_type(other, Quantity, "Equality can only be assessed between two `Quantity` objects.")
        try:
            other_in_self_units = other.convert_unit(self.unit)
        except ValueError as e:
            print(e)
            return False
        return np.isclose(self.value, other_in_self_units.value)

    def __add_common__(self, other, sign):
        # Common operations between __add__, __sub__, __iadd__ and __isub__
        raise_for_type(other, Quantity, "Addition can only be performed on another PhysicalQuantity object.")
        if other.unit == self.unit:
            new_value = self.value + sign * other.value
        else:
            try:
                other_in_self_units = other.convert_unit(self.unit)
                new_value = self.value + sign * other_in_self_units.value
            except ValueError:
                raise ValueError("Addends' units are not interconvertible.")
        return new_value

    def __add__(self, other):
        new_value = self.__add_common__(other, 1)
        return Quantity(new_value, self.unit)

    def __radd__(self, other):
        raise NotImplementedError("Addition can only be performed on another PhysicalQuantity object.")

    def __iadd__(self, other):
        self._value = self.__add_common__(other, 1)
        if self.normalization:
            self.normalize()
        return self

    def __sub__(self, other):
        new_value = self.__add_common__(other, -1)
        return Quantity(new_value, self.unit)

    def __rsub__(self, other):
        raise NotImplementedError("Addition can only be performed on another PhysicalQuantity object.")

    def __isub__(self, other):
        self._value = self.__add_common__(other, -1)
        if self.normalization:
            self.normalize()
        return self

    def __mul_common__(self, other):
        if isinstance(other, (int, float)):
            new_value = self.value * other
            new_unit = self.unit
        elif isinstance(other, Quantity):
            new_value = self.value * other.value
            new_unit = self.unit * other.unit
        else:
            raise NotImplementedError("Multiplicand should either be int, float, or a PhysicalQuantity object.")
        return new_value, new_unit

    def __mul__(self, other):
        new_value, new_unit = self.__mul_common__(other)
        return Quantity(new_value, new_unit)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self._value, self._unit = self.__mul_common__(other)
        if self.normalization:
            self.normalize()
        return self

    def __pow_common__(self, power):
        new_value = self.value ** power
        new_unit = self.unit ** power
        return new_value, new_unit

    def __pow__(self, power):
        new_value, new_unit = self.__pow_common__(power)
        return Quantity(new_value, new_unit)

    def __ipow__(self, power):
        self._value, self._unit = self.__pow_common__(power)
        if self.normalization:
            self.normalize()
        return self

    def __truediv_common__(self, other):
        if isinstance(other, (int, float)):
            new_value = self.value / other
            new_unit = self.unit
        elif isinstance(other, Quantity):
            new_value = self.value / other.value
            new_unit = self.unit / other.unit
        else:
            raise NotImplementedError("Dividend should either be int, float, or a PhysicalQuantity object.")
        return new_value, new_unit

    def __truediv__(self, other):
        new_value, new_unit = self.__truediv_common__(other)
        return Quantity(new_value, new_unit)

    def __rtruediv__(self, other):
        new_quant = self.__pow__(-1)
        return new_quant.__imul__(other)

    def __itruediv__(self, other):
        self._value, self._unit = self.__truediv_common__(other)
        if self.normalization:
            self.normalize()
        return self

    @property
    def value(self) -> Union[int, float, np.number]:
        """
        The numerical value of the physical quantity.
        """
        return self._value

    @property
    def unit(self) -> Unit:
        """
        unit.Unit object representing the unit of the quantity.
        """
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """
        dimension.Dimension object representing the dimension of the quantity.
        """
        return self.unit.dimension

    @property
    def is_in_si_unit(self) -> bool:
        """
        Whether the quantity is in SI units.
        """
        return self.unit.is_si_unit

    @property
    def normalization(self) -> bool:
        """
        Whether normalization is turned on for the quantity.
        If True, the value will be normalized each time it changes, i.e.
        it is brought to the range [1, 10), and the resulting exponent
        is added to the unit.
        Example:
        1234 g will be transformed into 1.234 kg.
        """
        return self._normalization

    def toggle_normalization(self) -> None:
        """
        Turn normalization on/off.
        When turned on, the value be normalized immediately afterwards,
        and also each time it changes.

        Returns
        -------
            None
        """
        self._normalization = not self._normalization
        if self.normalization:
            self.normalize(inplace=True)
        return

    def normalize(self, inplace: bool = False) -> Union[Quantity, None]:
        """
        Normalize the value, i.e. bring it to the range [1, 10),
        and add the resulting exponent to the unit.

        Parameters
        ----------
        inplace : bool
            Whether to apply normalization to the current object.

        Returns
        -------
            Union[Quantity, None]
            If `inplace` it True, returns None;
            If False, a new PhysicalQuantity object is returned.

        Examples
        --------
        1234 g will be transformed into 1.234 kg.
        """

        # Calculate the exponent of the value, when the value
        # is written with a mantissa in the range [1,10).
        exp = int(np.floor(np.log10(abs(self.value))))
        # Update value and unit accordingly
        new_value = self._value * (10 ** -exp) if exp != 0 else self._value
        new_unit = self.unit.modify_prefix(exp, inplace=False) if exp != 0 else self.unit
        if inplace:
            self._value, self._unit = new_value, new_unit
        else:
            return Quantity(new_value, new_unit)

    def convert_unit_to_si(self, inplace: bool = False) -> Union[Quantity, None]:
        """
        Convert the quantity to SI units.

        Parameters
        ----------
        inplace : bool
            Whether to apply conversion to the current object.

        Returns
        -------
            Union[Quantity, None]
            If `inplace` it True, returns None;
            If False, a new PhysicalQuantity object is returned.
        """
        return self._convert_unit(self.unit.convert_to_si, inplace)

    def convert_unit(self, new_unit: Union[str, Unit], inplace: bool = False) -> Union[Quantity, None]:
        """
        Convert the quantity to another unit.

        Parameters
        ----------
        new_unit : Union[str, Unit]
            The unit to convert into.
        inplace : bool
            Whether to apply conversion to the current object.

        Returns
        -------
            Union[Quantity, None]
            If `inplace` it True, returns None;
            If False, a new PhysicalQuantity object is returned.
        """
        return self._convert_unit(self.unit.convert_to(new_unit), inplace)

    def _convert_unit(self, conversion_results: Tuple[float, float, Unit], inplace: bool):
        """
        Base function used by both `convert_unit` and `convert_unit_to_si` functions.
        """
        conv_shift, conv_factor, new_unit = conversion_results
        new_value = (self._value + conv_shift) * conv_factor
        if inplace:
            self._value = new_value
            self._unit = new_unit
        else:
            return Quantity(new_value, new_unit)


class PhysicalConstants:
    """
    Container class for all available known physical constants as `Quantity` object.
    Each constant is defined as a property in this class, which returns a new
    `Quantity` object for that constant, each time it is called.
    """
    def __init__(self, phys_consts_dict):
        def const_gen(self, value, unit):
            return Quantity(value, Unit(unit))
        for const_name, data in phys_consts_dict.items():
            setattr(
                PhysicalConstants,
                const_name,
                property(
                    partial(
                        const_gen,
                        value=data["value"],
                        unit=data["unit"]
                    )
                )
            )


# Instantiate the container class,
# this can now be directly imported whenever needed.
constants = PhysicalConstants(phys_consts)
