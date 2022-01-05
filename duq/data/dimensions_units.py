"""
Data on primary and derived physical dimensions and units.
"""

import numpy as np

from .metric_prefixes import greater_one, smaller_one


primary = {
    "mass":
        {
            "name": "mass",
            "symbol": "M",
            "prim_dim_exps": np.array([1, 0, 0, 0, 0, 0, 0]),
            "units":
                {
                    "kilogram":
                        {
                            "name": "kilogram",
                            "symbol": "kg",
                            "conv_factor": 1,
                            "prefix_exp": 3
                        },
                    "gram":
                        {
                            "name": "gram",
                            "symbol": "g",
                            "conv_factor": 1e-3,
                            "prefix_exp": 0
                        },
                    "dalton":
                        {
                            "name": "dalton",
                            "symbol": "Da",
                            "conv_factor": 1.6605390666e-27,
                            "prefix_exp": 0
                        },
                    "electron_mass":
                        {
                            "name": "atomic unit of mass (a.u.)",
                            "symbol": "m_e",
                            "conv_factor": 9.1093837015e-31,
                            "prefix_exp": 0
                        }
                }
        },

    "length":
        {
            "name": "length",
            "symbol": "L",
            "prim_dim_exps": np.array([0, 1, 0, 0, 0, 0, 0]),
            "units":
                {
                    "metre":
                        {
                            "name": "metre",
                            "symbol": "m",
                            "conv_factor": 1,
                            "prefix_exp": 0
                        },
                    "angstrom":
                        {
                            "name": "angstrom",
                            "symbol": "Å",
                            "conv_factor": 1e-10,
                            "prefix_exp": 0
                        },
                    "bohr":
                        {
                            "name": "bohr radius (a.u.)",
                            "symbol": "a0",
                            "conv_factor": 5.29177210903e-11,
                            "prefix_exp": 0
                        }
                }
        },

    "time":
        {
            "name": "time",
            "symbol": "T",
            "prim_dim_exps": np.array([0, 0, 1, 0, 0, 0, 0]),
            "units":
                {
                    "second":
                        {
                            "name": "second",
                            "symbol": "s",
                            "conv_factor": 1,
                            "prefix_exp": 0
                        },
                }
        },

    "electric_current":
        {
            "name": "electric current",
            "symbol": "I",
            "prim_dim_exps": np.array([0, 0, 0, 1, 0, 0, 0]),
            "units":
                {
                    "ampere":
                        {
                            "name": "ampere",
                            "symbol": "A",
                            "conv_factor": 1,
                            "prefix_exp": 0
                        }
                }
        },

    "temperature":
        {
            "name": "temperature",
            "symbol": "Θ",
            "prim_dim_exps": np.array([0, 0, 0, 0, 1, 0, 0]),
            "units":
                {
                    "kelvin":
                        {
                            "name": "kelvin",
                            "symbol": "K",
                            "conv_factor": 0,
                            "prefix_exp": 0
                        },
                    "celsius":
                        {
                            "name": "degree Celsius",
                            "symbol": "°C",
                            "conv_factor": 273.15,
                            "prefix_exp": 0
                        }
                }
        },

    "amount_substance":
        {
            "name": "amount of substance",
            "symbol": "N",
            "prim_dim_exps": np.array([0, 0, 0, 0, 0, 1, 0]),
            "units":
                {
                    "mole":
                        {
                            "name": "mole",
                            "symbol": "mol",
                            "conv_factor": 1,
                            "prefix_exp": 0
                        }
                }
        },

    "luminous_intensity":
        {
            "name": "luminous intensity",
            "symbol": "J",
            "prim_dim_exps": np.array([0, 0, 0, 0, 0, 0, 1]),
            "units":
                {
                    "candela":
                        {
                            "name": "candela",
                            "symbol": "cd",
                            "conv_factor": 1,
                            "prefix_exp": 0
                        }
                }
        },
}

derived = {
    "dimensionless":
        {
            "name": "dimensionless",
            "symbol": "1",
            "prim_dim_exps": np.array([0, 0, 0, 0, 0, 0, 0]),
            "units":
                {
                    "radian":
                        {
                            "name": "radian",
                            "symbol": "rad",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        },
                    "degree":
                        {
                            "name": "degree",
                            "symbol": "deg",
                            "conv_factor": 1.74533e-2,
                            "prefix_exp": 0,
                        }
                }
        },

    "area":
        {
            "name": "area",
            "symbol": "Ar",
            "prim_dim_exps": np.array([0, 2, 0, 0, 0, 0, 0]),
            "units":
                {
                    "square_metre":
                        {
                            "name": "square metre",
                            "symbol": "m^2",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "volume":
        {
            "name": "volume",
            "symbol": "Vol",
            "prim_dim_exps": np.array([0, 3, 0, 0, 0, 0, 0]),
            "units":
                {
                    "cubic_metre":
                        {
                            "name": "cubic metre",
                            "symbol": "m^3",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "frequency":
        {
            "name": "frequency",
            "symbol": "ν",
            "prim_dim_exps": np.array([0, 0, -1, 0, 0, 0, 0]),
            "units":
                {
                    "hertz":
                        {
                            "name": "hertz",
                            "symbol": "Hz",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "density":
        {
            "name": "density",
            "symbol": "ρ",
            "prim_dim_exps": np.array([1, -3, 0, 0, 0, 0, 0]),
            "units":
                {
                    "kilogram_per_cubic_metre":
                        {
                            "name": "kilogram per cubic metre",
                            "symbol": "kg.m^-3",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "pressure":
        {
            "name": "pressure",
            "symbol": "P",
            "prim_dim_exps": np.array([1, -1, -2, 0, 0, 0, 0]),
            "units":
                {
                    "pascal":
                        {
                            "name": "pascal",
                            "symbol": "Pa",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "electric_charge":
        {
            "name": "electric charge",
            "symbol": "Q",
            "prim_dim_exps": np.array([0, 0, 1, 1, 0, 0, 0]),
            "units":
                {
                    "coulomb":
                        {
                            "name": "Coulomb",
                            "symbol": "C",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        },
                    "elementary_charge":
                        {
                            "name": "atomic unit of charge (a.u.)",
                            "symbol": "e",
                            "conv_factor": 1.602176634e-19,
                            "prefix_exp": 0,
                        },
                }
        },

    "velocity":
        {
            "name": "velocity",
            "symbol": "V",
            "prim_dim_exps": np.array([0, 1, -1, 0, 0, 0, 0]),
            "units":
                {
                    "metre_per_second":
                        {
                            "name": "metre per second",
                            "symbol": "m.s^-1",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "momentum":
        {
            "name": "momentum",
            "symbol": "Mom",
            "prim_dim_exps": np.array([1, 1, -1, 0, 0, 0, 0]),
            "units":
                {
                    "kilogram_metre_per_second":
                        {
                            "name": "kilogram metre per second",
                            "symbol": "kg.m.s^-1",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "acceleration":
        {
            "name": "acceleration",
            "symbol": "A",
            "prim_dim_exps": np.array([0, 1, -2, 0, 0, 0, 0]),
            "units":
                {
                    "metre_per_square_second":
                        {
                            "name": "metre per second squared",
                            "symbol": "m.s^-2",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "force":
        {
            "name": "force",
            "symbol": "F",
            "prim_dim_exps": np.array([1, 1, -2, 0, 0, 0, 0]),
            "units":
                {
                    "newton":
                        {
                            "name": "newton",
                            "symbol": "N",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        }
                }
        },

    "energy":
        {
            "name": "energy",
            "symbol": "E",
            "prim_dim_exps": np.array([1, 2, -2, 0, 0, 0, 0]),
            "units":
                {
                    "joule":
                        {
                            "name": "joule",
                            "symbol": "J",
                            "conv_factor": 1,
                            "prefix_exp": 0,
                        },
                    "kilocalorie":
                        {
                            "name": "kilocalorie",
                            "symbol": "kcal",
                            "conv_factor": 4184,
                            "prefix_exp": 3,
                        },
                    "electronvolt":
                        {
                            "name": "electronvolt",
                            "symbol": "eV",
                            "conv_factor": 1.602176634e-19,
                            "prefix_exp": 0,
                        },
                    "hartree":
                        {
                            "name": "hartree (a.u.)",
                            "symbol": "E_h",
                            "conv_factor": 4.3597447222071e-18,
                            "prefix_exp": 0,
                        }
                }
        },
}


# Creating more units from prefixes
for prefix in ["centi", "milli", "micro", "nano", "pico", "femto", "atto"]:
    pre = smaller_one[prefix]
    p_symb = pre['symbol']
    p_name = pre['name']
    p_exp = pre["exp"]

    # for length, from metre
    primary["length"]["units"][f"{p_name}metre"] = {
        "name": f"{p_name}metre",
        "symbol": f"{pre['symbol']}m",
        "conv_factor": p_exp,
        "prefix_exp": p_exp
    }
    # for time, from second
    primary["time"]["units"][f"{p_name}second"] = {
        "name": f"{p_name}second",
        "symbol": f"{p_symb}s",
        "conv_factor": p_exp,
        "prefix_exp": p_exp
    }
