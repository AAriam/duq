"""
Dictionaries containing metric prefixes, their names, symbols, and factor.
These are also categorized into two dictionaries `greater_one` and `smaller_one`.
"""

yotta = {
    "name": "yotta",
    "symbol": "Y",
    "exp": 24
}

zetta = {
    "name": "zetta",
    "symbol": "Z",
    "exp": 21
}

exa = {
    "name": "exa",
    "symbol": "E",
    "exp": 18
}

peta = {
    "name": "peta",
    "symbol": "P",
    "exp": 15
}

tera = {
    "name": "tera",
    "symbol": "T",
    "exp": 12
}

giga = {
    "name": "giga",
    "symbol": "G",
    "exp": 9
}

mega = {
    "name": "mega",
    "symbol": "M",
    "exp": 6
}

kilo = {
    "name": "kilo",
    "symbol": "k",
    "exp": 3
}

hecto = {
    "name": "hecto",
    "symbol": "h",
    "exp": 2
}

deca = {
    "name": "deca",
    "symbol": "da",
    "exp": 1
}

deci = {
    "name": "deci",
    "symbol": "d",
    "exp": -1
}

centi = {
    "name": "centi",
    "symbol": "c",
    "exp": -2
}

milli = {
    "name": "milli",
    "symbol": "m",
    "exp": -3
}

micro = {
    "name": "micro",
    "symbol": "μ",
    "exp": -6
}

nano = {
    "name": "nano",
    "symbol": "n",
    "exp": -9
}

pico = {
    "name": "pico",
    "symbol": "p",
    "exp": -12
}

femto = {
    "name": "femto",
    "symbol": "f",
    "exp": -15
}

atto = {
    "name": "atto",
    "symbol": "a",
    "exp": -18
}

zepto = {
    "name": "zepto",
    "symbol": "z",
    "exp": -21
}

yocto = {
    "name": "yocto",
    "symbol": "y",
    "exp": -24
}

greater_one = {
    "yotta": yotta,
    "zetta": zetta,
    "exa": exa,
    "peta": peta,
    "tera": tera,
    "giga": giga,
    "mega": mega,
    "kilo": kilo,
    "hecto": hecto,
    "deca": deca
}

smaller_one = {
    "deci": deci,
    "centi": centi,
    "milli": milli,
    "micro": micro,
    "nano": nano,
    "pico": pico,
    "femto": femto,
    "atto": atto,
    "zepto": zepto,
    "yocto": yocto
}