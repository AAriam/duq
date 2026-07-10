"""
Dictionaries containing metric prefixes, their names, symbols, and factor.
These are also categorized into two dictionaries `greater_one` and `smaller_one`.
"""

yotta = {"name": "yotta", "symbol": "Y", "factor": 1e24}

zetta = {"name": "zetta", "symbol": "Z", "factor": 1e21}

exa = {"name": "exa", "symbol": "E", "factor": 1e18}

peta = {"name": "peta", "symbol": "P", "factor": 1e15}

tera = {"name": "tera", "symbol": "T", "factor": 1e12}

giga = {"name": "giga", "symbol": "G", "factor": 1e9}

mega = {"name": "mega", "symbol": "M", "factor": 1e6}

kilo = {"name": "kilo", "symbol": "k", "factor": 1e3}

hecto = {"name": "hecto", "symbol": "h", "factor": 1e2}

deca = {"name": "deca", "symbol": "da", "factor": 1e1}

deci = {"name": "deci", "symbol": "d", "factor": 1e-1}

centi = {"name": "centi", "symbol": "c", "factor": 1e-2}

milli = {"name": "milli", "symbol": "m", "factor": 1e-3}

micro = {"name": "micro", "symbol": "μ", "factor": 1e-6}

nano = {"name": "nano", "symbol": "n", "factor": 1e-9}

pico = {"name": "pico", "symbol": "p", "factor": 1e-12}

femto = {"name": "femto", "symbol": "f", "factor": 1e-15}

atto = {"name": "atto", "symbol": "a", "factor": 1e-18}

zepto = {"name": "zepto", "symbol": "z", "factor": 1e-21}

yocto = {"name": "yocto", "symbol": "y", "factor": 1e-24}

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
    "deca": deca,
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
    "yocto": yocto,
}
