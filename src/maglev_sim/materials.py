"""Material data and unit helpers.

Convention: YAML configs use readable units with the unit in the key name
(``pitch_mm``, ``extra_mass_g``, ``max_current_A``). Everything inside the code
is SI (metres, kilograms, amps, tesla).
"""

from __future__ import annotations

import math

MU0 = 4e-7 * math.pi
G = 9.81  # m/s^2

# Remanence Br [T] (typical mid values) of sintered NdFeB grades.
NDFEB_BR = {
    "N35": 1.19,
    "N38": 1.24,
    "N40": 1.27,
    "N42": 1.30,
    "N45": 1.34,
    "N48": 1.39,
    "N50": 1.42,
    "N52": 1.45,
}
NDFEB_DENSITY = 7500.0  # kg/m^3

COPPER_RESISTIVITY_20C = 1.72e-8  # ohm*m
COPPER_TEMP_COEFF = 0.00393  # 1/K
COPPER_DENSITY = 8960.0  # kg/m^3
COPPER_HEAT_CAPACITY = 385.0  # J/(kg*K)
ENAMEL_BUILD = 1.07  # insulated diameter / bare diameter (grade-1 magnet wire, approx.)


def mm(x: float) -> float:
    return float(x) * 1e-3


def grams(x: float) -> float:
    return float(x) * 1e-3


def remanence(grade: str | float) -> float:
    """Br in tesla for a grade name like 'N42' or a number given directly."""
    if isinstance(grade, (int, float)):
        return float(grade)
    key = str(grade).upper()
    if key not in NDFEB_BR:
        raise KeyError(f"Unknown magnet grade '{grade}'. Use one of {sorted(NDFEB_BR)} or a number in tesla.")
    return NDFEB_BR[key]


def awg_diameter(awg: int) -> float:
    """Bare copper diameter [m] for an AWG wire gauge."""
    return 0.127e-3 * 92 ** ((36 - awg) / 39)


def copper_resistance(length: float, diameter: float, temp_c: float = 20.0) -> float:
    area = math.pi * diameter**2 / 4
    rho = COPPER_RESISTIVITY_20C * (1 + COPPER_TEMP_COEFF * (temp_c - 20.0))
    return rho * length / area
