"""Coil heating: the main practical limit for air-cooled levitation (lit review takeaway).

Simple lumped model per coil: all I^2 R heat leaves through the coil's outer
surface by convection,   T = T_ambient + P / (h * A).
Copper resistance rises ~0.4 %/K, which we include by iterating.
It ignores heat spreading into the tile/PCB and between coils, so treat it as an
order-of-magnitude check; measure a real coil before trusting the last 20 %.
"""

from __future__ import annotations

import numpy as np

from .coils import CoilModel


def steady_state(currents, coil: CoilModel, thermal_cfg: dict):
    """Steady temperature [C] and power [W] of each coil running a constant current."""
    I = np.abs(np.asarray(currents, float))
    h, area = thermal_cfg["h_conv_W_m2K"], coil.surface_area
    T = np.full(I.shape, float(thermal_cfg["ambient_C"]))
    for _ in range(8):
        P = I**2 * coil.resistance_at(np.minimum(T, RUNAWAY_C))  # element-wise on arrays
        T = thermal_cfg["ambient_C"] + P / (h * area)
    # Beyond ~RUNAWAY_C the coil would burn (and the model is meaningless): cap it.
    return np.minimum(T, RUNAWAY_C), P


RUNAWAY_C = 400.0  # reported temperatures are capped here ("would burn")


def max_continuous_current(coil: CoilModel, thermal_cfg: dict) -> float:
    """Current a coil can carry forever without passing max_coil_temp_C."""
    t_max = thermal_cfg["max_coil_temp_C"]
    dT = t_max - thermal_cfg["ambient_C"]
    return float(np.sqrt(dT * thermal_cfg["h_conv_W_m2K"] * coil.surface_area / coil.resistance_at(t_max)))


def thermal_time_constant(coil: CoilModel, thermal_cfg: dict) -> float:
    """Seconds for a coil to get ~63 % of the way to its steady temperature."""
    return coil.heat_capacity / (thermal_cfg["h_conv_W_m2K"] * coil.surface_area)
