"""Tile handoff: fading coils in and out as the mover travels (Jansen 2008).

Every coil gets a weight from its distance to the mover centre:
    1 inside ``fade_inner_mm``, 0 beyond ``fade_outer_mm``, a smooth cosine ramp
between. The allocator uses these weights, so coils join and leave gradually and
currents never jump when the mover crosses from one tile to the next.

``straight_line_sweep`` moves the mover slowly (no dynamics) along a line and
records currents, power and how smooth the commutation is - the quickest way
to judge a handoff setting before running the full simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation import allocate
from .materials import mm


def fade_weights(coil_xy: np.ndarray, mover_xy, inner: float, outer: float) -> np.ndarray:
    d = np.linalg.norm(coil_xy - np.asarray(mover_xy)[:2], axis=1)
    t = np.clip((d - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * t))


def system_weights(system, mover_xy) -> np.ndarray:
    h = system.cfg["handoff"]
    return fade_weights(system.grid.coil_xy, mover_xy, mm(h["fade_inner_mm"]), mm(h["fade_outer_mm"]))


def straight_line_sweep(system, start_xy, end_xy, n: int = 60, extra_wrench=None) -> pd.DataFrame:
    """Quasi-static path: at each point, allocate hover (+ optional extra wrench)."""
    start_xy, end_xy = np.asarray(start_xy, float), np.asarray(end_xy, float)
    s = np.linspace(0, 1, n)
    xy = start_xy[None] + s[:, None] * (end_xy - start_xy)[None]
    poses = np.column_stack([xy, np.full(n, system.gap), np.zeros((n, 3))])
    Ks = system.wrench_matrices(poses)
    wanted = system.hover_wrench() + (np.zeros(6) if extra_wrench is None else np.asarray(extra_wrench))
    rows, prev = [], None
    for k in range(n):
        w = system_weights(system, xy[k])
        a = allocate(Ks[k], wanted, system.control_axes, w, system.max_current, system.mover.size)
        step = 0.0 if prev is None else float(np.max(np.abs(a.currents - prev)))
        prev = a.currents
        rows.append(
            {
                "x_mm": xy[k, 0] * 1e3,
                "y_mm": xy[k, 1] * 1e3,
                "tile": system.grid.tile_at(xy[k]),
                "active_coils": int((w > 1e-9).sum()),
                "max_current_A": a.max_abs_current,
                "power_W": float(np.sum(a.currents**2) * system.coil.resistance),
                "force_error_N": a.error,
                "saturated": a.saturated,
                "current_step_A": step,
                **{f"I{c}": a.currents[c] for c in range(system.n_coils)},
            }
        )
    return pd.DataFrame(rows)
