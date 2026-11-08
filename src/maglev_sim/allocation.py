"""Current allocation: turn a wanted wrench into coil currents.

With wrench = K @ I there are usually more coils than wrench components, so many
current patterns give the same wrench. We pick the one with the least power
(sum of I^2, optionally weighted per coil). Weights are how tile handoff works:
a coil with weight 0 is off, a coil with weight 1 is fully available, and in
between it is "fading" (see handoff.py).

Also here:
  * condition_number - how evenly a layout can push in every direction
    (Berkelman 2013 used it to compare coil layouts; lower is better, inf = some
    direction cannot be controlled at all).
  * max_wrench - the most force along one axis the coils can make within the
    current limit while holding the other axes at given values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, lsq_linear

TORQUE_ROWS = (3, 4, 5)


def row_scale(length: float) -> np.ndarray:
    """Divide torque rows by a length so N and N*m rows are comparable."""
    d = np.ones(6)
    d[list(TORQUE_ROWS)] = 1.0 / max(length, 1e-6)
    return d


@dataclass
class Allocation:
    currents: np.ndarray  # (C,) amps
    achieved: np.ndarray  # (6,) wrench actually produced (K @ currents)
    error: float  # norm of (achieved - wanted) on the controlled axes, scaled units [N]
    saturated: bool  # True if the current limit forced a compromise

    @property
    def max_abs_current(self) -> float:
        return float(np.max(np.abs(self.currents))) if self.currents.size else 0.0


def allocate(
    K: np.ndarray,
    wanted: np.ndarray,
    axes: list[int],
    weights: np.ndarray | None = None,
    i_max: float | None = None,
    length: float = 0.02,
    damping: float = 0.003,
) -> Allocation:
    """Minimum-power currents that produce ``wanted`` on the ``axes`` rows of K.

    ``damping`` (relative to the strongest direction) regularises near-singular
    positions; 0 gives the exact pseudo-inverse."""
    C = K.shape[1]
    w = np.ones(C) if weights is None else np.asarray(weights, float)
    active = w > 1e-9
    I = np.zeros(C)
    D = row_scale(length)[axes]
    Ka = K[np.ix_(axes, np.flatnonzero(active))] * D[:, None]
    target = np.asarray(wanted, float)[axes] * D
    sw = np.sqrt(w[active])
    saturated = False
    if Ka.size:
        A = Ka * sw  # in scaled variables y = I / sqrt(w)
        # Damped least squares: like the exact minimum-power solution, but near
        # positions where some direction can barely be pushed (tiny singular
        # value) it accepts a small wrench error instead of huge, jumping currents.
        lam = (damping * np.linalg.norm(A, 2)) ** 2
        y = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(A.shape[0]), target)
        Ia = sw * y
        if i_max is not None and np.max(np.abs(Ia)) > i_max * (1 + 1e-9):
            # The least-power pattern breaks the limit; look for any pattern within it.
            reg = 1e-9 * max(np.linalg.norm(A, 2) ** 2, 1e-30)
            n = A.shape[1]
            Aaug = np.vstack([A, np.sqrt(reg) * np.eye(n)])
            baug = np.concatenate([target, np.zeros(n)])
            lim = i_max / sw
            y = lsq_linear(Aaug, baug, bounds=(-lim, lim), method="bvls").x
            Ia = sw * y
            # saturated = the limit really prevents reaching the wanted wrench
            saturated = np.linalg.norm(A @ y - target) > 1e-3 * max(np.linalg.norm(target), 1e-12)
        I[active] = Ia
    achieved = K @ I
    err = float(np.linalg.norm((achieved - np.asarray(wanted))[axes] * D))
    return Allocation(I, achieved, err, saturated)


def condition_number(K: np.ndarray, axes: list[int], weights=None, length: float = 0.02) -> float:
    w = np.ones(K.shape[1]) if weights is None else np.asarray(weights, float)
    active = w > 1e-9
    if active.sum() < len(axes):
        return float("inf")
    A = K[np.ix_(axes, np.flatnonzero(active))] * row_scale(length)[axes][:, None] * np.sqrt(w[active])
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 1e-12 * s[0] else float("inf")


def max_wrench(
    K: np.ndarray,
    axis: int,
    axes: list[int],
    hold: np.ndarray,
    i_max: float,
    weights=None,
) -> float:
    """Largest value of wrench[axis] reachable with |I| <= i_max, while every
    other controlled axis equals ``hold`` (e.g. zero torque, Fz = weight).
    Returns -inf if even the held values cannot be reached."""
    w = np.ones(K.shape[1]) if weights is None else np.asarray(weights, float)
    idx = np.flatnonzero(w > 1e-9)
    if idx.size == 0:
        return float("-inf")
    Ka = K[:, idx]
    others = [a for a in axes if a != axis]
    res = linprog(
        -Ka[axis],
        A_eq=Ka[others] if others else None,
        b_eq=np.asarray(hold, float)[others] if others else None,
        bounds=[(-i_max, i_max)] * idx.size,
        method="highs",
    )
    return float(-res.fun) if res.status == 0 else float("-inf")
