"""Control: reference trajectories and the position controller.

The fast loop (lit review: hundreds of Hz to ~1 kHz) is a PD controller per axis
with gravity feed-forward:

    F_wanted = m * (a_ref + wn^2 (p_ref - p) + 2 zeta wn (v_ref - v)) + m g z_hat

Gains are set from a closed-loop bandwidth in Hz (``control.bandwidth_Hz``)
rather than raw kp/kd, which is easier to reason about: higher = stiffer and
faster, but needs more current and a faster, less noisy sensor.
Torques are commanded to zero (keep the mover level and un-rotated).

To try a different controller (PID, LQR, ...), write a class with the same
``wrench(...)`` method and pass it to ``simulate(controller=...)``.
"""

from __future__ import annotations

import numpy as np

from .materials import G


class PDController:
    def __init__(self, system, bandwidth_Hz=None, damping=None):
        c = system.cfg["control"]
        bw = np.broadcast_to(np.asarray(bandwidth_Hz if bandwidth_Hz is not None else c["bandwidth_Hz"], float), (3,))
        self.wn = 2 * np.pi * bw
        self.zeta = float(damping if damping is not None else c["damping"])
        self.m = system.mover.mass

    def wrench(self, p, v, p_ref, v_ref, a_ref) -> np.ndarray:
        acc = a_ref + self.wn**2 * (p_ref - p) + 2 * self.zeta * self.wn * (v_ref - v)
        w = np.zeros(6)
        w[:3] = self.m * acc
        w[2] += self.m * G
        return w


def min_jerk(p0, p1, T, t):
    """Smooth point-to-point move (zero velocity and acceleration at both ends)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s = np.clip(t / T, 0, 1) if T > 0 else 1.0
    d = p1 - p0
    pos = p0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel = d * (30 * s**2 - 60 * s**3 + 30 * s**4) / T if 0 < s < 1 else np.zeros(3)
    acc = d * (60 * s - 180 * s**2 + 120 * s**3) / T**2 if 0 < s < 1 else np.zeros(3)
    return pos, vel, acc


class Trajectory:
    """Waypoints ``[(time_s, [x, y, z]), ...]`` (metres), min-jerk between them."""

    def __init__(self, waypoints):
        self.wp = sorted([(float(t), np.asarray(p, float)) for t, p in waypoints], key=lambda w: w[0])

    @classmethod
    def hold(cls, p):
        return cls([(0.0, p)])

    @classmethod
    def move(cls, p0, p1, t_start=0.2, duration=0.6):
        return cls([(0.0, p0), (t_start, p0), (t_start + duration, p1)])

    def __call__(self, t):
        if t <= self.wp[0][0] or len(self.wp) == 1:
            return self.wp[0][1], np.zeros(3), np.zeros(3)
        for (t0, p0), (t1, p1) in zip(self.wp[:-1], self.wp[1:]):
            if t <= t1:
                return min_jerk(p0, p1, t1 - t0, t - t0)
        return self.wp[-1][1], np.zeros(3), np.zeros(3)
