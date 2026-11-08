"""The standard design report: the same numbers for every design, so the team can
compare ideas fairly.

``evaluate_design(system)`` samples mover positions over the whole table at the
operating gap and, at each one, works out:

  hover current / power   currents needed just to float (min-power allocation)
  coil temperature        steady state at hover (hover is continuous!)
  lift margin             max upward force / weight, with all torques held at 0
  lateral accel           max sideways acceleration while hovering [g]
  condition number        how evenly the coils can push in all controlled directions

plus a lift-vs-gap curve at the table centre and plain-language verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocation import allocate, condition_number, max_wrench
from .handoff import system_weights
from .materials import mm
from .thermal import max_continuous_current, steady_state, thermal_time_constant


@dataclass
class DesignReport:
    summary: dict
    samples: pd.DataFrame
    lift_curve: pd.DataFrame
    verdicts: list = field(default_factory=list)

    def text(self) -> str:
        s = self.summary
        lines = [
            f"=== Design report: {s['name']} ===",
            s["description_text"],
            "",
            "Key numbers (worst case over sampled positions at the operating gap):",
            f"  mover mass                 {s['mover_mass_g']:.1f} g   (weight {s['weight_N']:.3f} N)",
            f"  hover current, max coil    {s['hover_max_current_A']:.2f} A  (usable limit {s['usable_current_A']:.2f} A,"
            f" continuous thermal limit {s['continuous_current_A']:.2f} A)",
            f"  hover power, total         {s['hover_power_W_worst']:.2f} W  (mean {s['hover_power_W_mean']:.2f} W)",
            f"  hottest coil at hover      {s['hover_coil_temp_C_worst']:.0f} C  (limit {s['max_coil_temp_C']:.0f} C,"
            f" heats up in ~{s['thermal_time_constant_s']:.0f} s)",
            f"  lift margin                {s['lift_margin_worst']:.2f} x weight (mean {s['lift_margin_mean']:.2f})",
            f"  lateral acceleration       {s['lateral_accel_g_worst']:.2f} g while hovering",
            f"  condition number           {s['condition_worst']:.1f} (mean {s['condition_mean']:.1f})",
            f"  positions where hover OK   {100*s['hover_ok_fraction']:.0f} %",
            "",
            "Verdicts:",
        ]
        lines += [f"  {v}" for v in self.verdicts]
        lines += ["", "Lift vs gap at table centre:", self.lift_curve.to_string(index=False, float_format="%.3g")]
        return "\n".join(lines)


def sample_positions(system, per_tile: int | None = None) -> np.ndarray:
    g = system.grid
    n = per_tile or int(system.cfg["analysis"]["xy_samples"])
    ex, ey = g.extent
    margin = min(system.mover.size / 2, min(g.tile_size) / 4)
    xs = np.linspace(margin, ex - margin, max(2, n * g.cols))
    ys = np.linspace(margin, ey - margin, max(2, n * g.rows))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def analyse_pose(system, K, xy) -> dict:
    axes = system.control_axes
    L = system.mover.size
    w = system_weights(system, xy)
    i_max = system.max_current
    hover = system.hover_wrench()
    a = allocate(K, hover, axes, w, i_max, L)
    temps, powers = steady_state(a.currents, system.coil, system.cfg["thermal"])
    fz_max = max_wrench(K, 2, axes, np.zeros(6), i_max, w)
    fx_max = max_wrench(K, 0, axes, hover, i_max, w)
    fy_max = max_wrench(K, 1, axes, hover, i_max, w)
    lat = min(fx_max, fy_max) if 0 in axes and 1 in axes else max(fx_max, fy_max)
    ok = (not a.saturated) and a.error < 0.01 * system.weight
    return {
        "x_mm": xy[0] * 1e3,
        "y_mm": xy[1] * 1e3,
        "tile": system.grid.tile_at(xy),
        "hover_ok": bool(ok),
        "hover_max_current_A": a.max_abs_current,
        "hover_power_W": float(powers.sum()),
        "hover_coil_temp_C": float(temps.max()),
        "lift_margin": fz_max / system.weight if np.isfinite(fz_max) else 0.0,
        "lateral_accel_g": max(lat, 0.0) / system.weight if np.isfinite(lat) else 0.0,
        "condition": condition_number(K, axes, w, L),
        "active_coils": int((w > 1e-9).sum()),
    }


def lift_vs_gap(system, gaps_mm=None, xy=None) -> pd.DataFrame:
    gaps_mm = gaps_mm if gaps_mm is not None else system.cfg["analysis"]["gaps_mm"]
    centre = system.pose() if xy is None else system.pose(x=xy[0], y=xy[1])
    poses = np.array([np.r_[centre[:2], mm(g), 0, 0, 0] for g in gaps_mm])
    Ks = system.wrench_matrices(poses)
    rows = []
    for g, K in zip(gaps_mm, Ks):
        w = system_weights(system, centre[:2])
        fz = max_wrench(K, 2, system.control_axes, np.zeros(6), system.max_current, w)
        a = allocate(K, system.hover_wrench(), system.control_axes, w, system.max_current, system.mover.size)
        rows.append(
            {
                "gap_mm": g,
                "max_lift_N": max(fz, 0.0),
                "lift_margin": max(fz, 0.0) / system.weight,
                "hover_current_A": a.max_abs_current if not a.saturated else np.nan,
                "hover_power_W": float(np.sum(a.currents**2) * system.coil.resistance) if not a.saturated else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_design(system, per_tile: int | None = None, gaps_mm=None, progress: bool = False) -> DesignReport:
    xy = sample_positions(system, per_tile)
    poses = np.column_stack([xy, np.full(len(xy), system.gap), np.zeros((len(xy), 3))])
    Ks = system.wrench_matrices(poses)
    it = range(len(xy))
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="positions")
    df = pd.DataFrame([analyse_pose(system, Ks[k], xy[k]) for k in it])
    curve = lift_vs_gap(system, gaps_mm)
    th = system.cfg["thermal"]
    finite_cond = df["condition"].replace(np.inf, np.nan)
    s = {
        "name": system.cfg["name"],
        "description_text": system.describe(),
        "mover_mass_g": system.mover.mass * 1e3,
        "weight_N": system.weight,
        "coil_resistance_ohm": system.coil.resistance,
        "coil_inductance_mH": system.coil.inductance * 1e3,
        "coil_turns": system.coil.turns,
        "n_coils": system.n_coils,
        "usable_current_A": system.max_current,
        "continuous_current_A": max_continuous_current(system.coil, th),
        "thermal_time_constant_s": thermal_time_constant(system.coil, th),
        "max_coil_temp_C": th["max_coil_temp_C"],
        "hover_ok_fraction": float(df["hover_ok"].mean()),
        "hover_max_current_A": float(df["hover_max_current_A"].max()),
        "hover_power_W_worst": float(df["hover_power_W"].max()),
        "hover_power_W_mean": float(df["hover_power_W"].mean()),
        "hover_coil_temp_C_worst": float(df["hover_coil_temp_C"].max()),
        "lift_margin_worst": float(df["lift_margin"].min()),
        "lift_margin_mean": float(df["lift_margin"].mean()),
        "lateral_accel_g_worst": float(df["lateral_accel_g"].min()),
        "condition_worst": float(df["condition"].max()),
        "condition_mean": float(finite_cond.mean()) if finite_cond.notna().any() else float("inf"),
        "gap_mm": system.gap * 1e3,
    }
    return DesignReport(s, df, curve, verdicts(s))


def verdicts(s: dict) -> list[str]:
    v = []
    ok = s["hover_ok_fraction"]
    if ok >= 0.999:
        v.append("[OK]   Can hover everywhere sampled within the current limit.")
    elif ok > 0:
        v.append(f"[WARN] Can hover at only {100*ok:.0f}% of positions - look at the hover_ok map for weak spots.")
    else:
        v.append("[FAIL] Cannot hover anywhere: not enough force. Try a smaller gap, more/larger magnets, "
                 "more ampere-turns, or a lighter mover.")
    m = s["lift_margin_worst"]
    if m >= 1.5:
        v.append(f"[OK]   Lift margin {m:.1f}x at the weakest spot leaves room for control and payload.")
    elif m >= 1.0:
        v.append(f"[WARN] Lift margin only {m:.2f}x at the weakest spot - little headroom to reject disturbances (aim >= 1.5x).")
    else:
        v.append(f"[FAIL] Lift margin {m:.2f}x < 1 somewhere: the mover would drop there.")
    if s["hover_coil_temp_C_worst"] > s["max_coil_temp_C"]:
        v.append(f"[FAIL] Coils overheat while hovering ({s['hover_coil_temp_C_worst']:.0f} C > "
                 f"{s['max_coil_temp_C']:.0f} C). Add cooling (thermal.h_conv_W_m2K), bigger coils or a lighter mover.")
    elif s["hover_coil_temp_C_worst"] > 0.8 * s["max_coil_temp_C"]:
        v.append(f"[WARN] Coils run hot at hover ({s['hover_coil_temp_C_worst']:.0f} C).")
    else:
        v.append(f"[OK]   Coils stay at or below {s['hover_coil_temp_C_worst']:.0f} C while hovering.")
    c = s["condition_worst"]
    if not np.isfinite(c):
        v.append("[FAIL] Some controlled direction cannot be pushed at some positions (condition number = inf). "
                 "Remove that axis from operating.control_axes or change the mover/layout.")
    elif c > 30:
        v.append(f"[WARN] Condition number up to {c:.0f}: some directions need far more current than others.")
    else:
        v.append(f"[OK]   Condition number <= {c:.0f}: force can be steered in every controlled direction.")
    if s["lateral_accel_g_worst"] < 0.05:
        v.append(f"[WARN] Only {s['lateral_accel_g_worst']:.2f} g sideways while hovering: motion will be slow.")
    return v
