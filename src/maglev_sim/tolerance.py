"""Tolerances: how accurately must we build it, and what does each error cost?

The real table will never match the drawing. Magnets vary in strength and get
glued slightly off, coils are hand-wound with a few turns more or less, tiles
are misaligned, the mover tilts, the position sensor has an offset...

The controller only knows the *nominal* design. So with an error present it
sends the currents that *should* make a clean hover force, but the real
(perturbed) table turns them into a slightly wrong wrench. We measure:

  force_error_pct      |vertical force error| / weight              -> gap disturbance
  lateral_accel_g      sideways force error / weight                -> drift the XY loop must fight
  torque_error_pct     tilt torque error / (weight * mover size)    -> tilt disturbance
  current_increase_pct extra peak current a perfectly re-calibrated controller would need
  lift_margin_loss_pct loss of maximum lift capacity

Each tolerance is a "plus/minus" limit: errors are drawn uniformly in
[-value, +value], independently per magnet / coil / tile, over many random builds
(Monte Carlo). We report the 95th percentile (p95) over builds and positions.

Main entry points:
  sensitivity(system)         one tolerance at a time at its assumed value -> what hurts most
  required_tolerances(system) largest value of each tolerance that keeps every
                              impact under ``tolerances.thresholds``
  monte_carlo(system)         all tolerances together -> distribution / "yield" of builds
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd

from .allocation import allocate, max_wrench
from .handoff import system_weights

# name -> (plain description, unit, default assumed tolerance)
TOLERANCES = {
    "magnet_br_pct": ("Magnet strength spread (Br), per magnet", "%", 3.0),
    "magnet_position_mm": ("Magnet placement error in the mover, per magnet", "mm", 0.2),
    "magnet_angle_deg": ("Magnetisation direction error, per magnet", "deg", 2.0),
    "coil_position_mm": ("Coil placement error in X/Y, per coil", "mm", 0.3),
    "coil_height_mm": ("Coil height error (top face not flush), per coil", "mm", 0.3),
    "coil_turns_pct": ("Turn-count error, per coil", "%", 2.0),
    "tile_offset_mm": ("Whole-tile misalignment, per tile", "mm", 0.5),
    "gap_mm": ("Mover height error vs. assumed gap", "mm", 0.5),
    "mover_tilt_deg": ("Mover tilt (roll/pitch) the tilt control allows", "deg", 1.0),
    "sensor_offset_mm": ("Position-sensor bias in X/Y", "mm", 0.3),
}

METRICS = {
    "force_error_pct": "vertical force error, % of weight",
    "lateral_accel_g": "sideways force error, in g",
    "torque_error_pct": "tilt torque error, % of weight x mover size",
    "current_increase_pct": "extra peak hover current needed, %",
    "lift_margin_loss_pct": "loss of max lift, %",
}

# Starting points only - the team should agree its own values (docs/tolerances.md).
DEFAULT_THRESHOLDS = {
    "force_error_pct": 10.0,
    "lateral_accel_g": 0.1,
    "torque_error_pct": 10.0,
    "current_increase_pct": 15.0,
    "lift_margin_loss_pct": 15.0,
}


def tolerance_settings(system) -> tuple[dict, dict, int]:
    """Assumed tolerances, thresholds and trial count from config (with defaults)."""
    tcfg = system.cfg.get("tolerances", {}) or {}
    values = {k: v[2] for k, v in TOLERANCES.items()}
    values.update(tcfg.get("values", {}) or {})
    unknown = set(values) - set(TOLERANCES)
    if unknown:
        raise KeyError(f"Unknown tolerance(s) {sorted(unknown)}. Known: {sorted(TOLERANCES)}")
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(tcfg.get("thresholds", {}) or {})
    return values, thresholds, int(tcfg.get("trials", 12))


# ---------------------------------------------------------------------------------
# Building one random "as-built" table
# ---------------------------------------------------------------------------------


def _u(rng, limit, size=None):
    return rng.uniform(-limit, limit, size) if limit else np.zeros(size if size is not None else ())


def _rotate_small(vec, angle, rng):
    """Rotate vec by `angle` [rad] about a random axis perpendicular to it."""
    if angle == 0:
        return vec
    from scipy.spatial.transform import Rotation

    n = np.linalg.norm(vec)
    r = rng.normal(size=3)
    axis = r - r.dot(vec) / n**2 * vec
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).apply(vec)


def as_built(system, tol: dict, rng):
    """Return (plant_system, pose_error). ``pose_error`` is added to the pose the
    controller believes in to get the true pose (6-vector)."""
    t = {k: float(tol.get(k, 0.0) or 0.0) for k in TOLERANCES}
    # --- mover ---
    mags = []
    for m in system.mover.magnets:
        m2 = copy.deepcopy(m)
        m2.center = m.center + np.r_[_u(rng, t["magnet_position_mm"] * 1e-3, 2), 0.0]
        pol = m.polarization * (1 + _u(rng, t["magnet_br_pct"] / 100))
        m2.polarization = _rotate_small(pol, _u(rng, np.radians(t["magnet_angle_deg"])), rng)
        mags.append(m2)
    mover = dataclasses.replace(system.mover, magnets=mags)
    # --- coils & tiles ---
    C = system.n_coils
    shift = np.zeros((C, 3))
    shift[:, :2] = _u(rng, t["coil_position_mm"] * 1e-3, (C, 2))
    shift[:, 2] = _u(rng, t["coil_height_mm"] * 1e-3, C)
    tile_shift = _u(rng, t["tile_offset_mm"] * 1e-3, (system.grid.n_tiles, 2))
    shift[:, :2] += tile_shift[system.grid.coil_tile]
    turns = 1 + _u(rng, t["coil_turns_pct"] / 100, C)
    grid = dataclasses.replace(system.grid, coil_xy=system.grid.coil_xy + shift[:, :2])
    plant = dataclasses.replace(
        system,
        mover=mover,
        grid=grid,
        seg_mid=system.seg_mid + shift[:, None, :],
        seg_dl=system.seg_dl * turns[:, None, None],
    )
    # --- pose errors (gap, tilt, sensor bias) ---
    pose_err = np.zeros(6)
    pose_err[:2] = _u(rng, t["sensor_offset_mm"] * 1e-3, 2)
    pose_err[2] = _u(rng, t["gap_mm"] * 1e-3)
    pose_err[3:5] = _u(rng, np.radians(t["mover_tilt_deg"]), 2)
    return plant, pose_err


# ---------------------------------------------------------------------------------
# Measuring the impact
# ---------------------------------------------------------------------------------


def check_positions(system) -> np.ndarray:
    """A few representative spots: table centre, between coils, a tile edge if any."""
    g = system.grid
    pitch = min(g.tile_size) / np.sqrt(len(g.layout.coil_xy))
    c = system.pose()[:2]
    pts = [c, c + [pitch / 2, 0], c + [pitch / 2, pitch / 2]]
    if g.cols > 1:
        pts.append(np.array([g.tile_size[0], c[1]]))  # first vertical tile edge
    return np.array(pts)


class _Nominal:
    """Pre-computed nominal currents and capacities at the check positions."""

    def __init__(self, system, positions):
        self.system = system
        self.positions = positions
        self.poses = np.column_stack([positions, np.full(len(positions), system.gap), np.zeros((len(positions), 3))])
        self.K = system.wrench_matrices(self.poses)
        self.w = [system_weights(system, p) for p in positions]
        axes, L, imax = system.control_axes, system.mover.size, system.max_current
        hover = system.hover_wrench()
        self.alloc = [allocate(K, hover, axes, w, imax, L) for K, w in zip(self.K, self.w)]
        self.lift = [max_wrench(K, 2, axes, np.zeros(6), imax, w) for K, w in zip(self.K, self.w)]


def _impacts(nom: _Nominal, plant, pose_err) -> list[dict]:
    s = nom.system
    axes, L, imax, W = s.control_axes, s.mover.size, s.max_current, s.weight
    hover = s.hover_wrench()
    true_poses = nom.poses + pose_err
    Kt = plant.wrench_matrices(true_poses)
    # Reference for the "permanent cost" metrics: the PERFECT table at the same true pose,
    # so pure pose errors (sensor offset, gap, tilt) are not blamed for normal position variation.
    Kref = s.wrench_matrices(true_poses) if np.any(pose_err) else nom.K
    out = []
    for k in range(len(nom.poses)):
        I = nom.alloc[k].currents
        err = Kt[k] @ I - nom.alloc[k].achieved
        re = allocate(Kt[k], hover, axes, nom.w[k], None, L)  # true need of the as-built table
        if np.any(pose_err):
            i0 = allocate(Kref[k], hover, axes, nom.w[k], None, L).max_abs_current
            lift0 = max_wrench(Kref[k], 2, axes, np.zeros(6), imax, nom.w[k])
        else:
            i0, lift0 = nom.alloc[k].max_abs_current, nom.lift[k]
        i0 = max(i0, 1e-9)
        lift_t = max_wrench(Kt[k], 2, axes, np.zeros(6), imax, nom.w[k])
        out.append(
            {
                "force_error_pct": abs(err[2]) / W * 100,
                "lateral_accel_g": float(np.hypot(err[0], err[1]) / W),
                "torque_error_pct": float(np.hypot(err[3], err[4]) / (W * L) * 100),
                "current_increase_pct": max(0.0, (re.max_abs_current - i0) / i0 * 100),
                "lift_margin_loss_pct": max(0.0, (lift0 - lift_t) / lift0 * 100)
                if np.isfinite(lift0) and lift0 > 0
                else 0.0,
            }
        )
    return out


def run_trials(system, tol: dict, trials: int, seed: int = 0, nominal: _Nominal | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nom = nominal or _Nominal(system, check_positions(system))
    rows = []
    for t in range(trials):
        plant, perr = as_built(system, tol, rng)
        for k, r in enumerate(_impacts(nom, plant, perr)):
            rows.append({"trial": t, "position": k, **r})
    return pd.DataFrame(rows)


def _p95(df: pd.DataFrame) -> dict:
    return {m: float(np.percentile(df[m], 95)) for m in METRICS}


def sensitivity(system, trials: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Each tolerance alone at its assumed value. Sorted by worst impact relative to thresholds."""
    values, thresholds, n = tolerance_settings(system)
    trials = trials or n
    nom = _Nominal(system, check_positions(system))
    rows = []
    for name, val in values.items():
        p = _p95(run_trials(system, {name: val}, trials, seed, nom))
        worst = max(p[m] / thresholds[m] for m in METRICS)
        rows.append(
            {
                "tolerance": name,
                "description": TOLERANCES[name][0],
                "assumed": f"+/-{val:g} {TOLERANCES[name][1]}",
                **p,
                "worst_vs_threshold": worst,
            }
        )
    return pd.DataFrame(rows).sort_values("worst_vs_threshold", ascending=False).reset_index(drop=True)


def required_tolerances(
    system, factors=(0.25, 0.5, 1, 2, 4), trials: int | None = None, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each tolerance, scale it up/down and find the largest value that keeps
    every impact metric (p95) below its threshold.

    Returns (summary, curves): summary has one row per tolerance with
    ``max_allowed`` and whether the assumed value passes; curves holds every
    point for plotting impact vs tolerance.
    """
    values, thresholds, n = tolerance_settings(system)
    trials = trials or n
    nom = _Nominal(system, check_positions(system))
    curve_rows, summary = [], []
    for name, val in values.items():
        base = val if val > 0 else 1.0
        mags = np.array([0.0] + [base * f for f in factors])
        ratios = []
        for mag in mags:
            p = _p95(run_trials(system, {name: mag}, trials, seed, nom)) if mag > 0 else {m: 0.0 for m in METRICS}
            ratio = max(p[m] / thresholds[m] for m in METRICS)
            ratios.append(ratio)
            curve_rows.append({"tolerance": name, "value": mag, **p, "worst_vs_threshold": ratio})
        ratios = np.array(ratios)
        # largest magnitude before the worst ratio crosses 1 (linear interpolation)
        allowed = mags[-1]
        limiting = ""
        over = np.flatnonzero(ratios > 1.0)
        if over.size:
            j = over[0]
            r0, r1 = ratios[j - 1], ratios[j]
            allowed = mags[j - 1] + (1.0 - r0) / max(r1 - r0, 1e-12) * (mags[j] - mags[j - 1])
            last = curve_rows[-len(mags) + j]
            limiting = max(METRICS, key=lambda m: last[m] / thresholds[m])
        unit = TOLERANCES[name][1]
        summary.append(
            {
                "tolerance": name,
                "description": TOLERANCES[name][0],
                "unit": unit,
                "assumed": val,
                "max_allowed": float(allowed),
                "max_allowed_is_lower_bound": not over.size,
                "limited_by": limiting or "(none in tested range)",
                "status": "OK" if val <= allowed else "TOO LOOSE",
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(curve_rows)


def monte_carlo(system, trials: int = 200, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """All tolerances at once. Returns per-build worst-position impacts and the
    'yield': fraction of random builds where every metric stays under threshold."""
    values, thresholds, _ = tolerance_settings(system)
    df = run_trials(system, values, trials, seed)
    worst = df.groupby("trial")[list(METRICS)].max()
    passed = np.all([worst[m] <= thresholds[m] for m in METRICS], axis=0)
    return worst.reset_index(), float(np.mean(passed))


def sensitivity_from_curves(system, curves: pd.DataFrame) -> pd.DataFrame:
    """The 'impact at the assumed value' table, reusing required_tolerances() results."""
    values, _, _ = tolerance_settings(system)
    rows = []
    for name, val in values.items():
        c = curves[(curves.tolerance == name) & np.isclose(curves.value, val if val > 0 else 1.0)]
        r = c.iloc[0].to_dict()
        rows.append({"tolerance": name, "description": TOLERANCES[name][0],
                     "assumed": f"+/-{val:g} {TOLERANCES[name][1]}",
                     **{m: r[m] for m in METRICS}, "worst_vs_threshold": r["worst_vs_threshold"]})
    return pd.DataFrame(rows).sort_values("worst_vs_threshold", ascending=False).reset_index(drop=True)


@dataclasses.dataclass
class ToleranceReport:
    sensitivity: pd.DataFrame
    required: pd.DataFrame
    curves: pd.DataFrame
    builds: pd.DataFrame
    yield_fraction: float
    thresholds: dict

    def text(self) -> str:
        o = dict(index=False, float_format="%.3g")
        return "\n".join([
            "=== Tolerance report ===",
            "Thresholds (p95 over random builds): " + ", ".join(f"{k} <= {v:g}" for k, v in self.thresholds.items()),
            "",
            "Impact of each tolerance alone, at its assumed value (worst first;",
            "worst_vs_threshold > 1 means it breaks a threshold on its own):",
            self.sensitivity.drop(columns=["description"]).to_string(**o),
            "",
            "Required tolerances (largest value keeping every impact under its threshold):",
            self.required.drop(columns=["description"]).to_string(**o),
            "",
            f"All assumed tolerances together: {100*self.yield_fraction:.0f}% of random builds stay within every threshold.",
            "",
            "Errors here are disturbances the feedback controller must reject. Calibrating the",
            "real table (measuring K) removes most static errors. See docs/tolerances.md.",
        ])


def tolerance_report(system, trials: int | None = None, mc_trials: int = 100) -> ToleranceReport:
    _, thresholds, _ = tolerance_settings(system)
    req, curves = required_tolerances(system, trials=trials)
    sens = sensitivity_from_curves(system, curves)
    builds, yld = monte_carlo(system, trials=mc_trials)
    return ToleranceReport(sens, req, curves, builds, yld, thresholds)
