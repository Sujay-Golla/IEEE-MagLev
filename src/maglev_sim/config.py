"""Load, check and edit design configs (YAML files in ``configs/``).

A config is a plain nested dict. Anything you leave out is filled from
``DEFAULTS`` below, so a config only needs the parts you want to change.

Useful helpers:
    cfg = load_config("configs/baseline.yaml")
    get_path(cfg, "coil.params.turns")          -> 300
    cfg2 = with_changes(cfg, {"coil.params.turns": 400})
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "unnamed_design",
    "description": "",
    "mover": {
        "type": "disk_pair",
        "params": {},
        "extra_mass_g": 0.0,  # frame, payload, glue... on top of the magnets
    },
    "coil": {
        "type": "round",
        "params": {},
        # How finely each coil is cut into thin current loops for the force calc.
        "mesh": {"radial": 3, "axial": 3, "segments": 32},
    },
    "tile": {"layout": "square", "params": {}},
    "grid": {"rows": 1, "cols": 1},
    "electrical": {
        "max_current_A": 3.0,  # per coil, driver/wire limit
        "supply_voltage_V": 24.0,  # per-coil driver supply; limits current to V/R
    },
    "thermal": {
        "ambient_C": 25.0,
        "h_conv_W_m2K": 12.0,  # natural air convection ~5-15, small fan ~25-50
        "max_coil_temp_C": 100.0,
    },
    "operating": {
        "gap_mm": 5.0,  # coil top surface -> bottom of mover magnets
        # Which force/torque components the controller actively commands.
        # fx, fy, fz = forces; tx, ty, tz = torques (roll, pitch, yaw).
        "control_axes": ["fx", "fy", "fz", "tx", "ty"],
    },
    "analysis": {
        "xy_samples": 7,  # sample points per tile per direction for the design report
        "cutoff_mm": 90.0,  # ignore coils farther than this from the mover centre
        "gaps_mm": [2, 4, 6, 8, 10, 12, 15, 20],  # for the lift-vs-gap curve
    },
    "handoff": {
        # Coil weights fade smoothly from 1 (within inner) to 0 (beyond outer),
        # measured from the mover centre (Jansen 2008 style commutation).
        "fade_inner_mm": 40.0,
        "fade_outer_mm": 70.0,
        "hysteresis_mm": 5.0,  # ownership switches only this far past a tile edge
        "link_delay_steps": 1,  # control steps before a neighbour applies currents it was sent
    },
    "control": {
        "rate_Hz": 1000.0,
        "bandwidth_Hz": [8.0, 8.0, 12.0],  # x, y, z closed-loop natural frequency
        "damping": 0.8,
        "velocity_filter_Hz": 150.0,
    },
    "tolerances": {
        # Assumed build tolerances (+/- limits). Missing ones use the defaults in
        # maglev_sim/tolerance.py (TOLERANCES). Example: {coil_position_mm: 0.5}
        "values": {},
        # Acceptable impact (p95). Missing ones use DEFAULT_THRESHOLDS.
        "thresholds": {},
        "trials": 20,
    },
    "simulation": {
        "duration_s": 1.0,
        "sensor_noise_mm": 0.02,
        "driver_time_constant_ms": 0.2,  # current-loop speed of the coil driver
        "substeps": 10,  # physics steps per control step
        "seed": 0,
    },
}

_TOP_LEVEL = set(DEFAULTS)


def deep_merge(base: Mapping, override: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) and k != "params":
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_unknown(user: Mapping, ref: Mapping, where: str = "") -> list[str]:
    """Warn about likely typos: keys not present in DEFAULTS (params are free-form)."""
    problems = []
    for k, v in user.items():
        path = f"{where}.{k}" if where else k
        if k == "params":
            continue
        if k not in ref:
            close = difflib.get_close_matches(k, ref.keys(), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            problems.append(f"Unknown config key '{path}'{hint}")
        elif isinstance(v, Mapping) and isinstance(ref[k], Mapping) and ref[k]:  # {} = free-form
            problems.extend(_check_unknown(v, ref[k], path))
    return problems


def validate(user_cfg: Mapping, strict: bool = True) -> list[str]:
    problems = _check_unknown(user_cfg, DEFAULTS)
    if problems and strict:
        raise ValueError("Config problems:\n  " + "\n  ".join(problems))
    return problems


def make_config(user_cfg: Mapping | None = None, strict: bool = True) -> dict:
    user_cfg = dict(user_cfg or {})
    validate(user_cfg, strict=strict)
    return deep_merge(DEFAULTS, user_cfg)


def load_config(path: str | Path, strict: bool = True) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = make_config(user, strict=strict)
    cfg.setdefault("_source", str(path))
    return cfg


def save_config(cfg: Mapping, path: str | Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


def get_path(cfg: Mapping, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"'{dotted}' not found in config (stopped at '{part}')")
        node = node[part]
    return node


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def with_changes(cfg: Mapping, changes: Mapping[str, Any]) -> dict:
    """Copy of cfg with dotted-path changes applied, e.g. {"tile.params.pitch_mm": 30}."""
    new = copy.deepcopy(dict(cfg))
    for k, v in changes.items():
        set_path(new, k, v)
    return new


def config_hash(cfg: Mapping, sections=("mover", "coil", "tile", "grid", "analysis")) -> str:
    subset = {k: cfg.get(k) for k in sections}
    blob = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
