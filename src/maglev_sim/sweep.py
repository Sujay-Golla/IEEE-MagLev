"""Parameter sweeps: try many variations of a design and compare them in one table.

    from maglev_sim.sweep import sweep
    df = sweep(cfg, {"coil.params.turns": [200, 300, 400],
                     "operating.gap_mm": [3, 5]})

Every combination is evaluated with the standard design report (metrics.py);
the result has one row per combination with the varied values and headline
numbers. Command line equivalent:

    python scripts/run_sweep.py configs/baseline.yaml --vary coil.params.awg=24,26,28 --vary operating.gap_mm=2:6:1
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import yaml

from .config import with_changes
from .metrics import evaluate_design
from .system import build_system

HEADLINE = [
    "hover_ok_fraction",
    "lift_margin_worst",
    "lift_margin_mean",
    "hover_max_current_A",
    "hover_power_W_worst",
    "hover_power_W_mean",
    "hover_coil_temp_C_worst",
    "lateral_accel_g_worst",
    "condition_worst",
    "mover_mass_g",
    "coil_turns",
    "coil_resistance_ohm",
    "continuous_current_A",
]


def parse_values(text: str) -> list:
    """'1,2,3' -> [1,2,3];  '2:6:1' -> [2,3,4,5,6];  'square,offset' -> strings."""
    text = text.strip()
    if ":" in text and "," not in text:
        a, b, c = (float(x) for x in text.split(":"))
        vals = np.arange(a, b + c / 2, c)
        return [int(v) if float(v).is_integer() else round(float(v), 9) for v in vals]
    return [yaml.safe_load(v) for v in text.split(",")]


def parse_vary(specs: list[str]) -> dict:
    out = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--vary needs path=values, got '{s}'")
        k, v = s.split("=", 1)
        out[k.strip()] = parse_values(v)
    return out


def sweep(cfg: dict, grid: dict, per_tile: int = 4, gaps_mm=None, progress: bool = True) -> pd.DataFrame:
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    it = combos
    if progress:
        from tqdm import tqdm

        it = tqdm(combos, desc="designs")
    rows = []
    for combo in it:
        changes = dict(zip(keys, combo))
        row = dict(changes)
        try:
            system = build_system(with_changes(cfg, changes))
            rep = evaluate_design(system, per_tile=per_tile, gaps_mm=gaps_mm or [system.gap * 1e3])
            row.update({k: rep.summary[k] for k in HEADLINE})
            row["warnings"] = " | ".join(system.coil.notes + system.check_geometry())
        except Exception as e:  # keep going: one bad combination should not kill the sweep
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)
