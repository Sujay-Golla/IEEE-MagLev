"""Command-line entry points (also wrapped by the files in scripts/).

    maglev-design   configs/baseline.yaml [--tolerances]
    maglev-sweep    configs/baseline.yaml --vary coil.params.turns=200:500:100
    maglev-sim      configs/baseline.yaml --to 140,50 [--duration 1.5]

Each run writes to results/<name>_<timestamp>/ : a copy of the config, the
numbers (JSON/CSV), a text report and figures. Keep the folder with your notes so
anyone can see exactly which design produced which result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .config import load_config, save_config  # noqa: E402
from .system import build_system  # noqa: E402


def _outdir(cfg, kind: str, root="results") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(root) / f"{cfg['name']}_{kind}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    save_config(cfg, d / "config.yaml")
    return d


def _write(d: Path, name: str, text: str):
    (d / name).write_text(text, encoding="utf-8")


def design_main(argv=None):
    ap = argparse.ArgumentParser(description="Standard design report for one config.")
    ap.add_argument("config")
    ap.add_argument("--samples", type=int, default=None, help="positions per tile per direction")
    ap.add_argument("--tolerances", action="store_true", help="also run the tolerance analysis (a few minutes)")
    ap.add_argument("--trials", type=int, default=None, help="random builds per tolerance setting")
    a = ap.parse_args(argv)

    from .metrics import evaluate_design
    from .plotting import plot_layout, plot_report

    cfg = load_config(a.config)
    system = build_system(cfg)
    print(system.describe(), flush=True)
    rep = evaluate_design(system, per_tile=a.samples, progress=True)
    d = _outdir(cfg, "design")
    text = rep.text()
    print(text)
    _write(d, "report.txt", text)
    rep.samples.to_csv(d / "samples.csv", index=False)
    rep.lift_curve.to_csv(d / "lift_vs_gap.csv", index=False)
    (d / "summary.json").write_text(json.dumps(rep.summary, indent=2, default=float), encoding="utf-8")
    plot_layout(system).savefig(d / "layout.png", dpi=130, bbox_inches="tight")
    plot_report(rep).savefig(d / "report.png", dpi=130, bbox_inches="tight")
    if a.tolerances:
        from .plotting import plot_tolerance
        from .tolerance import tolerance_report

        print("\nRunning tolerance analysis...", flush=True)
        tr = tolerance_report(system, trials=a.trials)
        print(tr.text())
        _write(d, "tolerances.txt", tr.text())
        tr.required.to_csv(d / "required_tolerances.csv", index=False)
        tr.curves.to_csv(d / "tolerance_curves.csv", index=False)
        plot_tolerance(tr).savefig(d / "tolerances.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved to {d}")
    return 0


def sweep_main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate every combination of the varied parameters.")
    ap.add_argument("config")
    ap.add_argument("--vary", action="append", required=True,
                    help="dotted.path=values, e.g. coil.params.turns=200:500:100 or tile.layout=square,offset")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args(argv)

    from .plotting import plot_sweep
    from .sweep import parse_vary, sweep

    cfg = load_config(a.config)
    grid = parse_vary(a.vary)
    df = sweep(cfg, grid, per_tile=a.samples)
    d = _outdir(cfg, "sweep")
    df.to_csv(d / "sweep.csv", index=False)
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.drop(columns=[c for c in ("warnings",) if c in df]).to_string(index=False, float_format="%.3g"))
    if len(grid) == 1 and "error" not in df:
        x = next(iter(grid))
        if all(isinstance(v, (int, float)) for v in df[x]):
            plot_sweep(df, x).savefig(d / "sweep.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved to {d}")
    return 0


def simulate_main(argv=None):
    ap = argparse.ArgumentParser(description="Closed-loop simulation: hover, or move to a point.")
    ap.add_argument("config")
    ap.add_argument("--start", default=None, help="x,y in mm (default: centre of tile 0)")
    ap.add_argument("--to", default=None, help="x,y in mm to move to (default: hover in place)")
    ap.add_argument("--move-time", type=float, default=0.8, help="seconds for the move")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--central", action="store_true", help="one central controller instead of tiles")
    a = ap.parse_args(argv)

    import numpy as np

    from .control import Trajectory
    from .plotting import plot_simulation

    cfg = load_config(a.config)
    system = build_system(cfg)
    gap = system.gap
    start = (np.array([float(v) for v in a.start.split(",")]) * 1e-3 if a.start
             else system.grid.tile_center(0))
    p0 = np.r_[start, gap]
    if a.to:
        p1 = np.r_[np.array([float(v) for v in a.to.split(",")]) * 1e-3, gap]
        traj = Trajectory.move(p0, p1, 0.2, a.move_time)
        duration = a.duration or (0.2 + a.move_time + 0.5)
    else:
        traj = Trajectory.hold(p0)
        duration = a.duration
    from .dynamics import simulate

    res = simulate(system, traj, duration=duration, decentralized=not a.central)
    print(res.summary())
    d = _outdir(cfg, "sim")
    _write(d, "summary.txt", res.summary())
    plot_simulation(res).savefig(d / "simulation.png", dpi=130, bbox_inches="tight")
    print(f"Saved to {d}")
    return 0


if __name__ == "__main__":
    sys.exit(design_main())
