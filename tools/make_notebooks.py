"""Generate the tutorial notebooks in notebooks/ from this one file.

Edit the text/code here, then run:  python tools/make_notebooks.py
(Keeping the source in Python makes notebook changes easy to review in git.)
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from maglev_sim import *
from maglev_sim import plotting"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        src = src.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return book


NOTEBOOKS = {}

NOTEBOOKS["01_tour.ipynb"] = [
    ("md", """
# 01 - A tour of the simulator (start here)

**Goal:** understand what the toolkit models, in plain words, in ~15 minutes.

A design has three parts, each set in a YAML file in `configs/`:

| Part | What it is | Config section |
|---|---|---|
| **Mover** | the floating puck: an array of permanent magnets | `mover` |
| **Coils** | flat air-core windings under the surface; current in them pushes on the magnets | `coil` |
| **Tiles** | identical boards of coils, placed edge to edge | `tile`, `grid` |

The one physics idea you need: **force on the mover = K x (coil currents)**. For the mover sitting at
one spot, `K` is a table (6 rows: Fx, Fy, Fz and the three tilting/turning torques; one column
per coil) saying how hard each coil pushes per amp. Double a current -> double its push.
Everything else (how much current to hover, heat, control, tile handoff) is built on `K`.
"""),
    ("code", SETUP),
    ("code", """cfg = load_config("../configs/baseline.yaml")
system = build_system(cfg)
print(system.describe())"""),
    ("md", "Top view. Orange = coils that are 'active' for the mover where it is now (darker = more involved). Red magnets point north-up, blue south-up."),
    ("code", "plotting.plot_layout(system);"),
    ("md", """
## The K table at one position
Columns are coils. Only coils near the mover have non-zero entries - coil influence is local
(lit review takeaway), which is what makes separate tiles possible.
"""),
    ("code", """pose = system.pose()            # table centre at the operating gap
K = system.wrench_matrix(pose)
near = np.flatnonzero(np.abs(K).sum(0) > 0)
import pandas as pd
pd.DataFrame(K[:, near], index=AXES, columns=[f"coil {c}" for c in near]).round(4)"""),
    ("md", """
## How much current to just float?
`allocate` picks the lowest-power set of currents that makes exactly the force we want
(here: lift = weight, everything else zero).
"""),
    ("code", """from maglev_sim.allocation import allocate
from maglev_sim.handoff import system_weights
a = allocate(K, system.hover_wrench(), system.control_axes, system_weights(system, pose[:2]), system.max_current)
print(f"weight {system.weight:.3f} N -> produced Fz {a.achieved[2]:.3f} N")
print(f"largest coil current {a.max_abs_current:.2f} A, total power {np.sum(a.currents**2)*system.coil.resistance:.2f} W")"""),
    ("md", """
## The standard design report
One call gives the numbers the team compares designs by. It samples many mover positions,
because a design that works at the centre can fail between coils.
"""),
    ("code", """rep = evaluate_design(system, per_tile=5)
print(rep.text())"""),
    ("code", "plotting.plot_report(rep);"),
    ("md", """
**Reading it:** *lift margin* > 1.5 everywhere is comfortable; *hover power* is heat you must get rid
of continuously; *condition number* says how evenly the coils can push in every direction (big =
some direction is expensive). Next: notebook 02 to change a coil yourself.
"""),
]

NOTEBOOKS["02_try_a_coil.ipynb"] = [
    ("md", """
# 02 - Try a coil

**Goal:** change the coil and see what happens to lift, current and heat.

Rules of thumb you will see in the numbers:
* For a **fixed coil size**, more turns of thinner wire gives the same force for the same
  *power* - it only changes the voltage/current trade-off. Heat depends on copper volume.
* A **smaller gap** is the cheapest way to more force.
* Coil heat is usually the real limit (lit review: papers 1, 2, 4).
"""),
    ("code", SETUP),
    ("code", """base = load_config("../configs/examples/single_tile_quick.yaml")
print(build_system(base).describe())"""),
    ("md", "Change any value below and re-run the cell. `with_changes` makes a modified copy; the original stays untouched."),
    ("code", """mine = with_changes(base, {
    "coil.params.outer_diameter_mm": 24,
    "coil.params.height_mm": 12,
    "coil.params.awg": 26,
    "operating.gap_mm": 3,
})
s = build_system(mine)
print(s.describe())
rep = evaluate_design(s, per_tile=4, gaps_mm=[2, 3, 4, 6, 8])
print(rep.text())"""),
    ("md", "## Wire gauge: same coil, different wire"),
    ("code", """from maglev_sim.sweep import sweep
df = sweep(base, {"coil.params.awg": [22, 24, 26, 28]}, per_tile=3)
df[["coil.params.awg", "coil_turns", "coil_resistance_ohm", "hover_max_current_A", "hover_power_W_mean", "lift_margin_worst"]].round(3)"""),
    ("md", "Notice hover **power** barely changes with gauge while current and resistance trade off. Now coil height:"),
    ("code", """df = sweep(base, {"coil.params.height_mm": [6, 9, 12, 16, 20]}, per_tile=3)
plotting.plot_sweep(df, "coil.params.height_mm", ["lift_margin_worst", "hover_power_W_mean", "hover_coil_temp_C_worst"]);"""),
    ("md", """
**Try next:** `coil.type: rect` (see `configs/examples/rect_coils_fan.yaml`), a bigger
`thermal.h_conv_W_m2K` (a fan), or a different `operating.gap_mm`. Save designs you like as new
YAML files in `configs/` so others can rerun them.
"""),
]

NOTEBOOKS["03_compare_layouts.ipynb"] = [
    ("md", """
# 03 - Compare layouts and movers

**Goal:** choose coil layout + magnet array together (Kuijpers 2023: pick the magnet pitch first,
then size the coils to it). Uses `sweep` to evaluate every combination.
"""),
    ("code", SETUP),
    ("code", """base = load_config("../configs/examples/single_tile_quick.yaml")
from maglev_sim.sweep import sweep
df = sweep(base, {
    "tile.params.pitch_mm": [20, 25, 30],
    "mover.params.size_mm": [6, 8, 10],
}, per_tile=3)
cols = ["tile.params.pitch_mm", "mover.params.size_mm", "mover_mass_g", "hover_ok_fraction",
        "lift_margin_worst", "hover_power_W_mean", "condition_worst"]
df[cols].sort_values("lift_margin_worst", ascending=False).round(2)"""),
    ("md", "Coils must fit their pitch: the sweep reports warnings like overlapping coils."),
    ("code", """df[["tile.params.pitch_mm", "mover.params.size_mm", "warnings"]]"""),
    ("md", "## Different mover types on the same tile"),
    ("code", """movers = {
    "cube checkerboard 4x4": {"type": "cube_grid", "params": {"nx": 4, "ny": 4, "size_mm": 8, "pitch_mm": 16}},
    "disk grid 3x3":        {"type": "disk_grid", "params": {"nx": 3, "ny": 3, "diameter_mm": 12, "height_mm": 6, "pitch_mm": 18}},
    "2D Halbach n=5":       {"type": "halbach_2d", "params": {"n": 5, "size_mm": 8}},
}
rows = []
for name, m in movers.items():
    cfg = with_changes(base, {"mover": m})
    r = evaluate_design(build_system(cfg), per_tile=3, gaps_mm=[3])
    rows.append({"mover": name, **{k: r.summary[k] for k in
                 ["mover_mass_g", "hover_ok_fraction", "lift_margin_worst", "hover_power_W_mean", "condition_worst"]}})
import pandas as pd
pd.DataFrame(rows).round(2)"""),
    ("code", """fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, m) in zip(axs, movers.items()):
    plotting.plot_layout(build_system(with_changes(base, {"mover": m})), ax=ax)
    ax.set_title(name)"""),
    ("md", """
**Open question for the team:** the 2D Halbach from the proposal does worse here than a simple
checkerboard. Is that the magnet size, the pitch match, or the coil shape? This notebook is the
place to find out - try `halbach_2d` with different `size_mm` and coil pitches.
"""),
]

NOTEBOOKS["04_levitate_one_tile.ipynb"] = [
    ("md", """
# 04 - Levitate on one tile (closed loop)

**Goal:** see the control loop work: sensor -> PD controller -> current allocation -> coils -> mover.

The controller is set by a *bandwidth* in Hz (`control.bandwidth_Hz`): higher = stiffer, but
needs more current and a faster, cleaner sensor. v1 simulates x, y, z motion; tilt torques are
commanded to zero and logged.
"""),
    ("code", SETUP),
    ("code", """cfg = load_config("../configs/examples/single_tile_quick.yaml")
s = build_system(cfg)
p0 = np.r_[s.grid.tile_center(0), s.gap]
res = simulate(s, Trajectory.hold(p0), duration=0.5)   # first run builds the fast force model (~10 s)
print(res.summary())
plotting.plot_simulation(res);"""),
    ("md", "## A small move, then a disturbance: drop the gap target by 0.5 mm"),
    ("code", """p1 = p0 + [0.015, 0.010, 0.0]
traj = Trajectory([(0, p0), (0.1, p0), (0.5, p1), (0.7, p1), (0.8, p1 - [0, 0, 0.0005])])
res = simulate(s, traj, duration=1.0)
print(res.summary())
plotting.plot_simulation(res);"""),
    ("md", "## Effect of controller bandwidth and sensor noise"),
    ("code", """rows = []
for bw in [4, 8, 16, 30]:
    for noise in [0.005, 0.02, 0.05]:
        c = with_changes(cfg, {"control.bandwidth_Hz": [bw, bw, bw * 1.5], "simulation.sensor_noise_mm": noise})
        r = simulate(build_system(c), Trajectory.move(p0, p1, 0.1, 0.4), duration=0.7)
        rows.append({"bandwidth_Hz": bw, "noise_mm": noise, **{k: r.info[k] for k in
                     ["rms_error_mm", "peak_current_A", "mean_power_W", "touchdown"]}})
import pandas as pd
pd.DataFrame(rows).round(3)"""),
    ("md", """
**What to look for:** noisy sensors + high bandwidth = jittery currents (heat, EMI into Hall
sensors). This tells the electronics subteam what sensor noise is acceptable.
"""),
]

NOTEBOOKS["05_handoff_two_tiles.ipynb"] = [
    ("md", """
# 05 - Handoff between tiles

**Goal:** move the mover across a tile edge with no central controller.

* Coils fade in/out smoothly by distance (`handoff.fade_inner_mm` / `fade_outer_mm`, Jansen 2008).
* The tile under the mover *owns* it and computes currents for nearby coils, including the
  neighbour's; ownership moves once the mover is `hysteresis_mm` past the edge.
* Neighbour tiles apply the currents they are sent `link_delay_steps` control steps late.
"""),
    ("code", SETUP),
    ("code", """cfg = with_changes(load_config("../configs/baseline.yaml"), {"grid.rows": 1, "grid.cols": 2})
s = build_system(cfg)
plotting.plot_layout(s, pose=s.pose(x=0.100, y=0.050));"""),
    ("md", "## 1. Quasi-static check: currents along a line across the edge (x = 100 mm)"),
    ("code", """from maglev_sim.handoff import straight_line_sweep
df = straight_line_sweep(s, [0.05, 0.05], [0.15, 0.05], n=81)
plotting.plot_handoff(df, s);"""),
    ("md", """
The pattern repeats every coil pitch and nothing special happens at the tile edge - the
"seamless" goal. The sharp swings are the coils commutating under the magnets, made worse by
this layout's weak tilt (torque) authority - a design point worth improving.
"""),
    ("md", "## 2. Full closed loop across the edge"),
    ("code", """p0, p1 = np.r_[0.05, 0.05, s.gap], np.r_[0.15, 0.05, s.gap]
res = simulate(s, Trajectory.move(p0, p1, 0.2, 0.8), duration=1.3)
print(res.summary())
print("handoff log (step, kind, from tile, to tile):", res.info["handoff_log"])
plotting.plot_simulation(res);"""),
    ("md", "## 3. How much link latency can we tolerate?"),
    ("code", """rows = []
for delay in [0, 1, 2, 5, 10]:
    c = with_changes(cfg, {"handoff.link_delay_steps": delay})
    r = simulate(build_system(c), Trajectory.move(p0, p1, 0.2, 0.8), duration=1.3)
    rows.append({"delay_steps": delay, "delay_ms": delay * 1000 / c["control"]["rate_Hz"],
                 **{k: r.info[k] for k in ["rms_error_mm", "max_error_mm", "peak_current_A", "touchdown"]}})
import pandas as pd
pd.DataFrame(rows).round(3)"""),
    ("md", """
This directly answers the lit-review question "can the tile link carry handoff data at the
control rate?" - read off the largest delay that keeps tracking acceptable, then pick UART/CAN/...
accordingly. To try a different ownership/communication strategy, subclass `TileNetwork` in
`src/maglev_sim/tiles.py`.
"""),
]

NOTEBOOKS["06_tolerances.ipynb"] = [
    ("md", """
# 06 - Tolerances: how precisely must we build it?

**Goal:** find which build errors matter, how much they hurt, and the tolerance each needs.

The controller knows the *drawing*; the real table is slightly off (magnet strength spread,
misplaced coils, tile misalignment, tilt, sensor offset...). We build many random "as-built"
tables and measure what each error costs (see `docs/tolerances.md`):

| Impact metric | Meaning |
|---|---|
| `force_error_pct` | unexpected vertical force, % of weight |
| `lateral_accel_g` | unexpected sideways push, in g |
| `torque_error_pct` | unexpected tilting torque |
| `current_increase_pct` | extra current needed once calibrated |
| `lift_margin_loss_pct` | lost lift capacity |

Assumed tolerances and the acceptable thresholds live in the config under `tolerances:` -
**agree the thresholds as a team**; they are starting points.
"""),
    ("code", SETUP),
    ("code", """from maglev_sim.tolerance import tolerance_report, run_trials, as_built, TOLERANCES
s = build_system(load_config("../configs/baseline.yaml"))
tr = tolerance_report(s, trials=8)   # ~1-2 minutes; raise trials for final numbers
print(tr.text())"""),
    ("code", "plotting.plot_tolerance(tr);"),
    ("md", """
**Reading it:** left plot ranks the tolerances (red bars break a threshold on their own at the
assumed value). The table gives `max_allowed` for each - that is the number to put on the
drawing / assembly instructions. `limited_by` says which effect sets it.

## Does it still levitate? Simulate a random as-built table
The controller still uses the nominal design; the 'plant' is a random build.
"""),
    ("code", """rng = np.random.default_rng(1)
values = {k: v for k, v in s.cfg["tolerances"]["values"].items()}
plant, pose_err = as_built(s, values, rng)
p0 = np.r_[s.grid.tile_center(0), s.gap]
p1 = np.r_[s.grid.tile_center(1), s.gap]
nominal = simulate(s, Trajectory.move(p0, p1, 0.1, 0.8), duration=1.1)
built = simulate(s, Trajectory.move(p0, p1, 0.1, 0.8), duration=1.1, plant=plant,
                 plant_pose_error=np.r_[0, 0, pose_err[2], 0, 0, 0], sensor_bias=np.r_[pose_err[:2], 0])
print("nominal :", nominal.summary())
print("as-built:", built.summary())"""),
    ("md", """
## Tighten or loosen one tolerance
E.g. what if tiles are aligned with dowel pins to +/-0.1 mm instead of +/-0.5 mm?
"""),
    ("code", """for tile_tol in [0.1, 0.25, 0.5, 1.0]:
    df = run_trials(s, {"tile_offset_mm": tile_tol}, trials=8)
    print(f"tile offset +/-{tile_tol} mm -> p95 force error {np.percentile(df.force_error_pct, 95):.1f} %, "
          f"lateral {np.percentile(df.lateral_accel_g, 95):.3f} g, extra current {np.percentile(df.current_increase_pct, 95):.1f} %")"""),
]


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nbf.write(nb(cells), OUT / name)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
