# maglev-sim

The simulation backbone for the **2D MagLev Tile** project: a modular table of identical coil
tiles that levitates a passive magnet puck and moves it from tile to tile with no central controller.

It lets anyone on the team try a coil or magnet idea by **editing a small YAML file** and get back
the numbers that decide whether it can work. No electromagnetics background needed.

| Question | Where it's answered |
|---|---|
| Will it float? How much current, how much heat? | `scripts/run_design.py` → design report |
| Which coil / magnet / layout is best? | `scripts/run_sweep.py`, notebooks 02–03 |
| Does closed-loop control work, and with what sensor noise? | `scripts/run_simulation.py`, notebook 04 |
| Is handing the mover between tiles seamless? What link delay is OK? | notebook 05 |
| **How precisely must we build it? Which errors matter most?** | `run_design.py --tolerances`, notebook 06 |

## 5-minute quickstart

You need Python 3.10+, git, and about 1 GB of free RAM (close browser tabs if a run fails with
`MemoryError`).

```bash
git clone <repo-url> maglev-sim
cd maglev-sim
python -m venv .venv
```

Activate the environment. On Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the package and its dev tools:

```bash
pip install -e ".[dev]"
```

Then run the baseline design report:

```bash
python scripts/run_design.py configs/baseline.yaml
```

Results (text report, CSVs, figures and a copy of the config) go to `results/<name>_design_<time>/`.

For a guided start, open `notebooks/01_tour.ipynb` in VS Code or Jupyter (`jupyter lab`).

## How to try your own idea

1. Copy `configs/baseline.yaml` to `configs/<you>_<idea>.yaml` and change the `name:` field.
2. Change what you want: magnet size, coil turns, coil pitch, gap... Units are in the key names
   (`pitch_mm`, `extra_mass_g`, `max_current_A`). Typos are caught with a "did you mean" hint.
3. Run `python scripts/run_design.py configs/<you>_<idea>.yaml` and read the **Verdicts**.
4. To vary something systematically:
   ```bash
   python scripts/run_sweep.py configs/baseline.yaml --vary coil.params.awg=24,26,28 --vary operating.gap_mm=2:5:1
   ```
5. Share the YAML file and the results folder; anyone can reproduce your numbers exactly.

## The commands

| Command | What it does | Typical time |
|---|---|---|
| `python scripts/run_design.py CFG` | design report over the whole table + lift-vs-gap | 5–20 s |
| `python scripts/run_design.py CFG --tolerances` | + tolerance sensitivity, required tolerances, Monte Carlo yield | +2–5 min |
| `python scripts/run_sweep.py CFG --vary path=values` | report for every combination | ~5 s per design |
| `python scripts/run_simulation.py CFG --start 50,50 --to 150,50` | closed-loop move across tiles | ~15 s per simulated second |
| `python -m pytest` | physics and workflow checks | ~1 min |

## What's inside

```
configs/            baseline.yaml (shared starting point) + examples/ (literature-inspired variants)
src/maglev_sim/
  config.py         YAML loading, defaults, typo checking
  magnets.py        mover builders: disk, disk_pair, disk_grid, cube_grid, halbach_1d, halbach_2d
  coils.py          coil builders: round, rect (turns, resistance, inductance, fill factor)
  layouts.py        coil layouts per tile (square, offset, custom) and the tile grid
  system.py         builds a design; computes K (force/torque per amp per coil) exactly
  forcemap.py       fast K for simulations (mover field sampled on a grid)
  allocation.py     wanted force -> least-power coil currents; condition number; max force
  thermal.py        coil heating
  metrics.py        the standard design report and verdicts
  tolerance.py      as-built errors -> impact, required tolerances, Monte Carlo yield
  handoff.py        smooth coil fade-in/out between tiles
  tiles.py          per-tile ownership and delayed neighbour links (decentralised control)
  control.py        PD controller and smooth trajectories
  dynamics.py       closed-loop time simulation (digital-twin starting point)
  sweep.py          parameter sweeps
  plotting.py       all figures
notebooks/          01 tour → 06 tolerances (generated from tools/make_notebooks.py)
docs/               CONTRIBUTING (adding your own parts), validation, tolerances, models & limits
tests/              pytest checks, including against Magpylib's independent force solver
```

**Physics in one paragraph.** Magnets are modelled analytically with [Magpylib](https://magpylib.readthedocs.io)
(no meshing, no licence). Each coil is split into thin current loops; the force on a coil is the
Lorentz force `I dl × B` summed over its wire, and the mover feels the opposite (Newton's third law,
checked against Magpylib's own force solver in the tests). Because force is linear in current, the
whole design reduces to a matrix `K(pose)`: wrench = `K · I` (the "added-up force model" of
Berkelman 2013). Currents are chosen by least-power allocation with smooth per-coil weights for
tile handoff (Jansen 2008).

## Glossary

| Term | Meaning |
|---|---|
| **mover / puck** | the floating part, an array of permanent magnets |
| **gap** | distance from the top of the coils to the bottom of the magnets |
| **wrench** | the 6 numbers acting on the mover: forces Fx, Fy, Fz and torques Tx, Ty (tilt), Tz (yaw) |
| **K matrix** | wrench produced per amp in each coil, at one mover position |
| **allocation** | choosing coil currents that produce a wanted wrench |
| **lift margin** | max upward force ÷ weight; < 1 means it drops, aim ≥ 1.5 |
| **condition number** | how unevenly the coils can push in different directions; ∞ = some direction impossible |
| **commutation / handoff** | switching which coils act as the mover travels, smoothly |
| **owner tile** | the tile in charge of the mover right now |
| **Halbach array** | magnet arrangement that concentrates field on one side |
| **fill factor** | fraction of the coil window that is copper (hand winding ≈ 0.5–0.6) |
| **p95** | 95th percentile: 95 % of random builds are at least this good |

## Known limits (v1)

Air-core coils only (no iron); the mover translates in x, y, z (tilt torques are computed, commanded
to zero and logged, but don't rotate it yet); the thermal model is per coil, convection only (pessimistic
without a heatsink); no eddy currents or wireless power. Each has a clear place to be added: see
`docs/models_and_limits.md`.
