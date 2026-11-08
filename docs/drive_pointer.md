# Text for a "Simulation" doc in the 2D MagLev Tile Drive folder

(Paste into a Google Doc; replace the repo link once it's on GitHub.)

---

**Simulation toolkit: maglev-sim**

Repo: <GitHub link>

What it is: a shared Python toolkit for the simulation phase. You describe a design (magnets, coils,
tile layout, gap) in a small text file and it tells you whether the mover floats, how much
current and heat that takes, how evenly it can be steered, whether tile handoff is smooth, and
**which build tolerances matter and how tight each must be**.

Getting started (~10 min): follow the README quickstart, then open `notebooks/01_tour.ipynb`.

How we use it:
1. Copy `configs/baseline.yaml`, change one idea, run `python scripts/run_design.py configs/<yours>.yaml`.
2. Paste the key numbers + config name into the Design Log below; commit the config to the repo.
3. Sweep promising ideas with `scripts/run_sweep.py`; check tolerances with `--tolerances`.

Early findings (baseline design, simulated):
- Hovers everywhere with ~2.5x lift margin at a 3 mm gap (61 g mover, 25 mm coil pitch).
- Coil heat is the main risk; we need a bench test of one coil's heating early.
- Tile-to-tile alignment needs about ±0.15 mm and coil placement about ±0.2 mm. Magnet quality and turn count hardly matter.
- The tile-to-tile link can be up to ~2 ms late without hurting tracking. At 10 ms the error grows to 1.4 mm.
- Handoff between tiles is seamless with smooth coil weighting.

Design Log: | date | who | config | idea | lift margin | hover W | verdict |
