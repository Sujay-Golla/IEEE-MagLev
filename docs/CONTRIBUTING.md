# Extending maglev-sim

The toolkit is built so that **adding an idea never means editing the core**. Most additions are
one function plus a decorator.

## Working together

* Make a git branch per experiment (`git switch -c alex/rect-coils`), commit configs and code,
  open a pull request so someone else can review.
* Put designs in `configs/`; keep the name in the file unique. Don't commit `results/` (it's
  ignored); paste the key numbers and the config name into the team doc instead.
* Run `python -m pytest` before merging. Add a test for anything you add.
* Notebooks are generated from `tools/make_notebooks.py`. Edit that file and regenerate, so diffs
  stay readable. Your own scratch notebooks can live anywhere outside `notebooks/`.

## Ownership by subteam (suggested)

| Subteam | Main files |
|---|---|
| Magnetics & mechanics | `magnets.py`, `coils.py`, `layouts.py`, `thermal.py`, `tolerance.py` (build errors) |
| Electronics & sensing | `thermal.py`, coil driver model in `dynamics.py`, a future `sensing.py` |
| Control & tile coordination | `control.py`, `allocation.py`, `handoff.py`, `tiles.py`, `dynamics.py` |

## Add a new mover (magnet array)

```python
# src/maglev_sim/magnets.py (or your own module imported in __init__.py)
@register_mover("ring")
def ring(p: dict) -> Mover:
    """Ring of n axial disks with alternating polarity."""
    n, R = int(p.get("n", 6)), mm(p.get("radius_mm", 20))
    d, h = mm(p.get("diameter_mm", 10)), mm(p.get("height_mm", 5))
    br = remanence(p.get("grade", "N42"))
    mags = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = np.array([R * np.cos(a), R * np.sin(a), h / 2])     # bottom face at z = 0
        mags.append(MagnetSpec("cylinder", (d, h), c, np.array([0, 0, br * (-1) ** k])))
    return Mover("ring", mags)
```

Then in YAML: `mover: {type: ring, params: {n: 8, radius_mm: 25}}`.

Rules: lengths in the params are mm (convert with `mm()`); the magnets' bottom faces sit at z = 0;
centre the array on x = y = 0; `polarization` is a vector in tesla (use `remanence(grade)`).

## Add a new coil shape

Return a `CoilModel` (see `round_coil` in `coils.py`): you provide turns, wire diameter, mean turn
length, winding window area, cooling surface area, and the current path as segment midpoints
`seg_mid` and vectors `seg_dl` (scaled by turns per filament). The coil's top face is z = 0.
A PCB spiral coil is a natural next addition.

## Add a new layout

```python
@register_layout("triangle")
def triangle(p: dict) -> TileLayout:
    ...
    return TileLayout(coil_xy=np.array([...]), size=(width, height), name="triangle")
```

Coil positions are measured from the tile's lower-left corner. Keep spacing across the tile edges
equal to spacing inside the tile.

## Add a new controller

Any object with `wrench(p, v, p_ref, v_ref, a_ref) -> 6-vector` works:

```python
import numpy as np
from maglev_sim import PDController, simulate

class PIDController(PDController):
    """PD + integral action, to remove steady errors (e.g. from build tolerances)."""
    def __init__(self, system, ki=200.0):          # ki in 1/s^3
        super().__init__(system)
        self.ki = ki
        self.dt = 1.0 / system.cfg["control"]["rate_Hz"]
        self.integral = np.zeros(3)

    def wrench(self, p, v, p_ref, v_ref, a_ref):
        self.integral += (p_ref - p) * self.dt
        w = super().wrench(p, v, p_ref, v_ref, a_ref)
        w[:3] += self.m * self.ki * self.integral
        return w

res = simulate(system, traj, controller=PIDController(system))
```

## Add a new tile strategy

Subclass `TileNetwork` (`tiles.py`) and override `update_owner` and/or `apply`. For example "the
owner broadcasts the wanted force; every tile allocates only its own coils" is a good experiment
to compare against the default "owner computes everything".

## Add a new tolerance

Add an entry to `TOLERANCES` in `tolerance.py` and apply it inside `as_built()` (perturb the mover,
coil arrays or the pose error). It then appears in every tolerance report automatically.

## Natural next modules

* `sensing.py`: Hall-sensor array model (field of the mover at sensor positions + coil crosstalk
  + noise) and a position estimator, replacing the ideal noisy sensor in `dynamics.py`.
* Tilt dynamics (roll/pitch/yaw) in `dynamics.py` using the torque rows already computed.
* Iron-backed coils (needs a FEM cross-check, e.g. FEMM).
* Routing (Krühn 2010 distance field) on top of `tiles.py` for multi-tile paths.
