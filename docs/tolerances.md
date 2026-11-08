# Tolerances: what they mean and how to use the results

## The idea

The controller is programmed with the *nominal* design (the drawing). The real table differs:
magnets vary in strength and are glued slightly off, coils are hand-wound with a few turns more or
less and mounted a bit high or low, tiles are misaligned, the mover tilts, the position sensor has
an offset. With an error present, the controller sends currents that *should* produce a clean
hover force, but the real table turns them into a slightly wrong one.

`tolerance.py` builds many random "as-built" tables (each error drawn uniformly within ±tolerance,
independently per magnet / coil / tile) and measures, at representative positions (table centre,
between coils, a tile edge):

| Metric | What it is | Why it matters |
|---|---|---|
| `force_error_pct` | unexpected vertical force, % of weight | gap disturbance the z-loop must reject |
| `lateral_accel_g` | unexpected sideways force ÷ weight | drift the x/y loops must fight |
| `torque_error_pct` | unexpected tilt torque ÷ (weight × mover size) | tilt disturbance |
| `current_increase_pct` | extra peak current the as-built table needs vs a perfect table at the same position | heat, driver sizing |
| `lift_margin_loss_pct` | lost maximum lift | less payload / disturbance headroom |

All are reported as **p95**: 95 % of random builds are at least this good.

## The three outputs

1. **Sensitivity**: each tolerance alone at its assumed value, ranked by `worst_vs_threshold`
   (its worst metric divided by the threshold). Above 1 means that error alone breaks a requirement.
   This tells you **where precision effort pays off**.
2. **Required tolerances**: each tolerance is scaled (×0.25 … ×4) and we find the largest value
   that keeps every metric under its threshold (`max_allowed`), and which metric sets it
   (`limited_by`). This is the number for drawings and assembly instructions.
   `max_allowed_is_lower_bound = True` means the threshold was never reached in the tested range.
3. **Monte Carlo yield**: all assumed tolerances at once; the fraction of random builds that pass
   every threshold.

## Setting it up

In your config:

```yaml
tolerances:
  values:            # what we think we can achieve (+/-)
    coil_position_mm: 0.3
    tile_offset_mm: 0.5
    ...
  thresholds:        # what we can accept (p95)
    force_error_pct: 10
    lateral_accel_g: 0.1
    ...
  trials: 12         # random builds per setting (more = smoother, slower)
```

**The thresholds are a team decision.** The defaults are starting points: a static force error is
easily rejected by feedback, so 10 % of weight is not dangerous by itself; what matters is that the
sum of all disturbances stays well inside the lift margin and lateral capability shown in the design
report. Revisit them once the control loop and sensors are better known.

## How to read it sensibly

* **Static errors can be calibrated out.** Once a table is built, measuring K (or a correction map)
  removes most of the force error. Tolerances that only hurt `force_error_pct` / `lateral_accel_g`
  are therefore candidates for "calibrate instead of machining precisely". Errors that cost
  `current_increase_pct` or `lift_margin_loss_pct` cannot be calibrated away; they cost heat and
  headroom permanently.
* **`gap_mm`, `mover_tilt_deg` and `sensor_offset_mm` are control-accuracy requirements**, not
  machining tolerances: they say how well the gap/tilt/position must be held or measured.
* **Check it in closed loop.** Notebook 06 runs the full simulation with a random as-built table as
  the real "plant" while the controller still uses the nominal model, to confirm the mover still
  levitates and tracks.

## Example: baseline design (run of 2026-11-08, 12 builds per setting)

| Tolerance | Assumed | Max allowed | Limited by | Status |
|---|---|---|---|---|
| Whole-tile misalignment | ±0.5 mm | **±0.16 mm** | extra current | too loose |
| Coil placement (x/y) | ±0.3 mm | **±0.20 mm** | extra current | too loose |
| Magnet placement in mover | ±0.2 mm | ±0.19 mm | extra current | borderline |
| Gap held by control | ±0.5 mm | ±0.35 mm | vertical force error | too loose |
| Coil height | ±0.3 mm | ±0.55 mm | extra current | OK |
| Mover tilt held by control | ±1° | ±1.2° | sideways force | OK |
| Sensor offset | ±0.3 mm | ±0.59 mm | sideways force | OK |
| Magnetisation angle | ±2° | ±3.5° | extra current | OK |
| Magnet strength (Br) | ±3 % | ±6 % | extra current | OK |
| Coil turns | ±2 % | > ±8 % | none | OK |

With every assumed tolerance at once, only 16 % of random builds meet every threshold.

What this says for the build:
* **Tile-to-tile alignment is the most important tolerance.** Dowel pins or a common baseplate
  to about ±0.15 mm. A misaligned tile makes the coils on both sides of a seam fight each other.
* **Coil placement within a tile needs ±0.2 mm.** That favours a PCB or 3D-printed jig with
  pockets over hand placement.
* **Magnet quality and winding accuracy hardly matter.** Off-the-shelf N42 magnets and
  hand-counted turns are fine.
* **The gap must be held to about ±0.35 mm**, which is a requirement on the z-sensor and z-loop.

Rerun `python scripts/run_design.py configs/baseline.yaml --tolerances` after any design change.
The numbers depend on the design and on the thresholds.
