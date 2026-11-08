# Models, assumptions and limits

What each part of the simulator assumes, so nobody over-trusts a number.

| Part | Model | Assumptions / limits | How to extend |
|---|---|---|---|
| Magnets | Analytical fields (Magpylib) of uniformly magnetised cuboids/cylinders | Ideal, rigid magnetisation (NdFeB is close); grade → typical Br | Add a builder; per-magnet spread is in `tolerance.py` |
| Coils | Air-core, turns spread evenly over `radial × axial` thin loops | No iron, no PCB copper traces yet; inductance by Wheeler's formula (±10–20 %) | New coil type (e.g. PCB spiral) |
| Force | Lorentz force on the coil, mover feels the reaction; torque about the mover's centre of mass | Coils beyond `analysis.cutoff_mm` ignored (check by raising it); no eddy currents | — |
| Fast model | Mover field on a 1 mm × 0.5 mm grid, trilinear interpolation | ~0.5 % error; valid within ~(gap − 3 mm, gap + 5 mm) | Finer grid in `FastForceModel` |
| Allocation | Least-power currents with per-coil handoff weights, slight damping near weak directions | Damping costs < 0.5 % force error | Other objectives (min peak current, thermal-aware) |
| Thermal | Each coil alone, steady state, convection only (`h` × outer area), R rises with temperature | Ignores conduction into the PCB/plate and neighbour heating; temperatures are capped at 400 °C ("would burn") | Calibrate `h` from a bench test; add a plate model |
| Dynamics | Mover translates in x, y, z; ideal current-mode drivers limited by L and supply voltage | Tilt torques computed, commanded to 0 and logged, but don't rotate the mover | Add rotational dynamics using the torque rows |
| Sensing | True position + Gaussian noise + optional bias, filtered velocity | No Hall sensor physics or coil crosstalk yet | `sensing.py` |
| Tiles | Owner (with hysteresis) computes all nearby currents; neighbours apply them after a fixed delay | No packet loss, no clock skew | Subclass `TileNetwork` |
| Tolerances | Uniform ±errors, independent per part; impacts at 3–4 representative positions; p95 | Correlated errors (e.g. one batch of magnets) not modelled | Add to `TOLERANCES` / `as_built` |

## Findings worth knowing early (baseline design)

* **Heat is the main risk.** In still air a 24 mm coil can carry only ~0.6 A continuously; the
  baseline needs ~1–2 A at some positions. Small gap, lighter mover, a fan, or a heat-spreading
  plate are the levers. Measure a real coil early (see `validation.md`).
* **Hover power depends strongly on where the mover sits** relative to the coils (5 W vs 90 W
  for the baseline), because of how the magnet pitch lines up with the coil pitch. Matching the
  two is a key design choice.
* **Tilt authority is weak** in the baseline (the tilt directions are ~40× harder to push than
  lift), which makes currents swing as the mover passes over coils. Improving this (magnet
  pattern, coil pitch, overlapping layers) is a good early project.
* **Tile handoff itself is seamless** with smooth coil weighting: crossing a tile edge is
  indistinguishable from crossing a coil boundary inside a tile.
* **Precision goes into alignment, not parts**: tiles to about ±0.15 mm and coils to about ±0.2 mm.
  Magnet strength spread and turn count barely matter (see `tolerances.md`).
* **Sensor noise drives heat**: a stiffer controller with a noisier position sensor raised mean
  coil power from ~6 W to ~48 W in notebook 04, at similar tracking accuracy.
* **Link latency**: up to ~2 ms of tile-to-tile delay barely changes tracking. At 10 ms the peak
  error grows to 1.4 mm (notebook 05).
