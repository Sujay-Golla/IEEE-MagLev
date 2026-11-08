# Validation: how we know the numbers are sensible

Simulation results are only as good as their checks. Here is what has been checked so far and
what still needs checking against hardware.

## Automated checks (`python -m pytest`)

| Check | Result |
|---|---|
| Force and torque on the mover from one coil vs Magpylib's **independent** `getFT` solver (force on the magnet, meshed) | agree within ~0.1 % (test tolerance 1–2 %) |
| Force is linear in current (wrench = K·I) | exact |
| Disk centred over a coil: no sideways force or torque (symmetry) | exact |
| Force falls with gap | yes |
| Batched and single-pose K identical | exact |
| Fast field-grid model vs exact model (random poses incl. yaw) | < 2 % (typically 0.4 %) |
| Halbach builders put the strong side towards the coils | bottom/top field ratio > 1.5 |
| Allocation hits the wanted wrench, respects the current limit, consistent with max-lift LP | yes |
| Handoff: currents repeat every coil pitch, so crossing a tile edge looks the same as crossing a coil boundary inside a tile | exact |
| Closed loop: hover + small move without touchdown, final error < 0.2 mm | yes |
| Every config in `configs/` loads and builds | yes |

## Literature ballpark: Berkelman & Dzadovsky 2013 (paper 1)

`configs/examples/berkelman_like.yaml` approximates their setup from the numbers in our lit review:
250 g platform, two disk magnets, 16 coils at 35 mm pitch, ~9 Ω coils, 3 A limit, ~25 mm
levitation range. **Coil and magnet dimensions are our guesses**, so only the order of magnitude
is comparable.

Our model of that setup gives (1×1 tile, all 16 coils active, 5-axis control, 3 A limit, 36 V supply),
at the table centre:

| Gap | Max lift / weight | Peak hover current | Hover power |
|---|---|---|---|
| 5 mm | 3.5× | 1.1 A | 93 W |
| 10 mm | 3.0× | 1.5 A | 119 W |
| 15 mm | 2.3× | 1.9 A | 181 W |
| 20 mm | 1.7× | 2.5 A | 285 W |
| 25 mm | 1.25× | 3.0 A (at the limit) | 462 W |

Coils come out at ~10.8 Ω. This matches the paper's picture: a 250 g platform can be levitated with
~9 Ω coils over a ~25 mm height range, reaching the 3 A limit near the top of that range, and
heating is the limit (hundreds of watts).
Rerun with `python scripts/run_design.py configs/examples/berkelman_like.yaml`.

## Still to validate with hardware (suggested first bench tests)

1. **One coil + one magnet on a kitchen scale**: measure the force vs current at 3–4 gaps and
   compare with `lift_vs_gap` for a `disk` mover over a 1×1 layout with your real coil dimensions.
   Cheapest, most valuable check.
2. **Coil heating**: run one coil at 0.5/1/1.5 A, log temperature with a thermocouple, and fit
   `thermal.h_conv_W_m2K` (our default 12 W/m²K, natural convection, is probably pessimistic once
   coils sit on a PCB or aluminium plate).
3. **Coil resistance and inductance** with a multimeter / LCR meter vs the `describe()` numbers.
4. **Hall sensor reading** of the mover field at known offsets (for the future `sensing.py`).

Record results in `docs/validation.md` with date, setup photo and the config used.
