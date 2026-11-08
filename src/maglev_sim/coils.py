"""Coils: the stationary air-core windings inside each tile.

Coil frame: centred on the coil axis, z = 0 at the coil's *top* face, the
winding occupies -height <= z <= 0.

For the force calculation each coil is cut into ``radial x axial`` thin loops
(filaments), each loop into ``segments`` straight pieces. The force on the coil
per amp is  F = sum( dl x B )  over all pieces, with the turns shared equally
between the filaments. By Newton's third law the mover feels -F.

Built-in coil types (``coil.type`` in YAML): round, rect
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import (
    COPPER_DENSITY,
    COPPER_HEAT_CAPACITY,
    ENAMEL_BUILD,
    awg_diameter,
    copper_resistance,
    mm,
)
from .registry import COILS, lookup, register_coil


@dataclass
class CoilModel:
    """One coil design (all coils in the table are identical)."""

    kind: str
    turns: int
    wire_diameter: float  # bare copper [m]
    mean_turn_length: float  # [m]
    height: float  # [m]
    outline: dict  # for plotting: {"shape": "circle", "r_in", "r_out"} or rect sizes
    window_area: float  # winding cross-section area (one side) [m^2]
    surface_area: float  # outside surface for cooling [m^2]
    equiv_radius: float  # for inductance estimate [m]
    radial_depth: float  # winding thickness [m]
    seg_mid: np.ndarray = field(repr=False, default=None)  # (S, 3) local segment midpoints
    seg_dl: np.ndarray = field(repr=False, default=None)  # (S, 3) dl * turns/filaments
    notes: list = field(default_factory=list)

    @property
    def wire_length(self) -> float:
        return self.turns * self.mean_turn_length

    @property
    def resistance(self) -> float:
        return copper_resistance(self.wire_length, self.wire_diameter)

    def resistance_at(self, temp_c: float) -> float:
        return copper_resistance(self.wire_length, self.wire_diameter, temp_c)

    @property
    def copper_mass(self) -> float:
        return self.wire_length * np.pi * self.wire_diameter**2 / 4 * COPPER_DENSITY

    @property
    def heat_capacity(self) -> float:
        return self.copper_mass * COPPER_HEAT_CAPACITY

    @property
    def fill_factor(self) -> float:
        """Copper area / winding window area. Hand winding achieves ~0.5-0.6."""
        return self.turns * np.pi * self.wire_diameter**2 / 4 / self.window_area

    @property
    def inductance(self) -> float:
        """Wheeler's multilayer air-core estimate [H] (good to ~10-20%)."""
        a = self.equiv_radius  # mean radius
        b = self.height
        c = max(self.radial_depth, 1e-6)
        return 31.6e-6 * self.turns**2 * a**2 / (6 * a + 9 * b + 10 * c)

    @property
    def time_constant(self) -> float:
        return self.inductance / self.resistance

    def describe(self) -> str:
        return (
            f"{self.kind} coil: {self.turns} turns of {self.wire_diameter*1e3:.3f} mm wire, "
            f"R = {self.resistance:.2f} ohm, L = {self.inductance*1e3:.2f} mH, "
            f"tau = {self.time_constant*1e3:.2f} ms, fill factor {self.fill_factor:.2f}, "
            f"copper {self.copper_mass*1e3:.0f} g"
        )


def _turns_and_wire(p: dict, window_area: float) -> tuple[int, float, list]:
    notes = []
    if "awg" in p:
        d = awg_diameter(int(p["awg"]))
    else:
        d = mm(p.get("wire_diameter_mm", 0.5))
    fill = float(p.get("fill_factor", 0.55))
    turns = p.get("turns", "auto")
    if turns in (None, "auto"):
        turns = int(fill * window_area / (np.pi * d**2 / 4))
        notes.append(f"turns set automatically from fill factor {fill}: {turns}")
    turns = int(turns)
    ff = turns * np.pi * d**2 / 4 / window_area
    if ff > 0.75:
        notes.append(
            f"WARNING: fill factor {ff:.2f} is not physically windable (>0.75). "
            "Use fewer turns, thinner wire or a bigger coil."
        )
    # Check insulated wire fits at least one layer
    if d * ENAMEL_BUILD > window_area**0.5:
        notes.append("WARNING: wire is thicker than the winding window.")
    return turns, d, notes


def _mesh(p_mesh: dict) -> tuple[int, int, int]:
    return int(p_mesh.get("radial", 3)), int(p_mesh.get("axial", 3)), int(p_mesh.get("segments", 32))


@register_coil("round")
def round_coil(p: dict, mesh: dict) -> CoilModel:
    """Cylindrical (pancake or solenoid-like) air-core coil."""
    ri, ro = mm(p.get("inner_diameter_mm", 8)) / 2, mm(p.get("outer_diameter_mm", 30)) / 2
    h = mm(p.get("height_mm", 12))
    if ro <= ri:
        raise ValueError("coil outer_diameter_mm must be larger than inner_diameter_mm")
    window = (ro - ri) * h
    turns, d, notes = _turns_and_wire(p, window)
    nr, nz, ns = _mesh(mesh)

    # filament radii/heights at the centres of equal sub-areas
    radii = ri + (np.arange(nr) + 0.5) * (ro - ri) / nr
    zs = -h + (np.arange(nz) + 0.5) * h / nz
    phi = np.linspace(0, 2 * np.pi, ns + 1)
    mids, dls = [], []
    share = turns / (nr * nz)
    for r in radii:
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
        mid2 = (pts[1:] + pts[:-1]) / 2
        dl2 = pts[1:] - pts[:-1]
        for z in zs:
            mids.append(np.column_stack([mid2, np.full(ns, z)]))
            dls.append(np.column_stack([dl2 * share, np.zeros(ns)]))
    r_mean = (ri + ro) / 2
    return CoilModel(
        kind="round",
        turns=turns,
        wire_diameter=d,
        mean_turn_length=2 * np.pi * r_mean,
        height=h,
        outline={"shape": "circle", "r_in": ri, "r_out": ro},
        window_area=window,
        surface_area=2 * np.pi * ro * h + 2 * np.pi * (ro**2 - ri**2),
        equiv_radius=r_mean,
        radial_depth=ro - ri,
        seg_mid=np.vstack(mids),
        seg_dl=np.vstack(dls),
        notes=notes,
    )


@register_coil("rect")
def rect_coil(p: dict, mesh: dict) -> CoilModel:
    """Rectangular (or square) air-core coil. Packs tighter on a square grid than round coils."""
    wo, lo = mm(p.get("outer_x_mm", 30)), mm(p.get("outer_y_mm", p.get("outer_x_mm", 30)))
    t = mm(p.get("winding_width_mm", 8))  # thickness of the winding band
    h = mm(p.get("height_mm", 12))
    if 2 * t >= min(wo, lo):
        raise ValueError("winding_width_mm too large: the coil would have no hole")
    window = t * h
    turns, d, notes = _turns_and_wire(p, window)
    nr, nz, ns = _mesh(mesh)
    per_side = max(2, ns // 4)

    offsets = (np.arange(nr) + 0.5) * t / nr  # inward from the outer edge
    zs = -h + (np.arange(nz) + 0.5) * h / nz
    share = turns / (nr * nz)
    mids, dls = [], []
    for off in offsets:
        hx, hy = wo / 2 - off, lo / 2 - off
        corners = np.array([[hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        pts = []
        for k in range(4):
            seg = np.linspace(corners[k], corners[k + 1], per_side + 1)[:-1]
            pts.append(seg)
        pts = np.vstack(pts + [corners[:1]])
        mid2 = (pts[1:] + pts[:-1]) / 2
        dl2 = pts[1:] - pts[:-1]
        for z in zs:
            mids.append(np.column_stack([mid2, np.full(len(mid2), z)]))
            dls.append(np.column_stack([dl2 * share, np.zeros(len(mid2))]))
    wm, lm = wo - t, lo - t
    return CoilModel(
        kind="rect",
        turns=turns,
        wire_diameter=d,
        mean_turn_length=2 * (wm + lm),
        height=h,
        outline={"shape": "rect", "outer": (wo, lo), "inner": (wo - 2 * t, lo - 2 * t)},
        window_area=window,
        surface_area=2 * (wo + lo) * h + 2 * (wo * lo - (wo - 2 * t) * (lo - 2 * t)),
        equiv_radius=np.sqrt(wm * lm / np.pi),
        radial_depth=t,
        seg_mid=np.vstack(mids),
        seg_dl=np.vstack(dls),
        notes=notes,
    )


def build_coil(coil_cfg: dict) -> CoilModel:
    builder = lookup(COILS, "coil", coil_cfg["type"])
    return builder(dict(coil_cfg.get("params", {})), dict(coil_cfg.get("mesh", {})))
