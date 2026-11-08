"""Movers: the permanent-magnet arrays that float above the tiles.

Mover frame (local coordinates):
  * x, y centred on the magnet array,
  * z = 0 at the *bottom* face of the magnets (the face that looks at the coils).

A mover *pose* is a 6-vector ``[x, y, z, roll, pitch, yaw]`` (metres, radians)
where (x, y) is the array centre in table coordinates, z is the gap between the
coil tops (z = 0 in the table frame) and the magnet bottoms, and the rotation is
about the mover's centre of mass.

Built-in mover types (``mover.type`` in YAML):
  disk, disk_pair, disk_grid, cube_grid, halbach_1d, halbach_2d
"""

from __future__ import annotations

from dataclasses import dataclass, field

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

from .materials import NDFEB_DENSITY, grams, mm, remanence
from .registry import MOVERS, lookup, register_mover


@dataclass
class MagnetSpec:
    shape: str  # "cuboid" or "cylinder"
    dimension: tuple  # cuboid: (a, b, c); cylinder: (diameter, height)   [m]
    center: np.ndarray  # local position of the magnet centre [m]
    polarization: np.ndarray  # magnetic polarization vector in local frame [T]

    @property
    def volume(self) -> float:
        if self.shape == "cuboid":
            a, b, c = self.dimension
            return a * b * c
        d, h = self.dimension
        return np.pi * d**2 / 4 * h

    @property
    def half_height(self) -> float:
        return (self.dimension[2] if self.shape == "cuboid" else self.dimension[1]) / 2


@dataclass
class Mover:
    name: str
    magnets: list[MagnetSpec]
    extra_mass: float = 0.0  # kg (frame, payload...)
    info: dict = field(default_factory=dict)

    # --- physical properties -------------------------------------------------
    @property
    def magnet_mass(self) -> float:
        return sum(m.volume for m in self.magnets) * NDFEB_DENSITY

    @property
    def mass(self) -> float:
        return self.magnet_mass + self.extra_mass

    @property
    def height(self) -> float:
        return max(m.center[2] + m.half_height for m in self.magnets)

    @property
    def com(self) -> np.ndarray:
        """Centre of mass in the mover frame (extra mass assumed at the magnets' centroid)."""
        w = np.array([m.volume for m in self.magnets])
        c = np.array([m.center for m in self.magnets])
        return (w[:, None] * c).sum(0) / w.sum()

    @property
    def footprint(self) -> tuple[float, float]:
        """(size_x, size_y) of the bounding box seen from above [m]."""
        xs, ys = [], []
        for m in self.magnets:
            hx, hy = _half_xy(m)
            xs += [m.center[0] - hx, m.center[0] + hx]
            ys += [m.center[1] - hy, m.center[1] + hy]
        return max(xs) - min(xs), max(ys) - min(ys)

    @property
    def size(self) -> float:
        """A characteristic length (largest footprint side); used to scale torques."""
        return max(self.footprint)

    # --- geometry in the table frame ---------------------------------------------
    def com_world(self, pose) -> np.ndarray:
        pose = np.asarray(pose, float)
        return pose[:3] + self.com

    def sources(self, poses) -> list:
        """Magpylib magnet objects following the given pose(s).

        ``poses`` is (6,) or (M, 6). For M poses each magnet gets a path of
        length M, so one ``magpy.getB`` call evaluates all poses at once.
        """
        poses = np.atleast_2d(np.asarray(poses, float))
        if poses.shape[1] == 3:
            poses = np.hstack([poses, np.zeros((len(poses), 3))])
        rot = Rotation.from_euler("xyz", poses[:, 3:6])
        com = self.com
        objs = []
        for m in self.magnets:
            # world centre = pose position + com + R (local centre - com)
            centers = poses[:, :3] + com + rot.apply(m.center - com)
            if m.shape == "cuboid":
                obj = magpy.magnet.Cuboid(dimension=m.dimension, polarization=tuple(m.polarization))
            else:
                obj = magpy.magnet.Cylinder(dimension=m.dimension, polarization=tuple(m.polarization))
            if len(poses) == 1:
                obj.position = centers[0]
                obj.orientation = rot[0]
            else:
                obj.position = centers
                obj.orientation = rot
            objs.append(obj)
        return objs

    def describe(self) -> str:
        fx, fy = self.footprint
        return (
            f"{self.name}: {len(self.magnets)} magnets, footprint {fx*1e3:.0f} x {fy*1e3:.0f} mm, "
            f"height {self.height*1e3:.1f} mm, mass {self.mass*1e3:.1f} g "
            f"(magnets {self.magnet_mass*1e3:.1f} g)"
        )


def _half_xy(m: MagnetSpec) -> tuple[float, float]:
    if m.shape == "cuboid":
        return m.dimension[0] / 2, m.dimension[1] / 2
    return m.dimension[0] / 2, m.dimension[0] / 2


def build_mover(mover_cfg: dict) -> Mover:
    builder = lookup(MOVERS, "mover", mover_cfg["type"])
    mover = builder(dict(mover_cfg.get("params", {})))
    mover.extra_mass = grams(mover_cfg.get("extra_mass_g", 0.0))
    return mover


# --- helpers ----------------------------------------------------------------------


def _bottom_vs_top_field(magnets: list[MagnetSpec], probe_mm: float = 3.0) -> float:
    """Mean |B| just below the array divided by mean |B| just above it."""
    mover = Mover("probe", magnets)
    fx, fy = mover.footprint
    xs = np.linspace(-fx / 2, fx / 2, 9)
    ys = np.linspace(-fy / 2, fy / 2, 9) if fy > 1e-9 else np.array([0.0])
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    src = mover.sources(np.zeros(6))
    below = pts + [0, 0, -mm(probe_mm)]
    above = pts + [0, 0, mover.height + mm(probe_mm)]
    b_below = np.linalg.norm(magpy.getB(src, below, sumup=True), axis=-1).mean()
    b_above = np.linalg.norm(magpy.getB(src, above, sumup=True), axis=-1).mean()
    return b_below / max(b_above, 1e-12)


def _pick_strong_side(build, strong_side: str) -> tuple[list[MagnetSpec], float]:
    """Halbach arrays are one-sided. Build both rotation senses and keep the one
    whose strong side matches ``strong_side`` ('bottom' faces the coils)."""
    a, b = build(+1), build(-1)
    ra, rb = _bottom_vs_top_field(a), _bottom_vs_top_field(b)
    if strong_side == "bottom":
        return (a, ra) if ra >= rb else (b, rb)
    return (a, ra) if ra <= rb else (b, rb)


# --- built-in movers ------------------------------------------------------------------


@register_mover("disk")
def disk(p: dict) -> Mover:
    """One axially magnetised disk. Simplest possible mover (cannot control yaw)."""
    d, h = mm(p.get("diameter_mm", 20)), mm(p.get("height_mm", 10))
    br = remanence(p.get("grade", "N42"))
    m = MagnetSpec("cylinder", (d, h), np.array([0, 0, h / 2]), np.array([0, 0, br]))
    return Mover("disk", [m])


@register_mover("disk_pair")
def disk_pair(p: dict) -> Mover:
    """Two axial disks side by side along x (Berkelman-style). Opposite polarity by default."""
    d, h = mm(p.get("diameter_mm", 20)), mm(p.get("height_mm", 10))
    s = mm(p.get("spacing_mm", 30))  # centre-to-centre
    pol = p.get("polarities", [1, -1])
    br = remanence(p.get("grade", "N42"))
    mags = [
        MagnetSpec("cylinder", (d, h), np.array([x, 0, h / 2]), np.array([0, 0, sgn * br]))
        for x, sgn in zip((-s / 2, s / 2), pol)
    ]
    return Mover("disk_pair", mags)


@register_mover("disk_grid")
def disk_grid(p: dict) -> Mover:
    """nx x ny axial disks on a square pitch, checkerboard polarity."""
    nx, ny = int(p.get("nx", 2)), int(p.get("ny", 2))
    d, h, pitch = mm(p.get("diameter_mm", 15)), mm(p.get("height_mm", 6)), mm(p.get("pitch_mm", 20))
    br = remanence(p.get("grade", "N42"))
    mags = []
    for i in range(nx):
        for j in range(ny):
            c = np.array([(i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, h / 2])
            mags.append(MagnetSpec("cylinder", (d, h), c, np.array([0, 0, br * (-1) ** (i + j)])))
    return Mover("disk_grid", mags)


@register_mover("cube_grid")
def cube_grid(p: dict) -> Mover:
    """nx x ny cuboids, checkerboard polarity, touching or with a gap."""
    nx, ny = int(p.get("nx", 2)), int(p.get("ny", 2))
    a = mm(p.get("size_mm", 10))
    h = mm(p.get("height_mm", p.get("size_mm", 10)))
    pitch = mm(p.get("pitch_mm", p.get("size_mm", 10)))
    br = remanence(p.get("grade", "N42"))
    mags = []
    for i in range(nx):
        for j in range(ny):
            c = np.array([(i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, h / 2])
            mags.append(MagnetSpec("cuboid", (a, a, h), c, np.array([0, 0, br * (-1) ** (i + j)])))
    return Mover("cube_grid", mags)


@register_mover("halbach_1d")
def halbach_1d(p: dict) -> Mover:
    """Linear Halbach array along x: magnetisation turns 90 deg per magnet."""
    n = int(p.get("n", 5))
    a = mm(p.get("size_mm", 10))
    w = mm(p.get("width_mm", p.get("size_mm", 10)))
    h = mm(p.get("height_mm", p.get("size_mm", 10)))
    br = remanence(p.get("grade", "N42"))

    def build(sense):
        mags = []
        for k in range(n):
            ang = sense * k * np.pi / 2
            pol = br * np.array([np.sin(ang), 0, np.cos(ang)])
            pol[np.abs(pol) < 1e-12] = 0.0
            c = np.array([(k - (n - 1) / 2) * a, 0, h / 2])
            mags.append(MagnetSpec("cuboid", (a, w, h), c, pol))
        return mags

    mags, ratio = _pick_strong_side(build, p.get("strong_side", "bottom"))
    return Mover("halbach_1d", mags, info={"bottom_to_top_field_ratio": ratio})


@register_mover("halbach_2d")
def halbach_2d(p: dict) -> Mover:
    """2D Halbach array (Jansen 2008 style) on an n x n grid of equal cubes.

    Even-even cells hold vertical 'main' magnets in a checkerboard, odd/even cells
    hold horizontal magnets pointing toward their neighbouring north-up main
    magnet, odd-odd cells are empty. ``n`` should be odd (3, 5, 7...).
    """
    n = int(p.get("n", 5))
    a = mm(p.get("size_mm", 8))
    h = mm(p.get("height_mm", p.get("size_mm", 8)))
    br = remanence(p.get("grade", "N42"))

    def main_sign(i, j):
        return (-1) ** ((i + j) // 2)

    def build(sense):
        mags = []
        for i in range(n):
            for j in range(n):
                c = np.array([(i - (n - 1) / 2) * a, (j - (n - 1) / 2) * a, h / 2])
                if i % 2 == 0 and j % 2 == 0:
                    pol = [0, 0, br * main_sign(i, j)]
                elif i % 2 == 1 and j % 2 == 0:
                    pol = [sense * br * main_sign(i + 1, j), 0, 0]
                elif i % 2 == 0 and j % 2 == 1:
                    pol = [0, sense * br * main_sign(i, j + 1), 0]
                else:
                    continue
                mags.append(MagnetSpec("cuboid", (a, a, h), c, np.array(pol, float)))
        return mags

    mags, ratio = _pick_strong_side(build, p.get("strong_side", "bottom"))
    return Mover("halbach_2d", mags, info={"bottom_to_top_field_ratio": ratio})
