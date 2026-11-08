"""The System: one complete design (mover + coils + tiles) built from a config,
plus the core physics call: the *wrench matrix*.

Wrench = the 6 numbers (Fx, Fy, Fz, Tx, Ty, Tz) acting on the mover, with
torques taken about the mover's centre of mass.

Key fact (Berkelman 2013): with the mover held at one pose, the wrench is
linear in the coil currents:

        wrench = K(pose) @ I          K is 6 x n_coils, in N/A and N*m/A

``System.wrench_matrix(pose)`` returns K. Everything else in the toolkit
(current allocation, lift capacity, control, handoff) is built on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import magpylib as magpy
import numpy as np

from .coils import CoilModel, build_coil
from .config import make_config
from .layouts import Grid, build_grid, build_layout
from .magnets import Mover, build_mover
from .materials import G, mm

AXES = ["fx", "fy", "fz", "tx", "ty", "tz"]


@dataclass
class System:
    cfg: dict
    mover: Mover
    coil: CoilModel
    grid: Grid
    seg_mid: np.ndarray  # (C, S, 3) world positions of every coil's segments
    seg_dl: np.ndarray  # (C, S, 3)

    # ---------- convenient numbers ----------
    @property
    def n_coils(self) -> int:
        return self.grid.n_coils

    @property
    def weight(self) -> float:
        return self.mover.mass * G

    @property
    def gap(self) -> float:
        return mm(self.cfg["operating"]["gap_mm"])

    @property
    def cutoff(self) -> float:
        return mm(self.cfg["analysis"]["cutoff_mm"])

    @property
    def max_current(self) -> float:
        """Usable current per coil: the smaller of the set limit and supply voltage / R."""
        e = self.cfg["electrical"]
        return float(min(e["max_current_A"], e["supply_voltage_V"] / self.coil.resistance))

    @property
    def control_axes(self) -> list[int]:
        return [AXES.index(a) for a in self.cfg["operating"]["control_axes"]]

    def pose(self, x=None, y=None, z=None, yaw=0.0) -> np.ndarray:
        """Build a pose; defaults to the centre of the table at the operating gap."""
        ex, ey = self.grid.extent
        return np.array(
            [ex / 2 if x is None else x, ey / 2 if y is None else y, self.gap if z is None else z, 0, 0, yaw],
            float,
        )

    def hover_wrench(self) -> np.ndarray:
        """The wrench needed to just hold the mover up: Fz = m g, everything else 0."""
        w = np.zeros(6)
        w[2] = self.weight
        return w

    # ---------- physics ----------
    def wrench_matrix(self, pose, coils=None) -> np.ndarray:
        """K (6 x n_coils) at one pose. Coils beyond ``analysis.cutoff_mm`` are 0."""
        return self.wrench_matrices(np.atleast_2d(pose), coils=coils)[0]

    def wrench_matrices(self, poses, coils=None, chunk: int = 32) -> np.ndarray:
        """K for many poses at once: returns (M, 6, n_coils)."""
        poses = np.atleast_2d(np.asarray(poses, float))
        if poses.shape[1] == 3:
            poses = np.hstack([poses, np.zeros((len(poses), 3))])
        M, C = len(poses), self.n_coils
        out = np.zeros((M, 6, C))
        allowed = np.ones(C, bool) if coils is None else np.isin(np.arange(C), coils)
        S = self.seg_mid.shape[1]
        # keep (poses x points x magnets) per Magpylib call bounded -> modest memory use
        per_pose = max(1, int((self.cutoff / max(min(self.grid.tile_size) / np.sqrt(len(self.grid.layout.coil_xy)), 1e-3)) ** 2 * 4)) * S
        chunk = int(max(1, min(chunk, 400_000 // (per_pose * max(len(self.mover.magnets), 1)) + 1)))
        for start in range(0, M, chunk):
            p = poses[start : start + chunk]
            d = np.linalg.norm(self.grid.coil_xy[None, :, :] - p[:, None, :2], axis=-1)  # (m, C)
            near = (d <= self.cutoff) & allowed[None, :]
            active = np.flatnonzero(near.any(0))
            if active.size == 0:
                continue
            mids = self.seg_mid[active].reshape(-1, 3)
            dls = self.seg_dl[active].reshape(-1, 3)
            B = magpy.getB(self.mover.sources(p), mids, sumup=True, squeeze=False)
            B = B.reshape(len(p), -1, 3)  # (m, N, 3)
            dF = np.cross(dls[None], B)  # force on coil pieces per amp
            com = p[:, :3] + self.mover.com
            dT = np.cross(mids[None] - com[:, None, :], dF)
            F = dF.reshape(len(p), active.size, S, 3).sum(2)  # (m, A, 3)
            T = dT.reshape(len(p), active.size, S, 3).sum(2)
            block = -np.concatenate([F, T], axis=2).transpose(0, 2, 1)  # mover feels the reaction
            block *= near[:, active][:, None, :]
            out[start : start + len(p)][:, :, active] = block
        return out

    def wrench(self, pose, currents) -> np.ndarray:
        return self.wrench_matrix(pose) @ np.asarray(currents, float)

    def describe(self) -> str:
        g = self.grid
        ex, ey = g.extent
        lines = [
            f"Design '{self.cfg['name']}'",
            f"  Mover : {self.mover.describe()}",
            f"  Coil  : {self.coil.describe()}",
            f"  Tiles : {g.rows} x {g.cols} tiles of {g.tile_size[0]*1e3:.0f} x {g.tile_size[1]*1e3:.0f} mm, "
            f"{len(g.layout.coil_xy)} coils/tile ({g.layout.name}), {g.n_coils} coils total, "
            f"table {ex*1e3:.0f} x {ey*1e3:.0f} mm",
            f"  Gap   : {self.gap*1e3:.1f} mm, usable current {self.max_current:.2f} A/coil",
        ]
        lines += [f"  Note  : {n}" for n in self.coil.notes + self.check_geometry()]
        return "\n".join(lines)

    def check_geometry(self) -> list[str]:
        """Plain-language warnings about designs that cannot be built."""
        warn = []
        xy = self.grid.layout.coil_xy
        if len(xy) > 1:
            dd = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
            dmin = dd[dd > 0].min()
            o = self.coil.outline
            span = 2 * o["r_out"] if o["shape"] == "circle" else min(o["outer"])
            if span > dmin + 1e-9:
                warn.append(
                    f"WARNING: coils overlap (coil is {span*1e3:.1f} mm wide, spacing {dmin*1e3:.1f} mm) "
                    "- only possible with stacked layers."
                )
        if self.mover.size > min(self.grid.tile_size):
            warn.append("Mover is larger than a tile (the project scope assumes a smaller mover).")
        return warn


def build_system(cfg: dict | None = None) -> System:
    """Build a System from a (full or partial) config dict."""
    if cfg is None or "_source" not in cfg:
        cfg = make_config(cfg or {}, strict=False)
    mover = build_mover(cfg["mover"])
    coil = build_coil(cfg["coil"])
    layout = build_layout(cfg["tile"])
    grid = build_grid(layout, int(cfg["grid"]["rows"]), int(cfg["grid"]["cols"]))
    # place every coil's segments in the table frame
    offs = np.column_stack([grid.coil_xy, np.zeros(grid.n_coils)])
    seg_mid = coil.seg_mid[None, :, :] + offs[:, None, :]
    seg_dl = np.broadcast_to(coil.seg_dl, (grid.n_coils,) + coil.seg_dl.shape).copy()
    return System(cfg, mover, coil, grid, seg_mid, seg_dl)
