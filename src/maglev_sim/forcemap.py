"""Fast force model for simulations: the mover's field on a grid.

The exact model asks Magpylib for the mover's field at every coil segment for
every pose (~20-40 ms per pose). A simulation needs K thousands of times, so
instead we sample the mover's magnetic field ONCE on a 3D grid in the mover's
own frame (a few seconds), and for each pose just move the coil segments into
that frame and interpolate. Same Lorentz-force sum as system.py, ~20x faster,
and it still handles yaw/tilt and as-built (shifted / scaled) coils.

Accuracy: trilinear interpolation on a 0.5-1 mm grid agrees with the exact
model to ~1-2 % (checked in tests/test_physics.py).
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

from .materials import mm


def _mover_key(mover) -> bytes:
    parts = []
    for m in mover.magnets:
        parts += [m.shape.encode(), np.asarray(m.dimension, float).tobytes(),
                  np.asarray(m.center, float).tobytes(), np.asarray(m.polarization, float).tobytes()]
    return b"".join(parts)


_FIELD_CACHE: dict = {}  # in-process cache: same mover + grid -> reuse the field


class FastForceModel:
    def __init__(self, system, step_mm: float = 1.0, z_step_mm: float = 0.5, gap_range_mm=(None, None)):
        self.system = system
        mover, coil = system.mover, system.coil
        reach = system.cutoff + float(np.max(np.linalg.norm(coil.seg_mid[:, :2], axis=1))) + mm(2)
        g_lo = mm(gap_range_mm[0]) if gap_range_mm[0] is not None else max(mm(0.2), system.gap - mm(3))
        g_hi = mm(gap_range_mm[1]) if gap_range_mm[1] is not None else system.gap + mm(5)
        self.gap_range = (g_lo, g_hi)
        n = int(np.ceil(reach / mm(step_mm)))
        # tiny offset keeps grid points off magnet face planes (field formulas are singular there)
        self.x0 = -n * mm(step_mm) + 1.234e-6
        self.dx = mm(step_mm)
        self.nx = 2 * n + 1
        z_top = -g_lo + mm(1)  # coil tops are at z = -gap in the mover frame
        z_bot = -(g_hi + coil.height) - mm(1)
        self.dz = mm(z_step_mm)
        self.nz = int(np.ceil((z_top - z_bot) / self.dz)) + 1
        self.z0 = z_bot + 1.234e-6
        key = (_mover_key(mover), self.x0, self.dx, self.nx, self.z0, self.dz, self.nz)
        if key not in _FIELD_CACHE:
            xy = self.x0 + np.arange(self.nx) * self.dx
            z = self.z0 + np.arange(self.nz) * self.dz
            X, Y, Z = np.meshgrid(xy, xy, z, indexing="ij")
            pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            src = mover.sources(np.zeros(6))  # mover at origin, level: frame = mover frame
            _FIELD_CACHE.clear()  # keep memory bounded: one field at a time
            B = np.empty((len(pts), 3), np.float32)
            step = max(5000, 400_000 // max(len(mover.magnets), 1))  # bounded peak memory
            for s in range(0, len(pts), step):
                B[s : s + step] = np.nan_to_num(magpy.getB(src, pts[s : s + step], sumup=True))
            del pts, X, Y, Z
            _FIELD_CACHE[key] = B.reshape(self.nx, self.nx, self.nz, 3)
        self.B = _FIELD_CACHE[key]
        self.Bflat = self.B.reshape(-1, 3)
        self.com = mover.com

    def field(self, pts: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the mover field at points in the mover frame (0 outside)."""
        fx = (pts[:, 0] - self.x0) / self.dx
        fy = (pts[:, 1] - self.x0) / self.dx
        fz = (pts[:, 2] - self.z0) / self.dz
        inside = (fx >= 0) & (fx <= self.nx - 1) & (fy >= 0) & (fy <= self.nx - 1) & (fz >= 0) & (fz <= self.nz - 1)
        i = np.clip(fx.astype(int), 0, self.nx - 2)
        j = np.clip(fy.astype(int), 0, self.nx - 2)
        k = np.clip(fz.astype(int), 0, self.nz - 2)
        tx, ty, tz = (fx - i)[:, None], (fy - j)[:, None], (fz - k)[:, None]
        Bf, sj, si = self.Bflat, self.nz, self.nx * self.nz
        base = i * si + j * sj + k

        def g(off):
            return np.take(Bf, base + off, axis=0)

        c00 = g(0) * (1 - tx) + g(si) * tx
        c10 = g(sj) * (1 - tx) + g(si + sj) * tx
        c01 = g(1) * (1 - tx) + g(si + 1) * tx
        c11 = g(sj + 1) * (1 - tx) + g(si + sj + 1) * tx
        out = (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz
        out[~inside] = 0.0
        return out

    def wrench_matrix(self, pose, plant=None) -> np.ndarray:
        """K (6 x n_coils). ``plant`` = an as-built System using the same mover."""
        sysm = plant or self.system
        pose = np.asarray(pose, float)
        if pose.size == 3:
            pose = np.r_[pose, 0, 0, 0]
        C = sysm.n_coils
        K = np.zeros((6, C))
        d = np.linalg.norm(sysm.grid.coil_xy - pose[:2], axis=1)
        near = np.flatnonzero(d <= sysm.cutoff)
        if near.size == 0:
            return K
        mids = sysm.seg_mid[near]  # (A, S, 3)
        dls = sysm.seg_dl[near]
        A, S = mids.shape[:2]
        com_w = pose[:3] + self.com
        rot = Rotation.from_euler("xyz", pose[3:6])
        local = rot.inv().apply(mids.reshape(-1, 3) - com_w) + self.com  # into the mover frame
        B = rot.apply(self.field(local)).reshape(A, S, 3)
        dF = np.cross(dls, B)
        dT = np.cross(mids - com_w, dF)
        K[:3, near] = -dF.sum(1).T
        K[3:, near] = -dT.sum(1).T
        return K
