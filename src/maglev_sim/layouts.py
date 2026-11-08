"""Tile layouts: where the coils sit inside one tile.

A layout builder returns ``TileLayout(coil_xy, size)`` with coil centres
measured from the tile's lower-left corner. Tiles are then repeated
``grid.rows x grid.cols`` times, edge to edge. Keep the coil spacing across a
tile edge the same as inside a tile (Jansen 2008) so the mover sees no seam.

Built-in layouts (``tile.layout`` in YAML): square, offset, custom
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import mm
from .registry import LAYOUTS, lookup, register_layout


@dataclass
class TileLayout:
    coil_xy: np.ndarray  # (n, 2) coil centres from the tile's lower-left corner [m]
    size: tuple[float, float]  # (width, height) of one tile [m]
    name: str = ""


@register_layout("square")
def square(p: dict) -> TileLayout:
    """n x n coils (or nx x ny) on a square pitch, half a pitch in from the tile edges."""
    nx = int(p.get("nx", p.get("n", 4)))
    ny = int(p.get("ny", p.get("n", 4)))
    pitch = mm(p.get("pitch_mm", 35))
    xs = (np.arange(nx) + 0.5) * pitch
    ys = (np.arange(ny) + 0.5) * pitch
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return TileLayout(np.column_stack([X.ravel(), Y.ravel()]), (nx * pitch, ny * pitch), "square")


@register_layout("offset")
def offset(p: dict) -> TileLayout:
    """Rows shifted by half a pitch on alternate rows (hexagonal-like packing).

    Use an even number of rows so the pattern continues across tile edges.
    """
    nx = int(p.get("nx", p.get("n", 4)))
    ny = int(p.get("ny", p.get("n", 4)))
    pitch = mm(p.get("pitch_mm", 35))
    row = mm(p.get("row_pitch_mm", p.get("pitch_mm", 35) * np.sqrt(3) / 2))
    pts = []
    for j in range(ny):
        shift = 0.25 * pitch if j % 2 == 0 else 0.75 * pitch
        for i in range(nx):
            pts.append([shift + i * pitch, (j + 0.5) * row])
    return TileLayout(np.array(pts), (nx * pitch, ny * row), "offset")


@register_layout("custom")
def custom(p: dict) -> TileLayout:
    """Any coil positions you like: ``positions_mm: [[x, y], ...]`` and ``size_mm: [w, h]``."""
    pts = np.array(p["positions_mm"], float) * 1e-3
    w, h = p["size_mm"]
    return TileLayout(pts, (mm(w), mm(h)), "custom")


def build_layout(tile_cfg: dict) -> TileLayout:
    builder = lookup(LAYOUTS, "layout", tile_cfg["layout"])
    return builder(dict(tile_cfg.get("params", {})))


@dataclass
class Grid:
    """All coils of the whole table."""

    layout: TileLayout
    rows: int
    cols: int
    coil_xy: np.ndarray  # (C, 2) table coordinates [m]
    coil_tile: np.ndarray  # (C,) tile index = row * cols + col
    tile_size: tuple[float, float]

    @property
    def n_coils(self) -> int:
        return len(self.coil_xy)

    @property
    def n_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def extent(self) -> tuple[float, float]:
        return self.cols * self.tile_size[0], self.rows * self.tile_size[1]

    def tile_rc(self, tile: int) -> tuple[int, int]:
        return divmod(int(tile), self.cols)

    def tile_center(self, tile: int) -> np.ndarray:
        r, c = self.tile_rc(tile)
        return np.array([(c + 0.5) * self.tile_size[0], (r + 0.5) * self.tile_size[1]])

    def tile_at(self, xy) -> int:
        """Tile index under a point (clamped to the table)."""
        c = int(np.clip(np.floor(xy[0] / self.tile_size[0]), 0, self.cols - 1))
        r = int(np.clip(np.floor(xy[1] / self.tile_size[1]), 0, self.rows - 1))
        return r * self.cols + c

    def neighbours(self, tile: int) -> list[int]:
        r, c = self.tile_rc(tile)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols:
                    out.append((r + dr) * self.cols + (c + dc))
        return out


def build_grid(layout: TileLayout, rows: int, cols: int) -> Grid:
    w, h = layout.size
    xy, tile = [], []
    for r in range(rows):
        for c in range(cols):
            xy.append(layout.coil_xy + [c * w, r * h])
            tile.append(np.full(len(layout.coil_xy), r * cols + c))
    return Grid(layout, rows, cols, np.vstack(xy), np.concatenate(tile), (w, h))
