"""Decentralised tiles: who is in charge of the mover, and what crosses the links.

Current strategy (``owner_computes``, the lit-review "source tile keeps control"
option with hysteresis):
  * The *owner* tile is the one under the mover centre. Ownership only moves to
    a neighbour once the mover is ``handoff.hysteresis_mm`` past the tile edge,
    so it does not flicker back and forth on the boundary.
  * The owner runs the controller and the current allocation for every coil
    near the mover, including coils in neighbouring tiles.
  * Its own coils get the new currents immediately; neighbour tiles receive
    theirs over a link and apply them ``handoff.link_delay_steps`` control steps
    later (models the UART/CAN latency).

To try another strategy (e.g. "each tile allocates its own coils from a shared
force command"), subclass TileNetwork and override ``apply``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .materials import mm


class TileNetwork:
    def __init__(self, system):
        self.system = system
        self.grid = system.grid
        h = system.cfg["handoff"]
        self.hysteresis = mm(h["hysteresis_mm"])
        self.delay = int(h["link_delay_steps"])
        self.owner: int | None = None
        self.history = deque(maxlen=self.delay + 1)
        self.messages = 0  # current-command messages sent to neighbours
        self.handoffs = 0  # ownership changes
        self.log: list[tuple] = []  # (step, kind, from_tile, to_tile)
        self.step = 0

    def update_owner(self, xy) -> int:
        under = self.grid.tile_at(xy)
        if self.owner is None:
            self.owner = under
        elif under != self.owner:
            r, c = self.grid.tile_rc(self.owner)
            w, h = self.grid.tile_size
            lo = np.array([c * w, r * h]) - self.hysteresis
            hi = np.array([(c + 1) * w, (r + 1) * h]) + self.hysteresis
            if np.any(xy < lo) or np.any(xy > hi):
                self.log.append((self.step, "handoff", self.owner, under))
                self.owner = under
                self.handoffs += 1
        return self.owner

    def apply(self, commanded: np.ndarray) -> np.ndarray:
        """Currents the coils actually receive this step."""
        self.history.append(np.array(commanded, float))
        old = self.history[0]  # sent `delay` steps ago (or the oldest we have)
        mine = self.grid.coil_tile == self.owner
        out = np.where(mine, commanded, old)
        others = set(np.unique(self.grid.coil_tile[(~mine) & (np.abs(commanded) > 1e-6)]))
        self.messages += len(others)
        self.step += 1
        return out


class CentralController(TileNetwork):
    """Reference case: one controller drives every coil instantly (no tiles)."""

    def apply(self, commanded):
        self.step += 1
        return np.array(commanded, float)
