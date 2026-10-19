"""The rig: how the world gets into the cortex and how the cortex's answer gets out.

This volume is visual cortex, so everything it is told arrives as a picture. The sheet of
cortex is cut into grid_x x grid_z columns and treated as a tiny screen. The screen is
split into slots_x x slots_z places. A cheese is one holder's wallet; it sits in one place
and lights every column of that place.

A lit column becomes Poisson input to its layer-4 cells (layer 4 is where the eyes' signal
enters cortex). The answer is read from layer-5 ET cells (cortex's output to the body).

One round, for the cheeses on the table:

    SNIFF   each cheese is shown on its own for one second of simulated cortex.
            Its smell is the mean layer-5 ET rate under that place.
    LIKING  smell divided by what the same place gave in calibration. Some patches of this
            cortex answer louder than others, whatever is shown; this takes that out.
    PICK    the cheese it likes most, if any place answered at all.

Which wallet sits in which place, how much the cheese is worth and when a round happens
are the rig's doing. The pick is the cortex's.
"""

from __future__ import annotations

import numpy as np

from .config import RigCfg
from .connectome import Connectome


class Rig:
    def __init__(self, conn: Connectome, cfg: RigCfg):
        self.cfg = cfg
        self.n = conn.n
        gx, gz = cfg.grid_x, cfg.grid_z
        x, z = conn.pos[:, 0], conn.pos[:, 2]
        self.ix = np.clip(((x - x.min()) / (np.ptp(x) + 1e-9) * gx).astype(int), 0, gx - 1)
        self.iz = np.clip(((z - z.min()) / (np.ptp(z) + 1e-9) * gz).astype(int), 0, gz - 1)
        self.is_in = conn.type_mask(cfg.input_types)
        self.is_out = conn.type_mask(cfg.output_types)
        if not self.is_in.any() or not self.is_out.any():
            raise ValueError("connectome has no cells of the configured input/output types")
        self._out_idx = np.flatnonzero(self.is_out)
        self._out_cell = self.iz[self.is_out] * gx + self.ix[self.is_out]
        self._out_per_cell = np.bincount(self._out_cell, minlength=gx * gz)
        # Places: slot s covers a block of columns; slot numbers run left to right, top row first.
        self.n_slots = cfg.slots_x * cfg.slots_z
        col_slot = np.minimum(np.arange(gx) * cfg.slots_x // gx, cfg.slots_x - 1)
        row_slot = np.minimum(np.arange(gz) * cfg.slots_z // gz, cfg.slots_z - 1)
        self.slot_of = row_slot[:, None] * cfg.slots_x + col_slot[None, :]          # [gz, gx]
        self._out_slot = self.slot_of.ravel()[self._out_cell]
        self._out_per_slot = np.bincount(self._out_slot, minlength=self.n_slots)

    # ---- world -> picture -------------------------------------------------------------
    def frame(self, slots) -> np.ndarray:
        """Picture with a cheese in each of the given places, [grid_z, grid_x] in 0..1."""
        return np.isin(self.slot_of, list(slots)).astype(np.float32)

    # ---- picture -> input rates ---------------------------------------------------------
    def drive(self, frame: np.ndarray) -> np.ndarray:
        d = np.zeros(self.n, np.float32)
        d[self.is_in] = frame[self.iz[self.is_in], self.ix[self.is_in]] * self.cfg.max_in_hz
        return d

    # ---- spikes -> readouts -------------------------------------------------------------
    def column_rates(self, counts: np.ndarray, sim_s: float) -> np.ndarray:
        """Mean layer-5 ET rate in each column, as a [grid_z, grid_x] picture in Hz."""
        c = self.cfg
        tot = np.bincount(self._out_cell, weights=counts[self._out_idx], minlength=c.grid_x * c.grid_z)
        return (tot / np.maximum(self._out_per_cell, 1) / sim_s).reshape(c.grid_z, c.grid_x)

    def slot_rates(self, counts: np.ndarray, sim_s: float) -> np.ndarray:
        """Mean layer-5 ET rate under each place, in Hz."""
        tot = np.bincount(self._out_slot, weights=counts[self._out_idx], minlength=self.n_slots)
        return tot / np.maximum(self._out_per_slot, 1) / sim_s

    def smell(self, counts: np.ndarray, slot: int, sim_s: float) -> tuple[float, float]:
        """(rate under the shown cheese, highest rate under any other place)."""
        r = self.slot_rates(counts, sim_s)
        others = np.delete(r, slot)
        return float(r[slot]), float(others.max()) if others.size else 0.0

    def pick(self, sniffs: list[dict]) -> int | None:
        """Index into `sniffs` of the cheese it likes most, or None if nothing answered.
        Equal likings go to the lower place number, so the result never depends on list order."""
        best = None
        for i, s in enumerate(sniffs):
            if s["liking"] < self.cfg.pick_min_frac:
                continue
            if best is None or (s["liking"], -s["slot"]) > (sniffs[best]["liking"], -sniffs[best]["slot"]):
                best = i
        return best
