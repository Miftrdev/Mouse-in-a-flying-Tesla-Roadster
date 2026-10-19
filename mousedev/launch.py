"""Launch rig: the cortex steers a cursor to the LAUNCH button and presses it.

What the mouse does:  moves the cursor and presses.
What the rig does:    types the name, ticker and picture (from config.toml), signs and sends.

How it works. The launch pad is a 1200 x 800 screen with one bright button. Each step the
rig shows the cortex a picture of the pad centred on the cursor (so the button appears off
to one side), runs 200 ms of simulation, and reads layer-5 ET activity patch by patch.
The cursor moves toward the most active patches. Layer 5 lights up under the lit layer-4
patches only because real cortical wiring is local and columnar, so with the wiring
scrambled (`--shuffled`) the cursor has nothing to follow and never arrives. When the
cursor sits on the button and the centre patches stay above the press threshold for three
steps, that is the click.

A blank screen is shown first to measure resting activity, which is subtracted afterwards.
The button and the starting point are drawn from the seed (a recent blockhash), with the
button inside the mouse's field of view. Every step is saved to state/launch.json.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import brain
from .config import Config
from .ledger import make_seed
from .rig import Rig

PAD_W, PAD_H = 1200, 800


def _view(gx: int, gz: int, half_w: float, half_h: float, cursor, button, bw: float, bh: float) -> np.ndarray:
    """Picture of the pad centred on the cursor. A patch is fully lit once a tenth of it is button."""
    cw, ch = 2 * half_w / gx, 2 * half_h / gz
    bx0, by0 = button[0] - bw / 2, button[1] - bh / 2
    f = np.zeros((gz, gx), np.float32)
    for iz in range(gz):
        for ix in range(gx):
            x0, y0 = cursor[0] - half_w + ix * cw, cursor[1] - half_h + iz * ch
            ox = max(0.0, min(x0 + cw, bx0 + bw) - max(x0, bx0))
            oy = max(0.0, min(y0 + ch, by0 + bh) - max(y0, by0))
            f[iz, ix] = min(1.0, ox * oy / (0.10 * cw * ch))
    return f


def run(conn, w, cfg: Config, entropy: str) -> dict:
    lc = cfg.launch
    rig = Rig(conn, replace(cfg.rig, grid_x=lc.grid_x, grid_z=lc.grid_z))
    gx, gz = lc.grid_x, lc.grid_z
    half_w, half_h = PAD_W * lc.fov / 2, PAD_H * lc.fov / 2
    cw, ch = 2 * half_w / gx, 2 * half_h / gz
    seed0 = make_seed("launch", entropy)
    rng = np.random.default_rng(seed0)
    button = [float(rng.uniform(lc.button_w / 2 + 20, PAD_W - lc.button_w / 2 - 20)),
              float(rng.uniform(lc.button_h / 2 + 20, PAD_H - lc.button_h / 2 - 20))]
    while True:   # start off the button, but with the button somewhere in view
        cursor = [float(rng.uniform(20, PAD_W - 20)), float(rng.uniform(20, PAD_H - 20))]
        dx0, dy0 = abs(cursor[0] - button[0]), abs(cursor[1] - button[1])
        if (dx0 > lc.button_w / 2 + 60 or dy0 > lc.button_h / 2 + 60) and dx0 < half_w - 80 and dy0 < half_h - 80:
            break

    blank = brain.simulate(w, cfg.brain, rig.drive(np.zeros((gz, gx), np.float32)), seed0, sim_ms=lc.step_ms * 3)
    rest = rig.column_rates(blank, lc.step_ms * 3 / 1000)
    gxs, gzs = np.meshgrid(np.arange(gx) + 0.5 - gx / 2, np.arange(gz) + 0.5 - gz / 2)
    centre = (slice(gz // 2 - 1, gz // 2 + 1), slice(gx // 2 - 1, gx // 2 + 1))

    steps, dwell, pressed = [], 0, False
    for k in range(lc.max_steps):
        frame = _view(gx, gz, half_w, half_h, cursor, button, lc.button_w, lc.button_h)
        counts = brain.simulate(w, cfg.brain, rig.drive(frame), seed0 + 1 + k, sim_ms=lc.step_ms)
        cols = rig.column_rates(counts, lc.step_ms / 1000)
        act = np.clip(cols - rest, 0, None)
        if act.max() > 0:
            act[act < 0.25 * act.max()] = 0      # ignore the faint halo, follow the bright spot
        tot = float(act.sum())
        dx = float((act * gxs).sum() / tot * cw) if tot > 1e-6 else 0.0
        dy = float((act * gzs).sum() / tot * ch) if tot > 1e-6 else 0.0
        norm = float(np.hypot(dx, dy))
        if norm > 160:
            dx, dy = dx / norm * 160, dy / norm * 160
        on_button = abs(cursor[0] - button[0]) <= lc.button_w / 2 and abs(cursor[1] - button[1]) <= lc.button_h / 2
        centre_hz = float(cols[centre].mean())
        dwell = dwell + 1 if (on_button and centre_hz >= lc.press_hz) else 0
        steps.append({"k": k, "cursor": [round(cursor[0], 1), round(cursor[1], 1)],
                      "move": [round(dx, 1), round(dy, 1)], "centre_hz": round(centre_hz, 2),
                      "on_button": bool(on_button), "frame": np.round(frame, 3).tolist(),
                      "patches_hz": np.round(cols, 2).tolist()})
        if dwell >= 3:
            pressed = True
            break
        cursor = [float(np.clip(cursor[0] + lc.gain * dx, 0, PAD_W)), float(np.clip(cursor[1] + lc.gain * dy, 0, PAD_H))]

    return {"pressed": pressed, "press_hz": steps[-1]["centre_hz"] if pressed else 0.0, "steps": steps,
            "button": [round(button[0], 1), round(button[1], 1), lc.button_w, lc.button_h],
            "pad": [PAD_W, PAD_H], "seed": seed0, "entropy": entropy,
            "brain": {"source": conn.label, "sha256": conn.sha256, "neurons": conn.n,
                      "synthetic": conn.synthetic, "shuffled": bool(conn.meta.get("shuffled"))},
            "cfg": cfg.decision_snapshot(),
            "operator_supplied": {"name": cfg.token.name, "symbol": cfg.token.symbol, "image": cfg.token.image}}
