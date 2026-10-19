"""Calibration: find where this brain gives a signal that means something.

Two numbers are set here. Both are global; nothing is fitted per cell.
    w_syn     mV per synapse, the same for every synapse
    inh_gain  how much stronger an inhibitory synapse is than an excitatory one

The rule. Show the cortex one cheese on its own. Raise w_syn as far as it will go while
layer-5 activity stays under that cheese instead of spreading over the whole sheet. That
point is the edge. A few settings below the edge are then tried, for a few inhibition
strengths, and scored:
    a stronger answer under the cheese is better (up to a point),
    an answer that stays under the cheese is better,
    an answer that every place on the screen can give is better,
    an answer that the same cells with scrambled wiring cannot match is better,
    an answer that is the same from one seed to the next is better,
    and the launch button has to stay a local spot too, or the cursor has nothing to follow.
The best one that can also press the launch button is kept.

Why. With strong weights and weak inhibition a recurrent network ignites: any picture sets
the whole sheet firing at the same rate, and the output no longer says where the cheese is.
The usable regime is below that point.

Then each place is measured: its rate with its own cheese shown, averaged over a few seeds.
A sniff is later read as a share of that number, because some patches of this cortex answer
louder than others whatever they are shown.

Printed and saved alongside: the same pictures on shuffled wiring, and an empty table.
If shuffled wiring does as well as the real one, the connectome is not doing the work.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import brain
from .config import Config
from .connectome import Connectome
from .rig import Rig

VERSION = 5
GAINS = (1.0, 2.0, 4.0, 8.0)
BACKOFFS = (0.85, 0.60)   # settings tried, as a share of the edge
SEEDS = (7, 8, 9)
LEAK = 0.15            # activity outside the picture may be at most this share of the activity under it
QUIET_HZ = 0.02        # below this under the picture there is nothing to judge yet
STRONG_HZ = 1.0        # an answer above this earns no extra credit
MIN_PRESS_HZ = 0.03    # below this under the launch button, a press cannot be told from silence
PATH = Path("data/calibration.json")


def status(conn: Connectome) -> str:
    """ok | missing | old | other-brain"""
    if not PATH.exists():
        return "missing"
    cal = json.loads(PATH.read_text(encoding="utf-8"))
    if cal.get("version") != VERSION:
        return "old"
    return "ok" if cal.get("brain_sha256") == conn.sha256 else "other-brain"


def run(conn: Connectome, cfg: Config, target_hz: float = 45.0, gains=GAINS, backoffs=BACKOFFS, iters: int = 6,
        write: bool = True, log=print) -> dict:
    from .launch import PAD_H, PAD_W, _view
    rig, sim_s = Rig(conn, cfg.rig), cfg.brain.sim_ms / 1000
    types, n_sims, t0, n_slots = cfg.rig.inhibitory_types, 0, time.time(), rig.n_slots
    ends = (0, n_slots - 1)                      # two places far apart, used where measuring all would take too long

    lc = cfg.launch
    lrig = Rig(conn, replace(cfg.rig, grid_x=lc.grid_x, grid_z=lc.grid_z))
    button = _view(lc.grid_x, lc.grid_z, PAD_W * lc.fov / 2, PAD_H * lc.fov / 2, (PAD_W / 2, PAD_H / 2),
                   (PAD_W / 2, PAD_H / 2), lc.button_w, lc.button_h)
    centre = (slice(lc.grid_z // 2 - 1, lc.grid_z // 2 + 1), slice(lc.grid_x // 2 - 1, lc.grid_x // 2 + 1))

    def scaled(m1, w):
        m = m1.copy()
        m.data *= np.float32(w)
        return m

    def sniff(m, slot, seed, ms=None):
        """(rate under the cheese, highest rate under any other place) for one cheese shown alone."""
        nonlocal n_sims
        n_sims += 1
        ms = ms or cfg.brain.sim_ms
        return rig.smell(brain.simulate(m, cfg.brain, rig.drive(rig.frame([slot])), seed, sim_ms=ms), slot, ms / 1000)

    def local(m, ms=None):
        for slot in ends:
            own, spill = sniff(m, slot, SEEDS[0], ms)
            if own > target_hz or (own >= QUIET_HZ and spill > LEAK * own):
                return False
        return True

    def press(m):
        """(rate under the button, rate elsewhere, step length) with the cursor on the launch button."""
        hz, off, step_ms = 0.0, 0.0, 200.0
        for step_ms in (200.0, 400.0, 800.0):
            cols = lrig.column_rates(brain.simulate(m, cfg.brain, lrig.drive(button), 11, sim_ms=step_ms * 3),
                                     step_ms * 3 / 1000)
            hz, off = float(cols[centre].mean()), float(cols[button == 0].mean()) if (button == 0).any() else 0.0
            if hz * step_ms / 1000 >= 0.3:          # at least ~0.3 spikes per cell per step
                break
        return hz, off, step_ms

    shuffled = conn.shuffled()
    rows = []
    for inh in gains:
        m1, ms1 = conn.matrix(1.0, inh, types), shuffled.matrix(1.0, inh, types)
        lo, hi = 0.0, 1.0
        while hi < 64 and local(scaled(m1, hi), 500.0):             # climb until the sheet ignites
            lo, hi = hi, hi * 2
        for _ in range(iters):                                      # then close in on that point
            mid = (lo + hi) / 2
            if local(scaled(m1, mid), 500.0):
                lo = mid
            else:
                hi = mid
        for back in backoffs:
            w = round(lo * back, 5)
            m, ms = scaled(m1, w), scaled(ms1, w)
            first = [sniff(m, s, SEEDS[0]) for s in range(n_slots)]
            again = [sniff(m, s, SEEDS[1])[0] for s in ends]
            ctrl = [sniff(ms, s, SEEDS[0])[0] for s in ends]
            own = np.array([f[0] for f in first])
            b, spill = float(own.mean()), float(np.mean([f[1] for f in first]))
            once = float(np.mean([own[s] for s in ends]))
            steady = 1.0 - 0.5 * abs(once - np.mean(again)) / (once + np.mean(again) + 1e-9)
            alive = float((own >= 0.25 * max(own.max(), 1e-9)).mean())
            hz, off, step_ms = press(m)
            n_sims += 1
            spot = hz < QUIET_HZ or off <= 2 * LEAK * hz
            score = 0.0 if b < QUIET_HZ else (min(b, STRONG_HZ) * (1.0 if spill <= LEAK * b else 0.25) * alive
                                              * (1.0 if once >= 2 * np.mean(ctrl) else 0.5) * steady
                                              * (1.0 if spot else 0.25))
            rows.append({"inh_gain": inh, "w_edge": round(lo, 5), "backoff": back, "w_syn": w,
                         "sniff_hz": round(b, 3), "by_place_hz": [round(float(x), 3) for x in own],
                         "spill_hz": round(spill, 3), "shuffled_hz": round(float(np.mean(ctrl)), 3),
                         "places_alive": round(alive, 2), "steady": round(float(steady), 3),
                         "press_hz": round(hz, 3), "press_off_hz": round(off, 3), "press_step_ms": step_ms,
                         "score": round(float(score), 3)})
            log(f"    inh x{inh:<3g} w_syn {w:<8g} cheese {b:5.2f} Hz  around it {spill:5.2f}  shuffled {rows[-1]['shuffled_hz']:5.2f}  "
                f"places {alive * 100:3.0f}%  button {hz:5.2f} / off {off:5.2f}  score {score:5.2f}   "
                f"({n_sims} runs, {time.time() - t0:.0f} s)")

    # Best score first. Equal scores: the setting that scrambled wiring is worst at copying.
    # The winner must also be able to press the launch button, so the next best is taken if not.
    ranked = sorted(rows, key=lambda r: (round(r["score"], 2), -r["shuffled_hz"] / max(r["sniff_hz"], 1e-9)),
                    reverse=True)
    best = next((r for r in ranked[:3] if r["score"] > 0 and r["press_hz"] >= MIN_PRESS_HZ), ranked[0])
    m = scaled(conn.matrix(1.0, best["inh_gain"], types), best["w_syn"])

    # Each place's own reference: its cheese alone, a few seeds. Places that barely answer get
    # a floor so that dividing by a tiny number cannot make them win on noise.
    by_seed = [[round(sniff(m, s, seed)[0], 3) for s in range(n_slots)] for seed in SEEDS[1:]]
    means = np.mean([best["by_place_hz"]] + by_seed, axis=0)
    floor = 0.25 * float(np.median(means))
    slot_ref = [round(float(max(x, floor, 1e-3)), 3) for x in means]
    n_sims += 1
    blank = rig.slot_rates(brain.simulate(m, cfg.brain, rig.drive(rig.frame([])), SEEDS[0]), sim_s)

    notes = []
    if best["score"] <= 0:
        notes.append("No setting gave a usable answer under a cheese. Do not run on this calibration.")
    if best["spill_hz"] > LEAK * best["sniff_hz"]:
        notes.append("Activity spreads well beyond the cheese at the chosen setting.")
    if best["places_alive"] < 1:
        notes.append(f"Only {best['places_alive'] * 100:.0f}% of the places answer their cheese properly; "
                     f"a wallet put on a deaf place will rarely be picked that round.")
    if best["sniff_hz"] < 2 * best["shuffled_hz"]:
        notes.append("Shuffled wiring does at least half as well as the real wiring. The connectome is doing little here.")
    if best["press_hz"] < MIN_PRESS_HZ:
        notes.append("Layer 5 barely answers the launch button at this setting. The launch may not press.")
    if float(blank.max()) >= cfg.rig.pick_min_frac * min(slot_ref):
        notes.append("An empty place reads above the pick threshold. Picks cannot be trusted here.")
    out = {
        "version": VERSION, "brain_sha256": conn.sha256, "source": conn.label,
        "w_syn": best["w_syn"], "inh_gain": best["inh_gain"], "backoff": best["backoff"], "target_hz": target_hz,
        "leak": LEAK, "slot_ref_hz": slot_ref, "slot_by_seed_hz": [best["by_place_hz"]] + by_seed,
        "empty_table_hz": [round(float(x), 3) for x in blank],
        "ref": {"sniff_hz": round(float(np.mean(means)), 3), "press_hz": best["press_hz"]},
        "launch_step_ms": best["press_step_ms"], "table": rows, "notes": notes,
        "runs": n_sims, "seconds": round(time.time() - t0),
    }
    if write:
        PATH.parent.mkdir(exist_ok=True)
        PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
