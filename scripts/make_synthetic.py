#!/usr/bin/env python3
"""Make a SYNTHETIC cortex-shaped test fixture. This is not a brain and not mouse data.

It exists so the code can be tested without the download. The file is flagged
synthetic, every record and post made with it says so, and the bot refuses to go live
on it. Wiring is random but local and layered, like cortex: 4 -> 2/3 -> 5 -> 6, with
connection probability falling off with distance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousedev import connectome  # noqa: E402

# type, depth range in um (y), share of cells
LAYERS = [("23P", 100, 350, .26), ("4P", 350, 480, .19), ("5P-IT", 480, 640, .09), ("5P-ET", 480, 640, .05),
          ("5P-NP", 480, 640, .01), ("6P-IT", 640, 820, .14), ("6P-CT", 640, 820, .10),
          ("BC", 100, 820, .06), ("MC", 100, 820, .05), ("BPC", 100, 500, .03), ("NGC", 50, 400, .02)]
INH = {"BC", "MC", "BPC", "NGC"}
# relative strength pre -> post
GAIN = {("4P", "23P"): 3, ("4P", "4P"): 1, ("4P", "5P-IT"): 1.5, ("4P", "5P-ET"): 1.5, ("23P", "23P"): 1,
        ("23P", "5P-IT"): 3, ("23P", "5P-ET"): 3, ("23P", "5P-NP"): 2, ("5P-IT", "5P-ET"): 2,
        ("5P-IT", "5P-IT"): 1, ("5P-IT", "6P-IT"): 2, ("5P-IT", "6P-CT"): 2, ("5P-ET", "5P-ET"): .5,
        ("6P-IT", "6P-CT"): 1, ("6P-CT", "4P"): .5}


def build(n: int, seed: int, out_deg: int, lam: float):
    rng = np.random.default_rng(seed)
    types, pos = [], []
    for t, y0, y1, share in LAYERS:
        k = int(round(n * share))
        types += [t] * k
        pos.append(np.column_stack([rng.uniform(0, 1300, k), rng.uniform(y0, y1, k), rng.uniform(0, 800, k)]))
    types, pos = np.array(types), np.vstack(pos).astype(np.float32)
    n = len(types)
    tree = cKDTree(pos[:, [0, 2]])
    pre, post, nsyn = [], [], []
    for i in range(n):
        near = np.array(tree.query_ball_point(pos[i, [0, 2]], 3 * lam))
        near = near[near != i]
        if near.size == 0:
            continue
        d = np.hypot(*(pos[near][:, [0, 2]] - pos[i, [0, 2]]).T)
        if types[i] in INH:
            g = np.ones(near.size)
        else:
            g = np.array([GAIN.get((types[i], t), 1.0 if t in INH else 0.0) for t in types[near]])
        p = g * np.exp(-d / lam)
        if p.sum() <= 0:
            continue
        tg = rng.choice(near, size=min(out_deg, near.size), replace=False, p=p / p.sum()) if (p > 0).sum() >= min(out_deg, near.size) \
            else near[p > 0]
        pre += [i] * len(tg)
        post += tg.tolist()
        nsyn += rng.geometric(0.6, len(tg)).tolist()
    return np.arange(n) + 10**9, pos, types, pre, post, nsyn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-degree", type=int, default=40)
    ap.add_argument("--lambda-um", type=float, default=70.0)
    ap.add_argument("--out", default="data/connectome.npz")
    a = ap.parse_args()
    rid, pos, types, pre, post, nsyn = build(a.n, a.seed, a.out_degree, a.lambda_um)
    connectome.save(a.out, rid, pos, types, pre, post, nsyn, {
        "source": "SYNTHETIC TEST FIXTURE (not a brain)", "synthetic": True, "seed": a.seed})
    print(f"wrote {a.out}: {len(rid)} fake neurons, {len(pre)} fake connections. Not for live use.")
