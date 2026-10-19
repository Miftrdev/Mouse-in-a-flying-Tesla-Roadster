"""Leaky integrate-and-fire simulation of the whole connectome.

Same model family as Shiu et al., Nature 2024 (the fly brain model):

    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn
    a presynaptic spike adds its weight to g of every target, after a fixed delay
    v >= v_th  ->  spike, v = v_reset, refractory for t_ref

Weights are fixed. There is no learning rule, no optimiser and no model call in here.
Given the same connectome, config, input rates and seed it returns the same spikes,
which is what makes every decision replayable.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .config import BrainCfg


def simulate(w: sparse.csr_matrix, cfg: BrainCfg, drive_hz: np.ndarray, seed: int,
             sim_ms: float | None = None) -> np.ndarray:
    """Run the network and return the spike count of every neuron.

    w         presynaptic x postsynaptic weights in mV (CSR)
    drive_hz  rate of each input fibre onto a neuron, from the rig (x cfg.afferents fibres)
    """
    n = w.shape[0]
    dt = cfg.dt_ms
    steps = int(round((sim_ms if sim_ms is not None else cfg.sim_ms) / dt))
    delay = max(1, int(round(cfg.delay_ms / dt)))
    ref_steps = int(round(cfg.t_ref_ms / dt))
    decay = np.float32(np.exp(-dt / cfg.tau_syn_ms))
    leak = np.float32(dt / cfg.tau_m_ms)

    rng = np.random.default_rng(seed)
    v = np.full(n, cfg.v_rest, np.float32)
    g = np.zeros(n, np.float32)
    refractory = np.zeros(n, np.int32)
    counts = np.zeros(n, np.int32)
    ring = np.zeros((delay, n), np.float32)

    # Expected input spikes per step. One cell hears `afferents` input fibres at once, so
    # the count per step is Poisson, not 0/1.
    lam_in = drive_hz.astype(np.float64) * cfg.afferents * dt / 1000.0
    lam_bg = cfg.bg_hz * dt / 1000.0
    driven = np.flatnonzero(lam_in > 0)
    lam_driven = lam_in[driven]

    indptr, indices, data = w.indptr, w.indices, w.data

    for t in range(steps):
        slot = t % delay
        g *= decay
        g += ring[slot]
        ring[slot] = 0.0

        if lam_bg > 0:
            g += np.float32(cfg.bg_w) * rng.poisson(lam_bg, n)
        if driven.size:
            g[driven] += np.float32(cfg.w_in) * rng.poisson(lam_driven)

        active = refractory == 0
        v[active] += leak * (cfg.v_rest - v[active] + g[active])
        refractory[~active] -= 1

        fired = np.flatnonzero(active & (v >= cfg.v_th))
        if fired.size:
            v[fired] = cfg.v_reset
            refractory[fired] = ref_steps
            counts[fired] += 1
            # Gather every outgoing synapse of the cells that fired, without a Python loop.
            starts = indptr[fired]
            lengths = indptr[fired + 1] - starts
            total = int(lengths.sum())
            if total:
                offsets = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(total)
                ring[slot] += np.bincount(indices[offsets], weights=data[offsets], minlength=n).astype(np.float32)
    return counts
