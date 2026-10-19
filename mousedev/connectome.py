"""Load a connectome file and turn it into a signed weight matrix.

File format (.npz), produced by scripts/fetch_microns.py:
    root_id   int64  [n]     segment id of each neuron in the source dataset
    pos_um    float32[n,3]   soma position in micrometres (x, y=depth, z)
    cell_type str    [n]     e.g. 23P, 4P, 5P-IT, 5P-ET, 5P-NP, 6P-IT, 6P-CT, BC, MC, BPC, NGC
    pre, post int32  [e]     edge list (indices into the arrays above)
    nsyn      float32[e]     number of synapses on that edge
    meta      str            JSON: source, version, citation, synthetic flag

Nothing here is trained. A weight is (number of synapses) x w_syn, negative if the
presynaptic cell is inhibitory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Connectome:
    root_id: np.ndarray
    pos: np.ndarray
    ctype: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    nsyn: np.ndarray
    meta: dict
    sha256: str

    @property
    def n(self) -> int:
        return int(self.root_id.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    @property
    def label(self) -> str:
        return str(self.meta.get("source", "unknown source"))

    def type_mask(self, types: list[str]) -> np.ndarray:
        return np.isin(self.ctype, np.asarray(types))

    def matrix(self, w_syn: float, inh_gain: float, inhibitory_types: list[str]) -> sparse.csr_matrix:
        """Rows = presynaptic neuron, columns = postsynaptic neuron, values in mV."""
        inh = self.type_mask(inhibitory_types)
        sign = np.where(inh[self.pre], -inh_gain, 1.0).astype(np.float32)
        data = self.nsyn.astype(np.float32) * sign * np.float32(w_syn)
        m = sparse.csr_matrix((data, (self.pre, self.post)), shape=(self.n, self.n), dtype=np.float32)
        m.sum_duplicates()
        return m

    def shuffled(self, seed: int = 0) -> "Connectome":
        """Control: same cells, same out-degrees and synapse counts, targets scrambled.

        If the real wiring matters, the shuffled brain should fail where the real one works.
        """
        rng = np.random.default_rng(seed)
        meta = dict(self.meta)
        meta["source"] = self.label + " [SHUFFLED CONTROL]"
        meta["shuffled"] = True
        return Connectome(self.root_id, self.pos, self.ctype, self.pre,
                          rng.permutation(self.post), self.nsyn, meta, self.sha256 + ":shuffled")

    def stats(self) -> dict:
        types, counts = np.unique(self.ctype, return_counts=True)
        return {
            "source": self.label,
            "synthetic": self.synthetic,
            "sha256": self.sha256,
            "neurons": self.n,
            "connections": self.n_edges,
            "synapses": int(self.nsyn.sum()),
            "cell_types": {str(t): int(c) for t, c in zip(types, counts)},
        }


def load(path: str) -> Connectome:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. The brain is not shipped with this repo.\n"
            "Build it from the public MICrONS release:  python scripts/fetch_microns.py\n"
            "(or, for tests only:  python scripts/make_synthetic.py)"
        )
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return Connectome(
        root_id=z["root_id"], pos=z["pos_um"].astype(np.float32), ctype=z["cell_type"].astype(str),
        pre=z["pre"].astype(np.int32), post=z["post"].astype(np.int32),
        nsyn=z["nsyn"].astype(np.float32), meta=meta, sha256=file_sha256(str(p)),
    )


def save(path: str, root_id, pos_um, cell_type, pre, post, nsyn, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, root_id=np.asarray(root_id, np.int64), pos_um=np.asarray(pos_um, np.float32),
        cell_type=np.asarray(cell_type, dtype=str), pre=np.asarray(pre, np.int32),
        post=np.asarray(post, np.int32), nsyn=np.asarray(nsyn, np.float32),
        meta=np.asarray(json.dumps(meta)),
    )
