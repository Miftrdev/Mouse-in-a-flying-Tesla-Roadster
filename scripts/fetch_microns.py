#!/usr/bin/env python3
"""Build data/connectome.npz from the public MICrONS cubic-millimetre release.

Source: one cubic millimetre of visual cortex from one mouse (dataset "minnie65"),
imaged by electron microscopy. https://www.microns-explorer.org/cortical-mm3
Cite: MICrONS Consortium et al., "Functional connectomics spanning multiple areas of
mouse visual cortex", Nature 640, 435-447 (2025). Read their terms before using the data.

What this script does
  1. Downloads the cell-type table (which segment is which kind of neuron, and where).
  2. Streams the connection table (about 3 GB compressed: one row per connected pair,
     with its synapse count) and keeps every pair whose two ends are both typed neurons
     with their cell bodies inside the volume.
  3. Writes one .npz. Nothing is filtered by hand.

Honest limits of the data, which carry over to anything built on it:
  - It is a piece of cortex, not a whole brain. No eyes, no muscles, no feeding circuit.
  - Most axons are automatically traced and not proofread, so many are cut short. Local
    connections are well covered; long-range ones are under-counted.

Usage
  pip install pandas
  python scripts/fetch_microns.py                       # public bucket, version 1412
  python scripts/fetch_microns.py --cells cells.csv.gz --synapses conn.csv.gz   # local files

Needs ~3 GB of bandwidth and a few GB of RAM. The downloads are kept in data/ and reused.
A raw synapse table (one row per synapse, no n_syn column) also works with --synapses.
"""
import argparse
import gzip
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousedev import connectome  # noqa: E402

BUCKET = "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v{ver}/"
CELLS = "aibs_metamodel_celltypes_v661_merged"     # which segment is which cell type, and where
SYN = "connections_with_nuclei"                    # pre, post, n_syn for every connected pair
VOXEL_NM = (4.0, 4.0, 40.0)


def _download(url: str, dest: Path) -> Path:
    import httpx
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  using cached {dest}")
        return dest
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        if r.status_code != 200:
            raise SystemExit(f"  {r.status_code} for {url}\n  Check the version number and file names at "
                             "https://tutorial.microns-explorer.org/static-repositories.html, download the two "
                             "files yourself, and pass them with --cells and --synapses.")
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 22):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:6.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def _try_header(url: str, dest: Path):
    """Header files in the bucket list one column per line as `name,dtype`."""
    try:
        p = _download(url, dest)
    except SystemExit:
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) == 1:                                   # plain one-line CSV header
        return [c.strip() for c in lines[0].split(",")]
    return [ln.split(",")[0].strip() for ln in lines]


def _has_header_row(path: Path, header) -> bool:
    """Some dumps repeat the column names as the first data row, some do not."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.readline().strip().split(",")[0].strip().strip('"')
    return first == header[0]


def _col(cols, *needles, required=True):
    for want in needles:
        for c in cols:
            if c == want or c.endswith("." + want) or c.endswith("_" + want) and want.startswith("pt_"):
                return c
    for want in needles:
        for c in cols:
            if want in c:
                return c
    if required:
        raise SystemExit(f"could not find a column like {needles} in {list(cols)}")
    return None


def load_cells(path: Path, header):
    import pandas as pd
    if header:
        df = pd.read_csv(path, names=header, header=None, skiprows=1 if _has_header_row(path, header) else 0,
                         low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    root = _col(cols, "pt_root_id")
    ctype = _col(cols, "cell_type")
    klass = _col(cols, "classification_system", required=False)
    if klass:
        df = df[df[klass].astype(str).str.contains("neuron") & ~df[klass].astype(str).str.contains("non")]
    df = df[df[root].astype("int64") != 0]
    df = df.drop_duplicates(subset=root, keep=False)       # two cell bodies in one segment = a merge error
    px = _col(cols, "pt_position_x", required=False)
    if px:
        pos = df[[px, px[:-1] + "y", px[:-1] + "z"]].to_numpy(float)
    else:
        raw = df[_col(cols, "pt_position")].astype(str).str.strip("[]() ").str.split(r"[,\s]+", expand=True)
        pos = raw.iloc[:, :3].to_numpy(float)
    if np.nanmax(pos) < 1e6:                                 # voxels -> nanometres
        pos = pos * np.array(VOXEL_NM)
    return df[root].to_numpy("int64"), (pos / 1000.0).astype(np.float32), df[ctype].astype(str).to_numpy()


def load_synapses(path: Path, header, root_ids: np.ndarray, chunk: int = 2_000_000):
    import pandas as pd
    index = pd.Index(root_ids)
    n = len(root_ids)
    opener = gzip.open if str(path).endswith(".gz") else open
    keys, vals, seen, kept = [], [], 0, 0
    skip = header is not None and _has_header_row(path, header)
    with opener(path, "rt", encoding="utf-8") as f:
        if header is None:
            header = [c.strip() for c in f.readline().strip().split(",")]
        elif skip:
            f.readline()
        pre_c, post_c = _col(header, "pre_pt_root_id"), _col(header, "post_pt_root_id")
        n_c = "n_syn" if "n_syn" in header else None       # absent in a raw one-row-per-synapse table
        use = [pre_c, post_c] + ([n_c] if n_c else [])
        for df in pd.read_csv(f, names=header, usecols=use, chunksize=chunk,
                              dtype={pre_c: "int64", post_c: "int64"}):
            i, j = index.get_indexer(df[pre_c]), index.get_indexer(df[post_c])
            ok = (i >= 0) & (j >= 0) & (i != j)
            key = i[ok].astype(np.int64) * n + j[ok]
            w = df[n_c].to_numpy(float)[ok] if n_c else np.ones(len(key))
            k, inv = np.unique(key, return_inverse=True)
            keys.append(k); vals.append(np.bincount(inv, weights=w, minlength=len(k)))
            seen += len(df); kept += int(ok.sum())
            print(f"\r  {seen / 1e6:8.1f} M rows read, {kept / 1e6:6.2f} M between typed neurons", end="", flush=True)
    print()
    k = np.concatenate(keys); v = np.concatenate(vals)
    uk, inv = np.unique(k, return_inverse=True)
    nsyn = np.bincount(inv, weights=v).astype(np.float32)
    return (uk // n).astype(np.int32), (uk % n).astype(np.int32), nsyn, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, default=1412, help="MICrONS materialization version")
    ap.add_argument("--cells", help="local cell-type table (csv or csv.gz) instead of downloading")
    ap.add_argument("--synapses", help="local connection or synapse table (csv or csv.gz) instead of downloading")
    ap.add_argument("--out", default="data/connectome.npz")
    a = ap.parse_args()
    base, data = BUCKET.format(ver=a.version), Path("data")
    data.mkdir(exist_ok=True)

    print("1/3 cell types")
    if a.cells:
        cells, cells_header = Path(a.cells), None
    else:
        cells_header = _try_header(f"{base}{CELLS}_header.csv", data / f"{CELLS}_header.csv")
        cells = _download(f"{base}{CELLS}.csv.gz", data / f"{CELLS}.csv.gz")
    root_ids, pos_um, ctype = load_cells(cells, cells_header)
    print(f"  {len(root_ids):,} typed neurons")

    print("2/3 connections")
    if a.synapses:
        syn, syn_header = Path(a.synapses), None
    else:
        syn_header = _try_header(f"{base}{SYN}_header.csv", data / f"{SYN}_header.csv")
        syn = _download(f"{base}{SYN}.csv.gz", data / f"{SYN}.csv.gz")
    pre, post, nsyn, seen = load_synapses(syn, syn_header, root_ids)

    print("3/3 writing")
    connectome.save(a.out, root_ids, pos_um, ctype, pre, post, nsyn, {
        "source": f"MICrONS minnie65 public v{a.version} (1 mm3 of mouse visual cortex)",
        "synthetic": False, "version": a.version, "cell_table": CELLS, "synapse_table": SYN,
        "rows_scanned": int(seen),
        "citation": "MICrONS Consortium et al., Nature 640, 435-447 (2025). doi:10.1038/s41586-025-08790-w",
    })
    c = connectome.load(a.out)
    print(f"  {a.out}: {c.n:,} neurons, {c.n_edges:,} connections, {int(c.nsyn.sum()):,} synapses")
    print(f"  sha256 {c.sha256}\nNext: mousedev check && mousedev calibrate")


if __name__ == "__main__":
    main()
