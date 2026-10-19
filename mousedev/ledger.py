"""Decision records: hash-chained, self-contained, replayable.

Every round is written to state/decisions/NNNNNN.json with everything needed to run it
again: the cheeses on the table, the eligible wallets, the seed, the config snapshot and the
SHA-256 of the connectome file. `mousedev replay` re-simulates a record and checks it gets
the same sniffs, the same pick, the same next table and the same round length.
Each record carries the hash of the one before it, so the log cannot be edited quietly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__, brain, rewards
from .config import Config, from_snapshot
from .connectome import Connectome
from .rig import Rig


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k not in ("hash", "txs", "after", "posted")}
    return hashlib.sha256(canonical(body)).hexdigest()


def make_seed(prev_hash: str, entropy: str) -> int:
    """Seed = previous record hash + a recent Solana blockhash, so it is fixed before the
    simulation runs."""
    return int.from_bytes(hashlib.sha256(f"{prev_hash}|{entropy}".encode()).digest()[:8], "big")


def think(conn: Connectome, w, rig: Rig, cfg: Config, table: list, seed: int):
    """Sniff every cheese on the table, one second of cortex each, then pick.
    Returns (readouts, picture, spike counts over all sniffs, column map in Hz)."""
    sim_s = cfg.brain.sim_ms / 1000.0
    total = np.zeros(conn.n, np.int64)
    columns = np.zeros((cfg.rig.grid_z, cfg.rig.grid_x))
    sniffs = []
    for c in sorted(table, key=lambda c: c["slot"]):
        slot = int(c["slot"])
        counts = brain.simulate(w, cfg.brain, rig.drive(rig.frame([slot])), seed + slot)
        rate, spill = rig.smell(counts, slot, sim_s)
        ref = cfg.rig.slot_ref_hz[slot]
        sniffs.append({"slot": slot, "wallet": c["wallet"], "rate_hz": round(rate, 3), "spill_hz": round(spill, 3),
                       "liking": round(rate / ref, 3) if ref > 0 else 0.0})
        total += counts
        here = rig.slot_of == slot                    # the map shows each place as it answered its own cheese
        columns[here] = rig.column_rates(counts, sim_s)[here]
    picked = rig.pick(sniffs)
    read = {"sniffs": sniffs, "picked": picked, "n_fired": int((total > 0).sum()),
            "mean_hz": round(float(total.mean() / (sim_s * max(len(sniffs), 1))), 3)}
    return read, rig.frame([s["slot"] for s in sniffs]), total, columns


def build_record(n: int, ts: float, senses: dict, seed: int, entropy: str, read: dict,
                 action: rewards.Action, state_before: dict, conn: Connectome, cfg: Config) -> dict:
    rec = {
        "n": n, "ts": round(ts, 3),
        "senses": senses, "seed": seed, "entropy": entropy,
        "read": read, "action": action.to_json(),
        "state_before": state_before,
        "brain": {"source": conn.label, "sha256": conn.sha256, "neurons": conn.n,
                  "connections": conn.n_edges, "synthetic": conn.synthetic},
        "cfg": cfg.decision_snapshot(),
        "code": {"mousedev": __version__, "numpy": np.__version__},
        "prev_hash": state_before["prev_hash"],
    }
    rec["hash"] = record_hash(rec)
    return rec


def write_record(state_dir: str, rec: dict) -> Path:
    d = Path(state_dir) / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rec['n']:06d}.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def after_round(rec: dict, sent: bool, n_slots: int) -> dict:
    """What the round leaves behind: who has been fed, the table drawn for the next round,
    and how long that round lasts. Depends on the record and on whether the transfer went
    through, nothing else."""
    s, st, a = rec["senses"], rec["state_before"], rec["action"]
    cfg = from_snapshot(rec["cfg"]).reward
    fed, last = list(st["fed"]), st["last_winner"]
    if a["kind"] == "feed" and sent:
        fed, last = fed + [a["wallet"]], a["wallet"]
    table, new_cycle = rewards.draw_table(s["eligible"], fed, last, n_slots, cfg.table_size, rec["seed"] + 1_000_003)
    return {"sent": bool(sent), "table": table, "fed": [] if new_cycle else fed, "last_winner": last,
            "new_cycle": new_cycle, "next_round_s": rewards.round_length(cfg, rec["seed"] + 2_000_003)}


def replay(rec: dict, conn: Connectome) -> dict:
    """Re-run a record from scratch. Returns {"ok": bool, "problems": [...]}"""
    problems = []
    if record_hash(rec) != rec.get("hash"):
        problems.append("record hash does not match its contents")
    if conn.sha256 != rec["brain"]["sha256"]:
        problems.append(f"different connectome file (yours {conn.sha256[:12]}, record {rec['brain']['sha256'][:12]})")
        return {"ok": False, "problems": problems}
    cfg = from_snapshot(rec["cfg"])
    w = conn.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
    rig = Rig(conn, cfg.rig)
    s, st = rec["senses"], rec["state_before"]
    keep, gone = rewards.still_good(st["table"], s["eligible"], st["fed"])
    added = []
    if "added" in s:                                    # records before 0.5.1 had no top-up
        keep, added = rewards.top_up(keep, s["eligible"], st["fed"], rig.n_slots, cfg.reward.table_size, rec["seed"] + 3_000_017)
    if keep != s["table"] or gone != s["gone"] or added != s.get("added", []):
        problems.append("the table in the record is not what the eligible list and the last table give")
    read = think(conn, w, rig, cfg, s["table"], rec["seed"])[0]
    if read != rec["read"]:
        problems.append(f"sniffs differ: got {read}, record says {rec['read']}")
    action = rewards.decide(read["sniffs"], read["picked"], s["pot_sol"], s["spendable_sol"], cfg.reward)
    if action.to_json() != rec["action"]:
        problems.append(f"action differs: got {action.to_json()}, record says {rec['action']}")
    if "after" in rec:
        after = after_round(rec, rec["after"]["sent"], rig.n_slots)
        if after != rec["after"]:
            problems.append("the next table or the round length differs from the record")
    return {"ok": not problems, "problems": problems, "read": read, "action": action.to_json()}


def verify_chain(state_dir: str) -> dict:
    files = sorted((Path(state_dir) / "decisions").glob("*.json"))
    prev, bad = None, []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if record_hash(rec) != rec["hash"]:
            bad.append(f"{f.name}: hash mismatch")
        if prev is not None and rec["prev_hash"] != prev:
            bad.append(f"{f.name}: does not follow the previous record")
        prev = rec["hash"]
    return {"records": len(files), "ok": not bad, "problems": bad}
