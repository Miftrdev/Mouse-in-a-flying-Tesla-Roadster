"""Who gets a cheese, and how big it is. Plain rules; no neurons in this file.

A round lasts between `round_min_seconds` and `round_max_seconds`. At its end:

  1. Creator fees are claimed. `holder_share` of what arrived goes into the pot.
  2. The table is checked: a wallet whose coins are now worth less than `min_hold_sol`,
     because it sold or because the price fell, loses its cheese.
  3. The cortex sniffs every cheese on the table and picks one (rig.py).
  4. The whole pot is sent to that wallet. If nothing was picked, the pot waits.
  5. The table is cleared and `table_size` wallets are drawn for the next round. The ones
     that were on the table and not picked go back in the draw.

No wallet is fed twice until every eligible wallet has been fed once. Then the cycle starts
again, and the wallet fed last is kept off the first table of the new cycle.

The draw and the length of the next round both come from the round's seed (previous record
hash + a recent blockhash), so they are fixed before the simulation runs and can be
re-derived by anyone from the record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import RewardCfg


@dataclass
class State:
    n: int = 0
    prev_hash: str = "genesis"
    table: list = field(default_factory=list)       # cheeses on show: {"slot", "wallet", "value_sol"}
    next_round_ts: float = 0.0                      # when this table is sniffed
    fed: list = field(default_factory=list)         # wallets fed in the current cycle
    cycle: int = 1
    last_winner: str = ""
    pot_sol: float = 0.0                            # holders' share that has not been sent yet
    last_round_ts: float = 0.0
    total_fees_claimed: float = 0.0
    total_rewards_sol: float = 0.0
    rewards_count: int = 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Action:
    kind: str                 # "feed" | "rest"
    wallet: str = ""
    slot: int = -1
    sol: float = 0.0
    why: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def eligible(holders: list, price_sol: float, cfg: RewardCfg, own_wallet: str = "") -> list:
    """[[wallet, value_sol], ...] for wallets whose coins are worth at least min_hold_sol,
    sorted by wallet so the list is the same whoever builds it."""
    skip = set(cfg.exclude) | {own_wallet}
    out = [[w, round(tokens * price_sol, 6)] for w, tokens in holders
           if w not in skip and tokens * price_sol >= cfg.min_hold_sol]
    return sorted(out)


def still_good(table: list, elig: list, fed: list) -> tuple[list, list]:
    """Split the table into cheeses that stay and wallets that no longer qualify and lose theirs."""
    worth = dict(elig)
    keep = [{"slot": c["slot"], "wallet": c["wallet"], "value_sol": worth[c["wallet"]]}
            for c in table if c["wallet"] in worth and c["wallet"] not in fed]
    gone = [c["wallet"] for c in table if c["wallet"] not in worth]
    return keep, gone


def top_up(table: list, elig: list, fed: list, n_slots: int, size: int, seed: int) -> tuple[list, list]:
    """Fill the empty places on the table from wallets that qualify right now and have not been
    fed this cycle. The table is normally drawn a round ahead; this covers the first round and a
    round after the holders changed, so cheeses are never missing while someone qualifies.
    Returns (table, wallets added)."""
    worth = dict(elig)
    on = {c["wallet"] for c in table}
    fresh = [w for w in sorted(worth) if w not in on and w not in set(fed)]
    room = max(0, min(size, n_slots) - len(table))
    if not fresh or not room:
        return table, []
    rng = np.random.default_rng(seed)
    chosen = [fresh[i] for i in rng.permutation(len(fresh))[:room]]
    free = [s for s in range(n_slots) if s not in {c["slot"] for c in table}]
    places = [free[i] for i in rng.permutation(len(free))[:len(chosen)]]
    added = [{"slot": s, "wallet": w, "value_sol": worth[w]} for s, w in zip(places, chosen)]
    return sorted(table + added, key=lambda c: c["slot"]), [c["wallet"] for c in added]


def draw_table(elig: list, fed: list, last_winner: str, n_slots: int, size: int, seed: int) -> tuple[list, bool]:
    """Draw the wallets for the next round and a place for each. Returns (table, new_cycle).
    new_cycle is True when every eligible wallet has been fed and the list starts over; the
    caller then clears `fed`."""
    worth = dict(elig)
    fresh = [w for w in sorted(worth) if w not in set(fed)]
    new_cycle = False
    if not fresh and worth:
        new_cycle = True
        fresh = [w for w in sorted(worth) if w != last_winner] or sorted(worth)
    if not fresh:
        return [], new_cycle
    rng = np.random.default_rng(seed)
    chosen = [fresh[i] for i in rng.permutation(len(fresh))[:min(size, n_slots)]]
    places = [int(s) for s in rng.permutation(n_slots)[:len(chosen)]]
    table = [{"slot": s, "wallet": w, "value_sol": worth[w]} for s, w in zip(places, chosen)]
    return sorted(table, key=lambda c: c["slot"]), new_cycle


def round_length(cfg: RewardCfg, seed: int) -> int:
    """Seconds until the next round, between the two limits in the config."""
    lo, hi = cfg.round_min_seconds, max(cfg.round_min_seconds, cfg.round_max_seconds)
    return int(np.random.default_rng(seed).integers(lo, hi + 1))


def decide(sniffs: list, picked: int | None, pot_sol: float, spendable_sol: float, cfg: RewardCfg) -> Action:
    """The pick is the cortex's. Whether anything can be sent, and how much, is decided here."""
    if not sniffs:
        return Action("rest", why="no cheese on the table")
    if picked is None:
        return Action("rest", why="it sniffed every cheese and none got an answer from layer 5")
    c = sniffs[picked]
    sol = round(min(pot_sol, max(0.0, spendable_sol)), 6)
    if sol < cfg.min_reward_sol:
        return Action("rest", wallet=c["wallet"], slot=c["slot"],
                      why="it picked a cheese, but the pot is empty")
    return Action("feed", wallet=c["wallet"], slot=c["slot"], sol=sol,
                  why=f"liked it most: {c['liking']:.2f}x that place's calibrated rate")
