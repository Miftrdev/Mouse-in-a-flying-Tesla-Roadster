"""The mouse's day: claim fees, check the table, sniff, eat one cheese, write it down, post it."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import connectome, launch, ledger, poster, rewards
from .chain import LiveChain, PaperChain
from .config import Config
from .market import LiveMarket, PaperMarket
from .rig import Rig

LAYER = {"23P": 0, "4P": 1, "5P-IT": 2, "5P-ET": 2, "5P-NP": 2, "6P-IT": 3, "6P-CT": 3}


class Postponed(Exception):
    """The round could not start (no price, no holder list). Nothing was changed; try again shortly."""


class Mouse:
    def __init__(self, cfg: Config, conn: connectome.Connectome | None = None, shuffled: bool = False,
                 paper_seed: int = 1):
        self.cfg = cfg
        self.conn = conn or connectome.load(cfg.brain.connectome)
        if shuffled:
            self.conn = self.conn.shuffled()
        self.paper = cfg.chain.dry_run
        if not self.paper and (self.conn.synthetic or self.conn.meta.get("shuffled")):
            raise SystemExit("Refusing to go live: this connectome is a synthetic test fixture or a shuffled "
                             "control, not mouse data. Build the real one with scripts/fetch_microns.py.")
        self.w = self.conn.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
        self.rig = Rig(self.conn, cfg.rig)
        if len(cfg.rig.slot_ref_hz) != self.rig.n_slots:
            raise SystemExit(f"rig.slot_ref_hz has {len(cfg.rig.slot_ref_hz)} values for {self.rig.n_slots} places. "
                             f"Run:  mousedev calibrate")
        self.state_dir = Path(cfg.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        if self.paper:
            self.market = PaperMarket(seed=paper_seed)
            self.chain = PaperChain(self.market, seed=paper_seed)
        else:
            key = os.environ.get("MOUSEDEV_PRIVATE_KEY", "")
            if not key:
                raise SystemExit("MOUSEDEV_PRIVATE_KEY is not set (see .env.example)")
            self.chain = LiveChain(os.environ.get("MOUSEDEV_RPC_URL", "https://api.mainnet-beta.solana.com"),
                                   key, cfg.chain.pool, cfg.chain.slippage_pct, cfg.chain.priority_fee)
            self.market = LiveMarket(cfg.chain.mint) if cfg.chain.mint else None
        self.poster = poster.Poster(cfg.state_dir, cfg.post.telegram_chat, cfg.post.x_enabled, self.paper)
        self.recent: list[dict] = []
        self._last = None                      # (picture, spike counts, column map) of the latest round
        self._seen: dict = {}
        self._export_points()
        self._export_state()

    # ---- persistence ------------------------------------------------------------------
    def _load_state(self) -> rewards.State:
        p = self.state_dir / "state.json"
        if not p.exists():
            return rewards.State()
        try:
            return rewards.State(**json.loads(p.read_text(encoding="utf-8")))
        except TypeError:
            raise SystemExit(f"{p} was written by an older version of this program. "
                             f"Move the {self.state_dir} folder away (or delete it) and start again.")

    def _save_state(self) -> None:
        (self.state_dir / "state.json").write_text(json.dumps(self.state.to_json(), indent=1), encoding="utf-8")

    # ---- one round --------------------------------------------------------------------
    def tick(self, now: float | None = None) -> dict:
        cfg, st, rc = self.cfg, self.state, self.cfg.reward
        now = round(time.time() if now is None else now, 3)
        mint = cfg.chain.mint or "PAPERMINT"
        if self.market is None:
            raise Postponed("no coin yet: run  mousedev launch  or set chain.mint")

        if self.paper:
            self.market.step()
            self.chain.accrue(self.market.volume_sol)
            self.chain.crowd_step()

        # Read the world first. If any of it is missing the round waits and nothing is touched.
        snap = self.market.snapshot(12)
        price_sol = float(snap.get("price_sol") or 0.0)
        if price_sol <= 0:
            raise Postponed(f"no price for the coin right now ({snap.get('error', 'no pair listed yet')})")
        try:
            holders, coverage = self.chain.holders(mint)
            elig = rewards.eligible(holders, price_sol, rc, self.chain.pubkey)
            plain = set(self.chain.plain_wallets([w for w, _ in elig]))
            elig = [e for e in elig if e[0] in plain]
        except Exception as e:
            raise Postponed(f"could not read the holders: {str(e)[:200]}")

        fees, claim_sig = 0.0, ""
        try:
            claim_sig, fees = self.chain.claim_fees()
        except Exception as e:                       # nothing to claim is the usual reason
            print("    claim:", str(e)[:160])
        fees = round(fees, 6)
        if fees > 0:
            st.total_fees_claimed += fees
            st.pot_sol = round(st.pot_sol + rc.holder_share * fees, 6)
            self.poster.post(poster.fees_text(fees, rc.holder_share, claim_sig))
            self._save_state()                       # the pot is on disk before anything else can fail

        entropy = self.chain.entropy()
        seed = ledger.make_seed(st.prev_hash, entropy)
        table, gone = rewards.still_good(st.table, elig, st.fed)
        table, added = rewards.top_up(table, elig, st.fed, self.rig.n_slots, rc.table_size, seed + 3_000_017)
        spendable = round(max(0.0, self.chain.sol_balance() - rc.reserve_sol), 6)
        senses = {
            "price_sol": price_sol, "price_usd": float(snap.get("price_usd") or 0.0),
            "mcap_usd": float(snap.get("mcap_usd") or 0.0),
            "fees_sol": fees, "pot_sol": st.pot_sol, "spendable_sol": spendable,
            "holders_seen": len(holders), "coverage": coverage,
            "eligible": elig, "eligible_count": len(elig), "table": table, "gone": gone, "added": added,
        }

        read, frame, counts, columns = ledger.think(self.conn, self.w, self.rig, cfg, table, seed)
        state_before = st.to_json()
        action = rewards.decide(read["sniffs"], read["picked"], st.pot_sol, spendable, rc)
        rec = ledger.build_record(st.n + 1, now, senses, seed, entropy, read, action, state_before, self.conn, cfg)
        path = ledger.write_record(cfg.state_dir, rec)       # written before anything is sent
        st.n, st.prev_hash, st.last_round_ts = rec["n"], rec["hash"], now
        self._save_state()
        self.poster.post(poster.round_text(rec))

        rec["txs"] = self._act(action)
        after = ledger.after_round(rec, sent="reward" in rec["txs"], n_slots=self.rig.n_slots)
        rec["after"] = after
        st.table, st.fed, st.last_winner = after["table"], after["fed"], after["last_winner"]
        st.next_round_ts = now + after["next_round_s"]
        if after["new_cycle"]:
            st.cycle += 1
        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        self._publish(path)
        self._save_state()
        self.recent = (self.recent + [rec])[-40:]
        self._last = (frame, counts, columns)
        self._seen = {"price_usd": senses["price_usd"], "price_sol": price_sol, "mcap_usd": senses["mcap_usd"],
                      "candles_pct": list(snap.get("candles_pct") or []), "holders_seen": len(holders),
                      "eligible": len(elig), "coverage": coverage}
        self._export_state()
        return rec

    def _act(self, action: rewards.Action) -> dict:
        st, txs = self.state, {}
        if action.kind != "feed":
            return txs
        try:
            sig = self.chain.send_sol(action.wallet, action.sol)
            txs["reward"] = sig
            st.pot_sol = round(max(0.0, st.pot_sol - action.sol), 6)
            st.total_rewards_sol = round(st.total_rewards_sol + action.sol, 6)
            st.rewards_count += 1
            self._save_state()                       # the pot is down on disk the moment the transfer lands
            self.poster.post(poster.fed_text(action.wallet, action.sol, sig),
                             short=f"🐭 #{st.n} ate the cheese of {poster.short(action.wallet)}: "
                                   f"{action.sol:.3f} SOL sent.\n{poster.solscan(sig)}")
        except Exception as e:
            txs["error"] = str(e)[:300]
            self.poster.post(poster.fail_text(str(e)))
        return txs

    # ---- launch -----------------------------------------------------------------------
    def launch(self) -> dict:
        if self.cfg.chain.mint:
            raise SystemExit(f"a coin already exists for this rig: {self.cfg.chain.mint}")
        trace = launch.run(self.conn, self.w, self.cfg, self.chain.entropy())
        if trace["pressed"]:
            tk, th = self.cfg.token, self.cfg.launch.press_hz
            # How hard it pressed sets the dev buy, inside the range the operator allowed.
            frac = self.cfg.launch.press_frac
            ref = th / max(frac, 1e-9)                       # the on-button rate measured in calibration
            k = float(np.clip((trace["press_hz"] / max(ref, 1e-9) - frac) / (1.5 - frac), 0, 1))
            dev_buy = round(tk.dev_buy_min_sol + k * (tk.dev_buy_sol - tk.dev_buy_min_sol), 4)
            sig, mint = self.chain.create_token(tk, dev_buy)
            trace["tx"], trace["mint"], trace["dev_buy_sol"] = sig, mint, dev_buy
            (self.state_dir / "token.json").write_text(json.dumps(
                {"mint": mint, "create_tx": sig, "name": tk.name, "symbol": tk.symbol, "dev_buy_sol": dev_buy}, indent=1))
            self.cfg.chain.mint = mint
            if not self.paper:
                self.market = LiveMarket(mint)
            self.poster.post(poster.launch_text(tk.name, tk.symbol, mint, len(trace["steps"]), sig,
                                                trace["press_hz"], dev_buy))
        (self.state_dir / "launch.json").write_text(json.dumps(trace), encoding="utf-8")
        web = Path(self.cfg.web_dir)
        if web.exists():
            (web / "launch.json").write_text(json.dumps(trace), encoding="utf-8")
        self._export_state()
        return trace

    # ---- website feed -------------------------------------------------------------------
    def _export_points(self, k: int = 5000) -> None:
        web = Path(self.cfg.web_dir)
        rng = np.random.default_rng(0)
        self._sample = np.sort(rng.choice(self.conn.n, size=min(k, self.conn.n), replace=False))
        if not web.exists():
            return
        p = self.conn.pos[self._sample]
        p = (p - p.mean(0)) / (np.ptp(p, axis=0).max() + 1e-9)
        layer = [LAYER.get(t, 4) for t in self.conn.ctype[self._sample]]
        (web / "brain_points.json").write_text(json.dumps(
            {"source": self.conn.label, "synthetic": self.conn.synthetic,
             "points": np.round(p, 4).tolist(), "layer": layer}))

    def _publish(self, path: Path) -> None:
        """Copy a record to web/decisions/ so the website serves it and anyone can replay it."""
        web = Path(self.cfg.web_dir)
        if web.exists():
            (web / "decisions").mkdir(exist_ok=True)
            (web / "decisions" / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    def _export_state(self) -> None:
        web = Path(self.cfg.web_dir)
        if not web.exists():
            return
        cfg, st, rc = self.cfg, self.state, self.cfg.reward
        frame, counts, columns = self._last or (None, None, None)

        def places(table: list) -> list:            # the site shows who is on the table, not what they hold
            return [{"slot": c["slot"], "wallet": c["wallet"]} for c in table]

        def brief(r: dict) -> dict:
            return {"n": r["n"], "ts": r["ts"], "fees_sol": r["senses"]["fees_sol"], "pot_sol": r["senses"]["pot_sol"],
                    "eligible": r["senses"]["eligible_count"], "gone": r["senses"]["gone"],
                    "table": places(r["senses"]["table"]), "read": r["read"], "action": r["action"],
                    "txs": r.get("txs", {}), "hash": r["hash"], "prev_hash": r["prev_hash"],
                    "seed": str(r["seed"]), "entropy": r["entropy"]}

        out = {
            "updated": time.time(), "paper": self.paper, "subject": poster.SUBJECT,
            "token": {"name": cfg.token.name, "symbol": cfg.token.symbol, "mint": cfg.chain.mint,
                      "twitter": cfg.token.twitter, "telegram": cfg.token.telegram},
            "wallet": self.chain.pubkey,
            "brain": {"source": self.conn.label, "sha256": self.conn.sha256, "neurons": self.conn.n,
                      "connections": self.conn.n_edges, "synthetic": self.conn.synthetic},
            "reward": {"round_min_seconds": rc.round_min_seconds, "round_max_seconds": rc.round_max_seconds,
                       "holder_share": rc.holder_share, "table_size": rc.table_size},
            "totals": {"rounds": st.n, "fees_claimed": round(st.total_fees_claimed, 4),
                       "rewards_sol": round(st.total_rewards_sol, 4), "holders_fed": st.rewards_count,
                       "cycle": st.cycle, "fed_this_cycle": len(st.fed)},
            "now": {"pot_sol": st.pot_sol, "last_round_ts": st.last_round_ts, **self._seen},
            "table": places(st.table),
            "frame": np.round(frame, 3).tolist() if frame is not None else [],
            "fired": np.flatnonzero(counts[self._sample] > 0).tolist() if counts is not None else [],
            "columns_hz": np.round(columns, 2).tolist() if columns is not None else [],
            "slots": {"x": cfg.rig.slots_x, "z": cfg.rig.slots_z, "ref_hz": cfg.rig.slot_ref_hz,
                      "pick_min": cfg.rig.pick_min_frac},
            "cfg": cfg.decision_snapshot(),
            "rounds": [brief(r) for r in self.recent[-30:]][::-1],
        }
        (web / "state.json").write_text(json.dumps(out), encoding="utf-8")

    # ---- loop -------------------------------------------------------------------------
    def run(self, ticks: int | None = None, fast: bool = False) -> None:
        if fast and not self.paper:
            raise SystemExit("--fast is for paper mode only")
        done, clock = 0, time.time()
        while ticks is None or done < ticks:
            # --fast (paper only) runs on a virtual clock that jumps straight to the next round.
            now = clock if fast else time.time()
            if fast or now >= self.state.next_round_ts:
                try:
                    rec = self.tick(now=now)
                except Postponed as e:
                    print(f"      waiting: {e}")
                    if fast:
                        raise SystemExit(1)
                    time.sleep(30)
                    continue
                except KeyboardInterrupt:
                    raise
                except Exception as e:                # a bad minute on the network must not stop the rounds
                    if fast:
                        raise
                    print(f"      trouble: {type(e).__name__}: {str(e)[:200]}  (trying again in 30 s)")
                    time.sleep(30)
                    continue
                s, a = rec["senses"], rec["action"]
                did = (f"ate {poster.short(a['wallet'])}  {a['sol']:.4f} SOL" if a["kind"] == "feed"
                       else f"rest ({a['why']})")
                print(f"#{rec['n']:>4} fees {s['fees_sol']:>7.4f}  pot {s['pot_sol']:>7.4f} SOL  "
                      f"qualify {s['eligible_count']:>4}  cheeses {len(s['table'])}  -> {did}"
                      + (f"  [failed: {rec['txs']['error'][:80]}]" if "error" in rec.get("txs", {}) else "")
                      + f"   next in {rec['after']['next_round_s'] / 60:.1f} min")
                clock, done = now + rec["after"]["next_round_s"], done + 1
            if not fast:
                time.sleep(5)
