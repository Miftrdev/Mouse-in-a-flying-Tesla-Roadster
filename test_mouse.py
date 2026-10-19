import json

import numpy as np
import pytest

from mousedev import brain, ledger, rewards
from mousedev.app import Mouse, Postponed
from mousedev.rig import Rig


def _table(*slots):
    return [{"slot": s, "wallet": f"W{s}", "value_sol": 1.0} for s in slots]


def _think(conn, cfg, table, seed=11, shuffled=False):
    c = conn.shuffled() if shuffled else conn
    w = c.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
    return ledger.think(c, w, Rig(c, cfg.rig), cfg, table, seed)[0]


def test_simulation_is_deterministic(conn, cfg):
    w = conn.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
    rig = Rig(conn, cfg.rig)
    d = rig.drive(rig.frame([2]))
    a = brain.simulate(w, cfg.brain, d, 5, sim_ms=200)
    assert np.array_equal(a, brain.simulate(w, cfg.brain, d, 5, sim_ms=200))
    assert not np.array_equal(a, brain.simulate(w, cfg.brain, d, 6, sim_ms=200))


def test_the_screen_is_eight_places_and_a_cheese_lights_one(conn, cfg):
    rig = Rig(conn, cfg.rig)
    assert rig.n_slots == 8 and rig.frame([]).sum() == 0
    assert all(rig.frame([s]).sum() == 12 for s in range(8))            # 3 x 4 columns each
    assert rig.frame(range(8)).min() == 1                               # together they cover the screen


def test_a_cheese_is_smelled_under_its_own_place_and_nowhere_else(conn, cfg):
    read = _think(conn, cfg, _table(0, 3, 6))
    assert [s["slot"] for s in read["sniffs"]] == [0, 3, 6]
    for s in read["sniffs"]:
        assert s["rate_hz"] > 10 and s["spill_hz"] < 0.15 * s["rate_hz"]
    assert read["picked"] is not None
    assert _think(conn, cfg, [])["picked"] is None                      # empty table, nothing to pick


def test_wiring_matters_shuffled_control_smells_nothing(conn, cfg):
    ctrl = _think(conn, cfg, _table(0, 3, 6), shuffled=True)
    assert all(s["rate_hz"] < 1 for s in ctrl["sniffs"]) and ctrl["picked"] is None


def test_the_pick_is_the_best_liked_cheese_above_the_floor(conn, cfg):
    rig = Rig(conn, cfg.rig)
    sn = lambda *likes: [{"slot": i, "wallet": f"W{i}", "liking": x} for i, x in enumerate(likes)]
    assert rig.pick(sn(0.9, 1.4, 1.1)) == 1
    assert rig.pick(sn(1.2, 1.2)) == 0                                  # a tie goes to the lower place
    assert rig.pick(sn(0.1, 0.2)) is None                               # nothing answered: no pick
    assert rig.pick([]) is None


def test_only_wallets_worth_enough_get_a_cheese(cfg):
    rc = cfg.reward
    rc.exclude = ["TEAM"]
    holders = [["B", 2_000_000.0], ["A", 600_000.0], ["SMALL", 150_000.0], ["TEAM", 9e9], ["ME", 9e9]]
    elig = rewards.eligible(holders, 1e-6, rc, own_wallet="ME")          # a coin is worth 0.000001 SOL
    assert elig == [["A", 0.6], ["B", 2.0]]
    assert rewards.eligible(holders, 2.5e-7, rc, own_wallet="ME") == [["B", 0.5]]   # price quarters, A drops out


def test_nobody_is_fed_twice_until_everybody_was_fed_once():
    elig = [[f"W{i:02d}", 1.0] for i in range(20)]
    table, fed, last, order, cycles = [], [], "", [], 0
    for rnd in range(70):
        if table:                                                       # the mouse eats one cheese per round
            last = table[rnd % len(table)]["wallet"]
            fed = fed + [last]
            order.append(last)
        table, new_cycle = rewards.draw_table(elig, fed, last, 8, 7, seed=rnd)   # the whole table is drawn again
        assert len({c["slot"] for c in table}) == len(table) <= 7 and all(0 <= c["slot"] < 8 for c in table)
        if new_cycle:
            assert last not in {c["wallet"] for c in table}             # the last one fed waits a round
            fed, cycles = [], cycles + 1
        assert not {c["wallet"] for c in table} & set(fed)
    assert cycles >= 2
    for i in range(0, len(order) - 19, 20):
        assert len(set(order[i:i + 20])) == 20                          # each block of twenty feeds twenty different wallets
    again, _ = rewards.draw_table(elig, [], "", 8, 7, seed=3)
    assert len(again) == 7
    assert again == rewards.draw_table(elig, [], "", 8, 7, seed=3)[0] != rewards.draw_table(elig, [], "", 8, 7, seed=4)[0]
    few, _ = rewards.draw_table(elig, [w for w, _ in elig[:18]], "", 8, 7, seed=1)
    assert {c["wallet"] for c in few} == {"W18", "W19"}                 # only two are left in this cycle


def test_a_round_lasts_between_the_two_limits_and_the_seed_fixes_it(cfg):
    rc = cfg.reward
    lengths = [rewards.round_length(rc, seed) for seed in range(200)]
    assert min(lengths) >= rc.round_min_seconds == 600 and max(lengths) <= rc.round_max_seconds == 900
    assert len(set(lengths)) > 50 and rewards.round_length(rc, 7) == rewards.round_length(rc, 7)
    rc.round_min_seconds = rc.round_max_seconds = 60
    assert rewards.round_length(rc, 1) == 60


def test_a_wallet_that_sells_loses_its_cheese():
    table = [{"slot": 0, "wallet": "A", "value_sol": 1.0}, {"slot": 4, "wallet": "B", "value_sol": 2.0}]
    keep, gone = rewards.still_good(table, [["B", 1.5]], [])
    assert keep == [{"slot": 4, "wallet": "B", "value_sol": 1.5}] and gone == ["A"]


def test_empty_places_are_filled_from_whoever_qualifies_now():
    elig = [[f"W{i}", 1.0] for i in range(20)]
    table, added = rewards.top_up([], elig, [], 8, 7, seed=5)
    assert len(table) == 7 and sorted(added) == sorted(c["wallet"] for c in table)
    assert len({c["slot"] for c in table}) == 7 and all(0 <= c["slot"] < 8 for c in table)
    assert rewards.top_up([], elig, [], 8, 7, seed=5)[0] == table                     # the seed fixes it
    kept = [{"slot": 4, "wallet": "B", "value_sol": 1.5}]
    full, more = rewards.top_up(kept, [["B", 1.5], ["C", 1.0], ["D", 1.0]], ["C"], 8, 7, seed=1)
    assert more == ["D"] and [c["wallet"] for c in full] != ["B"] and all(c["slot"] != 4 or c["wallet"] == "B" for c in full)
    assert rewards.top_up(full, [["B", 1.5], ["D", 1.0]], [], 8, 7, seed=1) == (full, [])  # nobody new, nothing changes
    assert rewards.top_up([], [["C", 1.0]], ["C"], 8, 7, seed=1) == ([], [])             # already fed this cycle


def test_the_reward_is_the_pot_and_never_more_than_the_wallet_can_spend(cfg):
    rc = cfg.reward
    sniffs = [{"slot": 2, "wallet": "A", "liking": 1.1}]
    assert rewards.decide([], None, 1.0, 5.0, rc).kind == "rest"
    assert rewards.decide(sniffs, None, 1.0, 5.0, rc).kind == "rest"
    a = rewards.decide(sniffs, 0, 1.5, 5.0, rc)
    assert (a.kind, a.wallet, a.slot, a.sol) == ("feed", "A", 2, 1.5)
    assert rewards.decide(sniffs, 0, 1.5, 0.4, rc).sol == 0.4
    assert rewards.decide(sniffs, 0, 0.0, 5.0, rc).kind == "rest"        # empty pot: the cheese stays


def test_paper_run_pays_half_the_fees_one_wallet_at_a_time_and_replays(conn, cfg):
    m = Mouse(cfg, conn=conn)
    assert m.paper
    recs = [m.tick(now=1000.0 + 600 * i) for i in range(14)]
    assert len(recs[0]["senses"]["table"]) == 7 and len(recs[0]["senses"]["added"]) == 7   # the first round fills its own table
    assert len(recs[0]["after"]["table"]) == 7
    fed = [r["action"]["wallet"] for r in recs if r["action"]["kind"] == "feed"]
    assert len(fed) >= 6
    seen, cycles = set(), 1
    for r in recs:
        if r["action"]["kind"] == "feed":
            assert r["action"]["wallet"] not in seen                    # no wallet twice inside a cycle
            seen.add(r["action"]["wallet"])
        if r["after"]["new_cycle"]:
            assert {w for w, _ in r["senses"]["eligible"]} <= seen      # it only starts over once everyone was fed
            seen, cycles = set(), cycles + 1
    st = m.state
    assert st.cycle == cycles and st.rewards_count == len(fed) and set(st.fed) == seen
    assert st.total_rewards_sol + st.pot_sol == pytest.approx(cfg.reward.holder_share * st.total_fees_claimed, abs=1e-4)
    for r in recs:
        s, a = r["senses"], r["action"]
        assert all(v >= cfg.reward.min_hold_sol for _, v in s["eligible"])
        if a["kind"] == "feed":
            assert a["wallet"] in {c["wallet"] for c in s["table"]} and a["sol"] == pytest.approx(s["pot_sol"])
            assert a["wallet"] not in {c["wallet"] for c in r["after"]["table"]}
            assert r["txs"]["reward"].startswith("DRYRUN")
        assert 600 <= r["after"]["next_round_s"] <= 900
    assert ledger.verify_chain(cfg.state_dir) == {"records": 14, "ok": True, "problems": []}
    one = next(r for r in recs if r["action"]["kind"] == "feed")
    on_disk = json.loads(open(f"{cfg.state_dir}/decisions/{one['n']:06d}.json").read())
    assert ledger.replay(on_disk, conn)["ok"]
    served = json.loads(open(f"{cfg.web_dir}/decisions/{one['n']:06d}.json").read())   # the site serves every record
    assert served == on_disk and ledger.replay(served, conn)["ok"]
    forged = json.loads(json.dumps(on_disk))
    forged["action"]["wallet"] = "SOMEONE-ELSE"                          # redirect the reward
    assert not ledger.replay(forged, conn)["ok"]
    swapped = json.loads(json.dumps(on_disk))
    swapped["senses"]["table"][0]["wallet"] = "SOMEONE-ELSE"             # put a friend on the table
    assert not ledger.replay(swapped, conn)["ok"]
    assert not ledger.replay(on_disk, conn.shuffled())["ok"]            # a different brain cannot pass as this one
    posts = open(f"{cfg.state_dir}/posts.log", encoding="utf-8").read()
    assert posts.count("[PAPER]") == posts.count("-" * 40)               # every paper post is marked
    state = json.loads(open(f"{cfg.web_dir}/state.json").read())
    assert state["paper"] and state["brain"]["synthetic"] and len(state["rounds"]) == 14
    assert state["table"] == [{"slot": c["slot"], "wallet": c["wallet"]} for c in st.table]      # no holdings on the site
    assert state["totals"]["holders_fed"] == len(fed) and "next_round_ts" not in state["now"]
    assert st.next_round_ts == pytest.approx(recs[-1]["ts"] + recs[-1]["after"]["next_round_s"])


def test_a_failed_transfer_keeps_the_pot_and_the_wallet_stays_in_the_draw(conn, cfg):
    m = Mouse(cfg, conn=conn)

    def broken(to, sol):
        raise RuntimeError("rpc down")
    m.chain.send_sol = broken
    rec = next(r for r in (m.tick(now=1000.0 + 600 * i) for i in range(12)) if r["action"]["kind"] == "feed")
    assert "error" in rec["txs"] and not rec["after"]["sent"]
    assert rec["action"]["wallet"] not in m.state.fed and m.state.pot_sol >= rec["action"]["sol"]
    assert m.state.rewards_count == 0 and m.state.total_rewards_sol == 0
    assert "did not go through" in open(f"{cfg.state_dir}/posts.log", encoding="utf-8").read()
    assert ledger.replay(rec, conn)["ok"]


def test_no_price_means_the_round_waits_and_nothing_changes(conn, cfg):
    m = Mouse(cfg, conn=conn)
    m.tick(now=1000.0)
    before = m.state.to_json()
    m.market.snapshot = lambda n=12: {"price_usd": 0.0, "price_sol": 0.0, "error": "feed down"}
    with pytest.raises(Postponed):
        m.tick(now=1600.0)
    assert m.state.to_json() == before


def test_live_mode_refuses_fake_brains(conn, cfg):
    cfg.chain.dry_run = False
    with pytest.raises(SystemExit, match="Refusing to go live"):
        Mouse(cfg, conn=conn)


def test_cursor_launch_reaches_button_only_with_real_wiring(conn, cfg):
    from mousedev import launch
    w = conn.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
    t = launch.run(conn, w, cfg, "blockhash-A")
    bx, by, bw, bh = t["button"]
    end = t["steps"][-1]["cursor"]
    assert t["pressed"] and abs(end[0] - bx) <= bw / 2 and abs(end[1] - by) <= bh / 2
    assert not t["steps"][0]["on_button"]
    s = conn.shuffled()
    ws = s.matrix(cfg.brain.w_syn, cfg.brain.inh_gain, cfg.rig.inhibitory_types)
    cfg.launch.max_steps = 25
    assert not launch.run(s, ws, cfg, "blockhash-A")["pressed"]


def test_reward_transfer_builds_and_signs_offline():
    from solders.keypair import Keypair
    from mousedev.chain import SYSTEM_PROGRAM, build_transfer_tx
    kp, to = Keypair(), Keypair().pubkey()
    tx = build_transfer_tx(kp, str(to), 1_500_000_000, "4vJ9JU1bJJE96FWSJKvHsmmFADCg4gpZQff4P3bkLKi")
    ix = tx.message.instructions[0]
    keys = tx.message.account_keys
    assert len(tx.message.instructions) == 1 and str(keys[ix.program_id_index]) == SYSTEM_PROGRAM
    assert bytes(ix.data) == (2).to_bytes(4, "little") + (1_500_000_000).to_bytes(8, "little")   # System transfer
    assert [keys[i] for i in ix.accounts] == [kp.pubkey(), to]
    assert tx.verify_with_results() == [True]


def test_calibration_finds_a_local_setting_and_measures_every_place(conn, cfg, tmp_path, monkeypatch):
    from mousedev import calibrate, config
    monkeypatch.chdir(tmp_path)
    assert calibrate.status(conn) == "missing"
    cal = calibrate.run(conn, cfg, gains=(2.0,), backoffs=(0.85,), iters=4, log=lambda *_: None)
    best = next(r for r in cal["table"] if r["w_syn"] == cal["w_syn"] and r["inh_gain"] == cal["inh_gain"])
    assert best["sniff_hz"] > 2 * best["shuffled_hz"] and best["spill_hz"] <= calibrate.LEAK * best["sniff_hz"]
    assert best["places_alive"] == 1 and best["press_hz"] >= calibrate.MIN_PRESS_HZ
    assert len(cal["slot_ref_hz"]) == 8 and min(cal["slot_ref_hz"]) > 0 and max(cal["empty_table_hz"]) < 0.1
    assert not cal["notes"]
    assert calibrate.status(conn) == "ok" and calibrate.status(conn.shuffled()) == "other-brain"
    loaded = config.load("no-such-config.toml")
    assert loaded.brain.w_syn == cal["w_syn"] and loaded.brain.inh_gain == cal["inh_gain"]
    assert loaded.rig.slot_ref_hz == cal["slot_ref_hz"]
    assert loaded.launch.press_hz == pytest.approx(loaded.launch.press_frac * cal["ref"]["press_hz"], abs=1e-3)


def test_an_old_config_file_is_refused_with_a_clear_message(tmp_path):
    from mousedev import config
    old = tmp_path / "config.toml"
    old.write_text("[policy]\nmeal_max_frac = 0.6\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="older version"):
        config.load(str(old))


def test_state_from_an_older_version_is_refused_with_a_clear_message(conn, cfg):
    from pathlib import Path
    Path(cfg.state_dir).mkdir(parents=True)
    (Path(cfg.state_dir) / "state.json").write_text('{"n": 3, "fullness": 0.2, "meals": []}', encoding="utf-8")
    with pytest.raises(SystemExit, match="older version"):
        Mouse(cfg, conn=conn)
