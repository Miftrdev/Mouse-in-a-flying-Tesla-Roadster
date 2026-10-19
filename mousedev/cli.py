"""Command line.

    mousedev start        everything in order: download the brain, calibrate, ask for
                          name / ticker / picture, launch, then run
    mousedev check        show what brain file is loaded and its hash
    mousedev calibrate    find the usable weights for this brain; runs the shuffled control
    mousedev launch       let the cortex steer the cursor to LAUNCH and press it
    mousedev run          the loop: claim fees, sniff the cheeses, feed one holder, post
    mousedev once         a single round
    mousedev replay F     re-run a decision record and compare
    mousedev verify       check the hash chain of all records
    mousedev serve        serve web/ (the site reads web/state.json)
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys

from . import calibrate, config, connectome, ledger


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mousedev", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    stp = sub.add_parser("start")
    stp.add_argument("--name"); stp.add_argument("--symbol"); stp.add_argument("--image")
    sub.add_parser("check")
    c = sub.add_parser("calibrate"); c.add_argument("--target-hz", type=float, default=45.0)
    for name in ("launch", "once"):
        s = sub.add_parser(name); s.add_argument("--shuffled", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--ticks", type=int, default=None); r.add_argument("--fast", action="store_true")
    r.add_argument("--shuffled", action="store_true")
    rp = sub.add_parser("replay"); rp.add_argument("record")
    sub.add_parser("verify")
    sv = sub.add_parser("serve"); sv.add_argument("--port", type=int, default=8787)
    a = ap.parse_args(argv)
    cfg = config.load(a.config)

    if a.cmd == "start":
        return _start(a, cfg)

    if a.cmd == "check":
        print(json.dumps(connectome.load(cfg.brain.connectome).stats(), indent=1))
    elif a.cmd == "calibrate":
        cal = calibrate.run(connectome.load(cfg.brain.connectome), cfg, a.target_hz)
        print(json.dumps({k: v for k, v in cal.items() if k != "table"}, indent=1))
    elif a.cmd == "replay":
        res = ledger.replay(json.loads(open(a.record, encoding="utf-8").read()), connectome.load(cfg.brain.connectome))
        print("REPLAY OK: same sniffs, same pick, same table" if res["ok"] else "REPLAY FAILED:\n  " + "\n  ".join(res["problems"]))
        return 0 if res["ok"] else 1
    elif a.cmd == "verify":
        res = ledger.verify_chain(cfg.state_dir)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    elif a.cmd == "serve":
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=cfg.web_dir)
        print(f"serving {cfg.web_dir}/ on http://localhost:{a.port}")
        http.server.ThreadingHTTPServer(("", a.port), h).serve_forever()
    else:
        from .app import Mouse
        conn = connectome.load(cfg.brain.connectome)
        st = calibrate.status(conn)
        if st != "ok":
            print(f"Calibration is {st} for this brain file. Run:  mousedev calibrate")
            return 1
        m = Mouse(cfg, conn=conn, shuffled=getattr(a, "shuffled", False))
        mode = "PAPER MODE (nothing is sent)" if m.paper else "LIVE"
        print(f"{mode} · brain: {m.conn.label} · {m.conn.n:,} neurons · sha {m.conn.sha256[:12]}")
        if a.cmd == "launch":
            t = m.launch()
            print(f"pressed: {t['pressed']} after {len(t['steps'])} steps" + (f" · mint {t['mint']}" if t["pressed"] else
                  " · the cursor never settled on the button, so nothing was launched"))
            return 0 if t["pressed"] else 2
        if a.cmd == "once":
            from .app import Postponed
            try:
                print(json.dumps(m.tick(), indent=1)[:3000])
            except Postponed as e:
                print(f"waiting: {e}")
                return 1
        if a.cmd == "run":
            m.run(a.ticks, a.fast)
    return 0


def _start(a, cfg) -> int:
    """Download the brain if missing, calibrate, take the coin's name / ticker / picture
    from the operator, let the cortex launch it, then hand over to the loop."""
    import subprocess
    from pathlib import Path
    from .app import Mouse

    if not Path(cfg.brain.connectome).exists():
        print("1/4 No brain file yet. Downloading the public MICrONS data (about 3 GB).")
        fetch = Path(__file__).resolve().parents[1] / "scripts" / "fetch_microns.py"
        if subprocess.run([sys.executable, str(fetch), "--out", cfg.brain.connectome]).returncode:
            return 1
    else:
        print("1/4 Brain file found.")
    conn = connectome.load(cfg.brain.connectome)
    print(f"    {conn.label} · {conn.n:,} neurons · sha256 {conn.sha256[:16]}")

    st = calibrate.status(conn)
    if st != "ok":
        why = {"missing": "", "old": " (the saved calibration is from an older version)",
               "other-brain": " (the saved calibration is for a different brain file)"}[st]
        print(f"2/4 Calibrating{why}. One line per setting tried; this takes about 20 minutes.")
        cal = calibrate.run(conn, cfg)
        print(f"    chosen: inh x{cal['inh_gain']:g}, w_syn {cal['w_syn']} ({cal['backoff'] * 100:.0f}% of the edge) · "
              f"a cheese gives {cal['ref']['sniff_hz']} Hz, the launch button {cal['ref']['press_hz']} Hz")
        print("    each place with its own cheese: " + "  ".join(f"{x:g}" for x in cal["slot_ref_hz"]) + "  Hz")
        print("    empty table:                    " + "  ".join(f"{x:g}" for x in cal["empty_table_hz"]) + "  Hz")
        for note in cal["notes"]:
            print("    Warning: " + note)
        cfg = config.load(a.config)
    else:
        print("2/4 Already calibrated.")
    rc = cfg.reward
    print(f"    rule: a round lasts {rc.round_min_seconds / 60:g} to {rc.round_max_seconds / 60:g} min; {rc.table_size} cheeses; "
          f"{rc.holder_share * 100:g}% of the round's creator fees go to one wallet holding at least {rc.min_hold_sol:g} SOL of the coin")

    m = Mouse(cfg, conn=conn)
    print(("    PAPER MODE: nothing is signed or sent." if m.paper else "    LIVE.") + f" Wallet {m.chain.pubkey}")
    if not cfg.chain.mint:
        print("3/4 The coin. These three come from you, not from the mouse.")
        ask = sys.stdin.isatty()
        for field, label in (("name", "Name"), ("symbol", "Ticker"), ("image", "Picture file")):
            given = getattr(a, field)
            if not given and ask:
                given = input(f"    {label} [{getattr(cfg.token, field)}]: ").strip()
            if given:
                setattr(cfg.token, field, given)
        if not m.paper and not cfg.token.metadata_uri and not Path(cfg.token.image).exists():
            print(f"    Picture not found: {cfg.token.image}")
            return 1
        t = m.launch()
        if not t["pressed"]:
            print("    The cursor never settled on the button. Nothing was launched.")
            return 2
        print(f"    Pressed after {len(t['steps'])} moves at {t['press_hz']} Hz · first buy {t['dev_buy_sol']} SOL · mint {t['mint']}")
    else:
        print(f"3/4 Coin already launched: {cfg.chain.mint}")
    print("4/4 Running. Ctrl-C to stop.")
    m.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
