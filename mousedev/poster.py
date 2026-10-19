"""Posts. Plain string templates filled from the decision record; no text model.

In paper mode nothing is sent anywhere: posts are only appended to state/posts.log, each
one marked [PAPER]. A reward that did not go through is never announced as sent.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from .chain import solscan

SUBJECT = "mouse-65"


def short(wallet: str) -> str:
    return wallet if len(wallet) <= 11 else f"{wallet[:4]}…{wallet[-4:]}"


def round_text(rec: dict) -> str:
    s, r, a = rec["senses"], rec["read"], rec["action"]
    lines = [f"🐭 {SUBJECT} · round #{rec['n']}",
             f"Fees this round {s['fees_sol']:.4f} SOL · pot for holders {s['pot_sol']:.4f} SOL",
             f"{s['eligible_count']} wallets qualify · {len(s['table'])} pads lit"]
    if s["gone"]:
        lines.append(f"{len(s['gone'])} lost their pad: they no longer hold enough of the coin")
    if r["sniffs"]:
        lines.append("")
        lines.append(f"Sniffs ({rec['brain']['neurons']:,} neurons, {rec['cfg']['brain']['sim_ms'] / 1000:g} s per cheese)")
        for i, x in enumerate(r["sniffs"]):
            mark = "  ← eats this one" if i == r["picked"] and a["kind"] == "feed" else "  ← picked" if i == r["picked"] else ""
            lines.append(f"• pad {x['slot'] + 1} · {short(x['wallet'])} · {x['rate_hz']:.2f} Hz · liking {x['liking']:.2f}{mark}")
    lines.append("")
    if a["kind"] == "feed":
        lines.append(f"Decision: eat the cheese of {short(a['wallet'])}. Send {a['sol']:.4f} SOL to that wallet.")
    else:
        lines.append(f"Decision: rest. {a['why'][0].upper() + a['why'][1:]}.")
    lines.append(f"Replay: record {rec['hash'][:12]} · brain {rec['brain']['sha256'][:12]}")
    return "\n".join(lines)


def fed_text(wallet: str, sol: float, sig: str) -> str:
    return f"🧀 Ate the cheese of {wallet}. {sol:.4f} SOL sent to that wallet.\n{solscan(sig)}"


def fees_text(sol: float, share: float, sig: str) -> str:
    return (f"💰 {sol:.4f} SOL of creator fees arrived. {sol * share:.4f} SOL goes into the pot for holders.\n"
            f"{solscan(sig)}")


def fail_text(err: str) -> str:
    return f"⚠️ The reward did not go through: {err[:160]}\nNothing was sent. The pot and the pad stay for the next flight."


def launch_text(name: str, symbol: str, mint: str, steps: int, sig: str, press_hz: float, dev_buy: float) -> str:
    return (f"🐭 {SUBJECT} launched ${symbol} ({name}).\n"
            f"Its layer-5 output steered the cursor to the launch button in {steps} moves and pressed it.\n"
            f"It pressed at {press_hz:.1f} Hz, which set its own first buy: {dev_buy:.4f} SOL.\n"
            f"Name, ticker and picture were typed by the rig, not the mouse.\n"
            f"CA {mint}\n{solscan(sig)}")


class Poster:
    def __init__(self, state_dir: str, telegram_chat: str, x_enabled: bool, paper: bool):
        self.log = Path(state_dir) / "posts.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.chat, self.x_enabled, self.paper = telegram_chat, x_enabled, paper
        self.tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    def post(self, text: str, short: str | None = None) -> None:
        tag = "[PAPER] " if self.paper else ""
        with self.log.open("a", encoding="utf-8") as f:
            f.write(tag + text + "\n" + "-" * 40 + "\n")
        if self.paper:
            return
        if self.chat and self.tg_token:
            try:
                httpx.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", timeout=15,
                           json={"chat_id": self.chat, "text": text, "disable_web_page_preview": True})
            except Exception as e:
                print("telegram:", e)
        if self.x_enabled:
            try:
                import tweepy  # optional dependency: pip install mousedev[x]
                c = tweepy.Client(consumer_key=os.environ["X_API_KEY"], consumer_secret=os.environ["X_API_SECRET"],
                                  access_token=os.environ["X_ACCESS_TOKEN"],
                                  access_token_secret=os.environ["X_ACCESS_SECRET"])
                c.create_tweet(text=(short or text)[:280])
            except Exception as e:
                print("x:", e)
