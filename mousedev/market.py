"""Market numbers the rig needs. Public endpoints only, no keys.

The price in SOL decides what a wallet's coins are worth, and so who gets a cheese.
The candles are only drawn on the website; the cortex is not shown the chart.

Live:  DexScreener (price, market cap) + GeckoTerminal (5-minute candles).
Paper: a seeded random walk, for dry runs before a coin exists.
"""

from __future__ import annotations

import httpx
import numpy as np


class LiveMarket:
    def __init__(self, mint: str):
        self.mint = mint
        self.http = httpx.Client(timeout=15, headers={"accept": "application/json"})
        self.price = 0.0

    def snapshot(self, n_candles: int = 6) -> dict:
        out = {"price_usd": 0.0, "price_sol": 0.0, "mcap_usd": 0.0, "candles_pct": [], "ok": False}
        try:
            r = self.http.get(f"https://api.dexscreener.com/latest/dex/tokens/{self.mint}")
            pairs = (r.json() or {}).get("pairs") or []
            if not pairs:
                return out
            p = max(pairs, key=lambda x: (x.get("liquidity") or {}).get("usd") or 0)
            in_sol = (p.get("quoteToken") or {}).get("symbol") in ("SOL", "WSOL")
            out.update(price_usd=float(p.get("priceUsd") or 0), price_sol=float(p.get("priceNative") or 0) if in_sol else 0.0,
                       mcap_usd=float(p.get("marketCap") or p.get("fdv") or 0), ok=True)
            self.price = out["price_usd"]
            g = self.http.get(f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{p['pairAddress']}"
                              f"/ohlcv/minute", params={"aggregate": 5, "limit": n_candles})
            rows = g.json()["data"]["attributes"]["ohlcv_list"]
            out["candles_pct"] = [round((c - o) / o * 100, 3) for _, o, _, _, c, _ in reversed(rows) if o]
        except Exception as e:  # a missing feed means an empty chart, never a crash
            out["error"] = repr(e)[:120]
        return out


class PaperMarket:
    """Random walk with fat tails. Each step() is one 5-minute candle."""
    sol_usd = 150.0

    def __init__(self, seed: int = 1, price: float = 0.00004):
        self.rng = np.random.default_rng(seed)
        self.price = price
        self.candles: list[float] = []
        self.volume_sol = 0.0

    def step(self) -> None:
        pct = float(np.clip(self.rng.standard_t(3) * 6.0, -45, 80))
        self.price *= 1 + pct / 100
        self.candles = (self.candles + [round(pct, 3)])[-12:]
        self.volume_sol = float(abs(self.rng.normal(60, 40)) * (1 + abs(pct) / 10))

    def snapshot(self, n_candles: int = 6) -> dict:
        return {"price_usd": self.price, "price_sol": self.price / self.sol_usd, "mcap_usd": self.price * 1e9,
                "candles_pct": self.candles[-n_candles:], "ok": True}
