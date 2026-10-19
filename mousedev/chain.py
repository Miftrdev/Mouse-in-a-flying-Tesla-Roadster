"""Solana side: create the coin, claim creator fees, list the holders, send a reward.

Transactions for pump.fun (create, claim) are built by PumpPortal's local-transaction API,
which returns an unsigned transaction. It is signed here, on this machine, and sent through
your own RPC. The secret key never leaves the process. A reward is a plain SOL transfer
built in this file. There is no buy, sell or burn anywhere in this code after the launch.

PaperChain is the dry-run twin: same interface, nothing signed, nothing sent, and a made-up
crowd of holders.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path

import httpx

PUMPPORTAL = "https://pumpportal.fun/api/trade-local"
PINATA = "https://uploads.pinata.cloud/v3/files"
LAMPORTS = 1_000_000_000
SYSTEM_PROGRAM = "11111111111111111111111111111111"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def solscan(sig: str) -> str:
    return f"https://solscan.io/tx/{sig}"


class ChainError(RuntimeError):
    pass


class LiveChain:
    dry = False

    def __init__(self, rpc_url: str, secret_b58: str, pool: str, slippage_pct: float, priority_fee: float):
        from solders.keypair import Keypair
        self.kp = Keypair.from_base58_string(secret_b58)
        self.pubkey = str(self.kp.pubkey())
        # One URL, or several separated by commas: the first is used, the next ones take over
        # when it does not answer, and all of them are asked for the holder list in turn.
        self.rpc_urls = [u.strip() for u in rpc_url.split(",") if u.strip()] or [rpc_url]
        self.rpc_url, self.pool = self.rpc_urls[0], pool
        self.slippage, self.priority_fee = slippage_pct, priority_fee
        self.http = httpx.Client(timeout=30)

    # ---- RPC ------------------------------------------------------------------------
    RETRIES = 5                       # a dropped or slow answer is asked again: 0.5, 1, 2, 4, 8 s apart

    def rpc(self, method: str, params: list, url: str | None = None):
        last = None
        for attempt in range(self.RETRIES + 1):
            try:
                at = url or self.rpc_urls[attempt % len(self.rpc_urls)]
                r = self.http.post(at, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code} from the RPC", request=r.request, response=r)
                r.raise_for_status()
                j = r.json()
                if "error" in j:
                    raise ChainError(f"{method}: {j['error']}")
                return j["result"]
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
                last = e
                if attempt < self.RETRIES:
                    time.sleep(0.5 * 2 ** attempt)
        raise ChainError(f"{method}: the RPC did not answer after {self.RETRIES + 1} tries ({str(last)[:120]})")

    def entropy(self) -> str:
        return self.rpc("getLatestBlockhash", [{"commitment": "finalized"}])["value"]["blockhash"]

    def sol_balance(self) -> float:
        return self.rpc("getBalance", [self.pubkey, {"commitment": "confirmed"}])["value"] / LAMPORTS

    def holders(self, mint: str) -> tuple[list, str]:
        """([[wallet, coins], ...], how complete the list is). Coins are summed per owner.

        The full list needs getProgramAccounts, which free public RPC endpoints often refuse,
        and which a load-balanced endpoint sometimes answers with an empty list. A real list is
        never empty (the coin's own pool account holds coins), so an empty answer is asked
        again, then the other URLs are asked. If none gives the list, the 20 largest token
        accounts are used instead and the record says so.
        """
        program = self.rpc("getAccountInfo", [mint, {"encoding": "base64"}])["value"]["owner"]
        filters = [{"memcmp": {"offset": 0, "bytes": mint}}]
        if program == TOKEN_PROGRAM:
            filters.insert(0, {"dataSize": 165})
        infos, coverage = [], ""
        for url in self.rpc_urls:
            for _ in range(3):
                try:
                    res = self.rpc("getProgramAccounts", [program, {"encoding": "jsonParsed", "commitment": "confirmed",
                                                                    "filters": filters}], url=url)
                except (ChainError, httpx.HTTPError):
                    break                                            # this URL refuses the method
                if res:
                    infos, coverage = [a["account"]["data"] for a in res], "all holders"
                    break
                time.sleep(1)
            if infos:
                break
        if not infos:
            top = self.rpc("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])["value"]
            res = self.rpc("getMultipleAccounts", [[t["address"] for t in top],
                                                   {"encoding": "jsonParsed", "commitment": "confirmed"}])["value"]
            infos, coverage = [a["data"] for a in res if a], "20 largest token accounts only (no RPC gave the full list)"
        coins: dict[str, float] = {}
        for d in infos:
            parsed = d.get("parsed") if isinstance(d, dict) else None
            if not parsed or parsed.get("type") != "account":
                continue
            info = parsed["info"]
            amount = float(info["tokenAmount"]["uiAmountString"] or 0)
            if amount > 0:
                coins[info["owner"]] = coins.get(info["owner"], 0.0) + amount
        return sorted([w, c] for w, c in coins.items()), coverage

    def plain_wallets(self, wallets: list[str]) -> list[str]:
        """Keep only ordinary wallets. Coins held by a program (the bonding curve, a liquidity
        pool, a lock contract) sit under an address that a program owns, and those are dropped."""
        keep = []
        for i in range(0, len(wallets), 100):
            chunk = wallets[i:i + 100]
            res = self.rpc("getMultipleAccounts", [chunk, {"encoding": "base64", "commitment": "confirmed",
                                                           "dataSlice": {"offset": 0, "length": 0}}])["value"]
            keep += [w for w, a in zip(chunk, res) if a is None or a["owner"] == SYSTEM_PROGRAM]
        return keep

    def _send(self, tx, last_valid_height: int | None = None) -> str:
        """Send and wait. With `last_valid_height` (transactions built here) a failure is
        final: the wait runs until the blockhash has expired, so a transaction reported as
        not landed can never land later and be paid twice."""
        raw = base64.b64encode(bytes(tx)).decode()
        sig = self.rpc("sendTransaction", [raw, {"encoding": "base64", "preflightCommitment": "confirmed",
                                                 "maxRetries": 3}])

        def landed() -> bool:
            st = self.rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": True}])["value"][0]
            if st and st.get("err"):
                raise ChainError(f"failed on chain: {st['err']} ({sig})")
            return bool(st) and st.get("confirmationStatus") in ("confirmed", "finalized")

        deadline = time.time() + (180 if last_valid_height else 75)
        while time.time() < deadline:
            if landed():
                return sig
            if last_valid_height and self.rpc("getBlockHeight", [{"commitment": "confirmed"}]) > last_valid_height:
                time.sleep(2.0)
                if landed():
                    return sig
                raise ChainError(f"expired without landing ({sig})")
            time.sleep(1.5)
        raise ChainError(f"not confirmed in time ({sig})")

    def _pumpportal(self, payload: dict, signers: list | None = None, as_json: bool = False) -> str:
        from solders.transaction import VersionedTransaction
        r = None
        for attempt in range(4):                 # it only builds an unsigned transaction, so asking again is safe
            try:
                r = self.http.post(PUMPPORTAL, **({"json": payload} if as_json else {"data": payload}))
                if r.status_code < 500 and r.status_code != 429:
                    break
            except httpx.TransportError:
                r = None
            time.sleep(1.0 * 2 ** attempt)
        if r is None:
            raise ChainError(f"PumpPortal {payload.get('action')}: no answer")
        if r.status_code != 200:
            raise ChainError(f"PumpPortal {payload.get('action')}: {r.status_code} {r.text[:200]}")
        unsigned = VersionedTransaction.from_bytes(r.content)
        return self._send(VersionedTransaction(unsigned.message, signers or [self.kp]))

    # ---- actions --------------------------------------------------------------------
    def claim_fees(self) -> tuple[str, float]:
        before = self.sol_balance()
        sig = self._pumpportal({"publicKey": self.pubkey, "action": "collectCreatorFee",
                                "priorityFee": self.priority_fee})
        return sig, max(0.0, self.sol_balance() - before)

    def send_sol(self, to: str, sol: float) -> str:
        lamports = int(round(sol * LAMPORTS))
        if lamports <= 0:
            raise ChainError("nothing to send")
        bh = self.rpc("getLatestBlockhash", [{"commitment": "confirmed"}])["value"]
        return self._send(build_transfer_tx(self.kp, to, lamports, bh["blockhash"]), bh["lastValidBlockHeight"])

    def create_token(self, token, dev_buy_sol: float) -> tuple[str, str]:
        from solders.keypair import Keypair
        uri = token.metadata_uri or self._upload_metadata(token)
        mint_kp = Keypair()
        sig = self._pumpportal({
            "publicKey": self.pubkey, "action": "create",
            "tokenMetadata": {"name": token.name, "symbol": token.symbol, "uri": uri},
            "mint": str(mint_kp.pubkey()), "denominatedInSol": "true", "amount": dev_buy_sol,
            "slippage": self.slippage, "priorityFee": self.priority_fee, "pool": "pump",
        }, signers=[mint_kp, self.kp], as_json=True)
        return sig, str(mint_kp.pubkey())

    def _upload_metadata(self, token) -> str:
        import json
        jwt = os.environ.get("PINATA_JWT", "")
        if not jwt:
            raise ChainError("set token.metadata_uri in config.toml, or PINATA_JWT so the rig can upload the picture")
        hdr = {"Authorization": f"Bearer {jwt}"}
        img = Path(token.image)
        r = self.http.post(PINATA, headers=hdr, data={"network": "public"},
                           files={"file": (img.name, img.read_bytes())})
        r.raise_for_status()
        image_url = f"https://ipfs.io/ipfs/{r.json()['data']['cid']}"
        meta = {"name": token.name, "symbol": token.symbol, "description": token.description,
                "image": image_url, "twitter": token.twitter, "telegram": token.telegram,
                "website": token.website}
        r = self.http.post(PINATA, headers=hdr, data={"network": "public"},
                           files={"file": ("metadata.json", json.dumps(meta).encode(), "application/json")})
        r.raise_for_status()
        return f"https://ipfs.io/ipfs/{r.json()['data']['cid']}"


def build_transfer_tx(kp, to: str, lamports: int, blockhash: str):
    """A plain SOL transfer from the mouse's wallet, signed by it."""
    from solders.hash import Hash
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer
    from solders.transaction import VersionedTransaction
    ix = transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=Pubkey.from_string(to), lamports=lamports))
    msg = MessageV0.try_compile(kp.pubkey(), [ix], [], Hash.from_string(blockhash))
    return VersionedTransaction(msg, [kp])


class PaperChain:
    """Dry run. Balances and holders are simulated; signatures are clearly fake (DRYRUN...)."""
    dry = True

    def __init__(self, market, start_sol: float = 1.0, seed: int = 1):
        import numpy as np
        self.market = market
        self.pubkey = "PAPER-WALLET-NOTHING-IS-SENT"
        self._sol = start_sol
        self._pending_fees = 0.0
        self._k = 0
        self._rng = np.random.default_rng(seed + 1000)
        self._crowd: dict[str, float] = {}       # made-up wallet -> coins
        self._born = 0
        for _ in range(40):                      # a crowd is already there when the dry run starts
            self._arrive()

    def _sig(self, what: str) -> str:
        self._k += 1
        return "DRYRUN" + hashlib.sha256(f"{what}{self._k}".encode()).hexdigest()[:40]

    def entropy(self) -> str:
        return f"paper-{self._k}-{int(self.market.price * 1e12)}"

    def accrue(self, volume_sol: float) -> None:
        self._pending_fees += volume_sol * 0.003

    def _arrive(self) -> None:
        self._born += 1
        name = "PAPER" + hashlib.sha256(f"holder{self._born}".encode()).hexdigest()[:35]
        spent_sol = float(self._rng.lognormal(-0.4, 0.9))            # most buy under 1 SOL, a few much more
        self._crowd[name] = spent_sol / max(self.market.price / self.market.sol_usd, 1e-15)

    def crowd_step(self) -> None:
        """A few made-up buyers arrive each round and now and then one sells out."""
        for _ in range(int(self._rng.poisson(1.6))):
            self._arrive()
        for name in list(self._crowd):
            if self._rng.random() < 0.03:
                del self._crowd[name]

    def sol_balance(self) -> float:
        return self._sol

    def holders(self, mint: str) -> tuple[list, str]:
        return sorted([w, c] for w, c in self._crowd.items()), "paper crowd (made-up wallets)"

    def plain_wallets(self, wallets: list[str]) -> list[str]:
        return list(wallets)

    def claim_fees(self) -> tuple[str, float]:
        got, self._pending_fees = self._pending_fees, 0.0
        self._sol += got
        return self._sig("claim"), got

    def send_sol(self, to: str, sol: float) -> str:
        self._sol -= sol
        return self._sig("reward")

    def create_token(self, token, dev_buy_sol: float) -> tuple[str, str]:
        self._sol -= dev_buy_sol
        return self._sig("create"), "PAPERMINT" + hashlib.sha256(token.symbol.encode()).hexdigest()[:30]
