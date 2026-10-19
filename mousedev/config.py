"""Configuration. Everything that shapes a decision is plain data in config.toml."""

from __future__ import annotations

import json
import os
try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10
    import tomli as tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class BrainCfg:
    connectome: str = "data/connectome.npz"
    dt_ms: float = 0.5
    sim_ms: float = 1000.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest: float = -65.0
    v_th: float = -50.0
    v_reset: float = -65.0
    t_ref_ms: float = 2.0
    delay_ms: float = 1.5
    w_syn: float = 0.25
    inh_gain: float = 1.0
    afferents: int = 30
    w_in: float = 1.0
    bg_hz: float = 800.0
    bg_w: float = 0.5


@dataclass
class RigCfg:
    grid_x: int = 12
    grid_z: int = 8
    slots_x: int = 4                # the screen is split into slots_x x slots_z places, one cheese each
    slots_z: int = 2
    input_types: list[str] = field(default_factory=lambda: ["4P"])
    output_types: list[str] = field(default_factory=lambda: ["5P-ET"])
    inhibitory_types: list[str] = field(default_factory=lambda: ["BC", "MC", "BPC", "NGC"])
    max_in_hz: float = 160.0
    pick_min_frac: float = 0.25     # a sniff only counts above this share of that place's calibrated rate
    # Rate of each place when its cheese is shown alone, in Hz. Calibration overwrites this;
    # it is what a decision record stores.
    slot_ref_hz: list[float] = field(default_factory=lambda: [1.0] * 8)


@dataclass
class LaunchCfg:
    grid_x: int = 6            # a coarser screen for steering: 6 x 4 patches of cortex
    grid_z: int = 4
    fov: float = 1.25          # the mouse sees 1.25x the pad, centred on the cursor
    button_w: int = 420
    button_h: int = 300
    press_frac: float = 0.5    # press above this share of the on-button rate measured in calibration
    press_hz: float = 6.0      # resolved value; calibration overwrites it. 3 steps running = click
    step_ms: float = 200.0
    max_steps: int = 80
    gain: float = 0.7


@dataclass
class RewardCfg:
    round_min_seconds: int = 600    # a round lasts between these two, drawn with the round's seed
    round_max_seconds: int = 900
    table_size: int = 7             # cheeses drawn for each round
    holder_share: float = 1.00      # share of each round's creator fees that goes to the holder (1.0 = all of it)
    min_hold_sol: float = 0.2       # a wallet gets a cheese only while its coins are worth at least this
    reserve_sol: float = 0.05       # never paid out: pays transaction fees
    min_reward_sol: float = 0.001   # smaller than this waits for the next round
    exclude: list[str] = field(default_factory=list)    # wallets that never get a cheese


@dataclass
class ChainCfg:
    dry_run: bool = True
    mint: str = ""
    pool: str = "auto"
    slippage_pct: float = 12.0
    priority_fee: float = 0.0002


@dataclass
class TokenCfg:
    name: str = "mouse dev"
    symbol: str = "MOUSE"
    description: str = "A cubic millimetre of real mouse brain is the dev."
    image: str = "assets/coin.png"
    metadata_uri: str = ""
    twitter: str = ""
    telegram: str = ""
    website: str = ""
    dev_buy_min_sol: float = 0.1   # dev buy if the press is only just above threshold
    dev_buy_sol: float = 0.5       # dev buy if the press is at twice the threshold or more


@dataclass
class PostCfg:
    telegram_chat: str = ""
    x_enabled: bool = False


@dataclass
class Config:
    brain: BrainCfg = field(default_factory=BrainCfg)
    rig: RigCfg = field(default_factory=RigCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    launch: LaunchCfg = field(default_factory=LaunchCfg)
    chain: ChainCfg = field(default_factory=ChainCfg)
    token: TokenCfg = field(default_factory=TokenCfg)
    post: PostCfg = field(default_factory=PostCfg)
    state_dir: str = "state"
    web_dir: str = "web"

    def decision_snapshot(self) -> dict:
        """The parts of the config a decision depends on. Stored in every record."""
        return {"brain": asdict(self.brain), "rig": asdict(self.rig), "reward": asdict(self.reward),
                "launch": asdict(self.launch)}


def _fill(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"config.toml has settings this version does not know, in [{cls.__name__}]: {sorted(unknown)}. "
                         f"If it is from an older version, copy config.example.toml over it and put your own values back.")
    return cls(**data)


def load_env(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load(path: str | None = None) -> Config:
    load_env()
    path = path or os.environ.get("MOUSEDEV_CONFIG", "config.toml")
    raw: dict = {}
    if Path(path).exists():
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "policy" in raw:
        raise SystemExit("config.toml is from an older version (it has a [policy] section). "
                         "Copy config.example.toml over it and put your own values back.")
    cfg = Config(
        brain=_fill(BrainCfg, raw.get("brain", {})),
        rig=_fill(RigCfg, raw.get("rig", {})),
        reward=_fill(RewardCfg, raw.get("reward", {})),
        launch=_fill(LaunchCfg, raw.get("launch", {})),
        chain=_fill(ChainCfg, raw.get("chain", {})),
        token=_fill(TokenCfg, raw.get("token", {})),
        post=_fill(PostCfg, raw.get("post", {})),
        state_dir=raw.get("state_dir", "state"),
        web_dir=raw.get("web_dir", "web"),
    )
    apply_calibration(cfg)
    # `mousedev launch` writes the mint here.
    tok = Path(cfg.state_dir) / "token.json"
    if tok.exists():
        t = json.loads(tok.read_text(encoding="utf-8"))
        cfg.chain.mint = cfg.chain.mint or t.get("mint", "")
        cfg.token.name, cfg.token.symbol = t.get("name", cfg.token.name), t.get("symbol", cfg.token.symbol)
    return cfg


def apply_calibration(cfg: Config, path: str = "data/calibration.json") -> bool:
    """`mousedev calibrate` writes the two global weights and the reference rates here."""
    cal = Path(path)
    if not cal.exists():
        return False
    c = json.loads(cal.read_text(encoding="utf-8"))
    if c.get("version") != 5:
        return False
    cfg.brain.w_syn, cfg.brain.inh_gain = float(c["w_syn"]), float(c["inh_gain"])
    cfg.rig.slot_ref_hz = [float(x) for x in c["slot_ref_hz"]]
    cfg.launch.press_hz = round(cfg.launch.press_frac * c["ref"]["press_hz"], 3)
    cfg.launch.step_ms = float(c.get("launch_step_ms", cfg.launch.step_ms))
    return True


def from_snapshot(snap: dict) -> Config:
    """Rebuild the decision-relevant config from a record (used by replay)."""
    return Config(
        brain=_fill(BrainCfg, snap["brain"]),
        rig=_fill(RigCfg, snap["rig"]),
        reward=_fill(RewardCfg, snap["reward"]),
        launch=_fill(LaunchCfg, snap.get("launch", {})),
    )
