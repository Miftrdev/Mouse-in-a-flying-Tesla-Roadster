# mousedev

**A cubic millimetre of real mouse brain is the dev.**

`mouse-65` is one cubic millimetre of a real mouse's visual cortex, mapped synapse by
synapse by the [MICrONS](https://www.microns-explorer.org/cortical-mm3) project and
simulated here from that public map. It runs a pump.fun coin:

1. Wallets that hold $MIFTR coin can get a cheese. Each round seven
   of them are drawn for the table, one cheese each.
2. A round lasts 10 to 15 minutes. At its end creator fees are claimed and all of them go
   into a pot for holders (`holder_share`, 1.0 by default).
3. The rig shows the cortex each cheese on its own, on a 12 × 8 "screen" fed into layer 4.
   One second of simulated cortex per cheese.
4. The cheese its layer 5 answers most is **eaten**: the whole pot is sent to that wallet.
   If no cheese gets an answer, nothing is sent and the pot waits.
5. The table is cleared and seven wallets are drawn again. No wallet is fed twice until
   every wallet that qualifies has been fed once.
6. It also launches the coin: its layer-5 output steers a cursor onto the LAUNCH button and presses.

No trader, no language model, no trained network. Weights are synapse counts from the data.

## What is the mouse and what is the rig

This is a piece of cortex, not a whole brain. Nobody has mapped a whole mouse brain. It has
no mouth and no hands, so a rig stands between it and the chain. This is the exact split.

| Job | Done by | Where |
|---|---|---|
| Pick which cheese is eaten, and so which wallet is paid | **Mouse** | firing rate of layer-5 ET cells under each cheese, `brain.py` → `rig.py: Rig.pick` |
| Steer the cursor to LAUNCH and press | **Mouse** | `launch.py`, recorded in `state/launch.json` |
| Size of its own first buy | **Mouse**, inside a range you set | harder press = bigger buy, `app.py: Mouse.launch` |
| Which wallets qualify | Rig | holdings read from chain, `rewards.py: eligible` |
| Which wallets are on the table | Rig | a seeded draw anyone can redo, `rewards.py: draw_table` |
| When it eats and how much is sent | Rig | a seeded length between `round_min_seconds` and `round_max_seconds`; `holder_share` of that round's fees, `rewards.py: round_length`, `decide` |
| Name, ticker, picture of the coin | Rig | typed by a person in `config.toml` |
| Sign and send transactions | Rig | `chain.py`, with the mouse's one wallet |
| Write the posts | Rig | `poster.py`, fill-in templates |

Whoever runs the rig holds the wallet key and could move the funds by hand. The code cannot
prevent that; the chain shows it if it happens. Publish the wallet address.

## Check it yourself

Every round is written to `state/decisions/NNNNNN.json` **before** anything is sent,
with the wallets that qualified, the cheeses on the table, its seed (previous record hash + a recent blockhash), the config, and the
SHA-256 of the brain file. Records are hash-chained. A running mouse also serves every
record on its website under `/decisions/` (one file per round), so anyone can fetch one
and replay it.

```bash
mousedev replay state/decisions/000068.json   # re-simulates; same sniffs, pick and table, or it says why not
mousedev verify                               # checks the hash chain
mousedev launch --shuffled                    # control: same cells, wiring scrambled
mousedev calibrate                            # real vs shuffled wiring, for each inhibition strength
```

The shuffled control is the point. If scrambling the wiring changes nothing, the connectome
is not doing the work. Look at the numbers before you believe anyone, including this README.

## Run it

```bash
pip install -e ".[fetch,dev]"
cp config.example.toml config.toml && cp .env.example .env
mousedev start
```

`mousedev start` does everything in order:

1. **Downloads the brain** if `data/connectome.npz` is missing (public MICrONS data, ~3 GB).
   The brain is not in this repo.
2. **Calibrates** on your brain file: finds the two global weights and the reference rates,
   and prints real wiring vs shuffled wiring side by side. This takes a while.
3. **Asks you** for the coin's name, ticker and picture. The mouse does not choose these.
4. **Launches**: the cortex steers the cursor onto LAUNCH and presses. How hard it presses
   sets its own first buy, between `dev_buy_min_sol` and `dev_buy_sol`.
5. **Runs**: claim fees, sniff the cheeses, feed one holder, post. Every 10 to 15 minutes.

It starts in paper mode: nothing is signed or sent, and the holders are a made-up crowd.
To go live, fill `.env` (`MOUSEDEV_PRIVATE_KEY`, `MOUSEDEV_RPC_URL`, `TELEGRAM_BOT_TOKEN`)
and set `[chain] dry_run = false` in `config.toml`.

The full holder list needs an RPC endpoint that allows `getProgramAccounts`. Some
load-balanced endpoints answer it with an empty list now and then; the bot asks again, then
asks the next URL. `MOUSEDEV_RPC_URL` can hold several URLs separated by commas: the first is
used for everything, the others take over when it fails or gives no holder list. If no URL
gives the list, the bot uses the 20 largest token accounts instead and says so in every record.

The steps also exist one by one: `python scripts/fetch_microns.py`, `mousedev check`,
`mousedev calibrate`, `mousedev launch`, `mousedev run` (`--ticks 30 --fast` for a quick
paper run).

Open `rig/launchpad.html` (served from the repo root) to replay the launch run for a video.

The website is a separate project. If its folder is at `web/` (or wherever `web_dir` in
`config.toml` points), the bot writes `state.json` there for it to read.

Live mode refuses to start on the synthetic test fixture or on a shuffled brain.
Paper mode never posts to Telegram or X; it only writes `state/posts.log`, marked `[PAPER]`.

## The model

Leaky integrate-and-fire, the same family as the fly brain model of Shiu et al. (2024):

```
dv/dt = (v_rest − v + g) / τ_m        dg/dt = −g / τ_syn
weight(i→j) = synapses(i→j) × w_syn    (negative if i is an inhibitory cell type)
```

- **Input:** the screen is split into 4 × 2 places. A cheese lights its place: Poisson
  input to the layer-4 cells (`4P`) of those columns.
- **One sniff per cheese.** Each cheese is shown alone, for one second of simulated cortex.
- **Output:** mean rate of layer-5 extratelencephalic cells (`5P-ET`) under that place.
  Liking = that rate ÷ the rate the same place gave in calibration, because some patches of
  this cortex answer louder than others whatever they are shown. The best-liked cheese is
  picked, if its liking is above `pick_min_frac`.
- **Two global weights, set by `mousedev calibrate`:** `w_syn` (mV per synapse, the same for
  every synapse) and `inh_gain` (how much stronger an inhibitory synapse is). The rule: raise
  `w_syn` until layer-5 activity stops staying under a cheese and spreads over the whole
  sheet. Past that point a recurrent network ignites and its output stops depending on the
  input. A few settings below that edge are tried, and the one that is strong, local, works
  in every place, repeats across seeds and beats shuffled wiring is kept. Every setting
  tried is printed and saved in `data/calibration.json`.

The rules around the pick (`rewards.py`, numbers in `config.toml`):

```
qualifies   coins worth ≥ min_hold_sol at the current price; not the dev wallet,
            not an address owned by a program (bonding curve, pools)
pot         += holder_share × creator fees claimed this round
reward      = the whole pot, to the wallet whose cheese was eaten
table       cleared after every round; table_size wallets are drawn with the round's
            seed from those that qualify and have not been fed in this cycle
next round  between round_min_seconds and round_max_seconds, drawn with the same seed
```

A wallet whose coins fall below `min_hold_sol`, by selling or by the price falling, loses
its cheese at the next round. A round with
nothing eaten carries its pot forward. After launch the code never buys, sells or burns.

## Reading the code (for people and for AI reviewers)

Each claim, and the function that makes it true:

| Claim | Look at |
|---|---|
| Weights are synapse counts from the data, nothing trained | `connectome.py: Connectome.matrix` |
| The simulation has no model call and no learning rule | `brain.py: simulate` (the whole file) |
| The pick comes from layer-5 firing | `ledger.py: think` → `rig.py: Rig.pick` |
| Nobody chooses who is on the table | `rewards.py: draw_table`, seed from `ledger.py: make_seed` |
| No wallet is fed twice in a cycle | `rewards.py: draw_table`, `tests/test_mouse.py` |
| The reward is the pot and nothing else | `rewards.py: decide`, `app.py: Mouse.tick` |
| The record is written before any transaction is sent | `app.py: Mouse.tick` |
| A round can be re-run exactly | `ledger.py: replay` |
| A reward is a plain SOL transfer | `chain.py: build_transfer_tx` |
| The key signs locally and is never sent anywhere | `chain.py: LiveChain._pumpportal`, `_send` |
| Launch is steered by layer-5 activity | `launch.py: run` |
| Scrambled wiring fails | `connectome.py: shuffled`, `tests/test_mouse.py` |
| No network calls except RPC, PumpPortal, Pinata, price feeds, Telegram/X | `chain.py`, `market.py`, `poster.py` |

What a careful reviewer will also tell you, correctly: the picture-in / pick-out mapping in
`rig.py` is a human design choice; when and how much are rules, not neurons; and whoever
holds the wallet key can act outside this code.



## Layout

```
mousedev/     connectome.py brain.py rig.py rewards.py ledger.py  the mouse and its rules
              chain.py market.py poster.py launch.py app.py cli.py the rig around it
scripts/      fetch_microns.py   make_synthetic.py (tests only)
rig/          launchpad.html     replay of the launch run
tests/
```

## Data and credit

Connectome: MICrONS Consortium et al., "Functional connectomics spanning multiple areas of
mouse visual cortex", *Nature* 640, 435–447 (2025). Follow their terms and citation policy.
Model: Shiu et al., "A Drosophila computational brain model reveals sensorimotor
processing", *Nature* 634, 210–219 (2024). Code: MIT.

This is a memecoin experiment. It promises no value. Burns reduce supply; they do not set a price.
