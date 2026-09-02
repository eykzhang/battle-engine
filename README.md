# battle-engine

An ML/search battle engine for competitive Pokémon: "Stockfish for Pokémon."

The engine plays and analyzes [Pokémon Showdown](https://pokemonshowdown.com/)
battles, built in stages that mirror how chess engines actually evolved: hand-crafted
evaluation and search first, then a learned evaluation trained on millions of human
replays, then reinforcement learning through self-play, then a compiled C++ search
core. Every stage that shipped is measured and documented below, including the ones
that didn't clear their gate — those results shaped what came next as much as the
wins did.

**The project is complete.** The final architecture is search plus opponent-set
prediction over a complete forward model, with no learned policy in the loop. It beats
most human players on the real competitive ladder. The reinforcement-learning and
from-scratch C++ search tracks were both tried, both measured, and both superseded —
they stay in the repo as recorded work, not as the current architecture. The "Status"
section below is the full account of why.

It's the intelligence layer behind [BattleBrain](https://github.com/eykzhang/battle-brain),
a native iOS app that surfaces the engine's per-turn analysis for replay review, the
same relationship Stockfish has to a chess GUI.

`scripts/analyze_replay.py` is the seam between them. It walks a Showdown replay turn
by turn, searches over sampled opponent teams at each turn, and emits one JSON
document holding the win probability, the engine's ranked candidate actions, and what
the move actually played cost against the best one. The engine needs a compiled gen9
poke-engine build and cannot run on a phone, so the app ships those documents
pre-computed rather than calling a service:

```bash
.venv/bin/python scripts/analyze_replay.py \
  --replay gen9ou-2672927322 --perspective p1 --out analysis.json
```

Two coverage rates come out of it, and they are different questions. **Every** turn of
a real replay translates into a position the engine can evaluate, so a win-probability
curve is continuous. Only about half of them let the search also express the move the
player actually chose, which is what grading requires — the limit there is that replays
never reveal EV spreads or items, the same gap M4's set prediction measures.

## Why Pokémon is a hard AI problem

Chess engines get to assume a lot that Pokémon takes away. Every turn is:

- **Simultaneous** — both players commit moves at once, so there's no simple minimax
  alternation; each turn is a matrix game, not a tree node.
- **Stochastic** — damage rolls, critical hits, secondary effects, and accuracy
  checks mean identical decisions produce different futures, so search has to reason
  over outcome distributions, not lines.
- **Imperfect-information** — the opponent's movesets, items, abilities, and EV
  spreads are hidden until revealed through play.

Any one of these breaks textbook chess techniques. Pokémon has all three at once.

## Approach

Early phases reused the official Showdown simulator as a rules engine via
[poke-env](https://github.com/hsahovic/poke-env), but that simulator only exposes
what's legal, not what every outcome of a move actually is — the final architecture
instead binds [poke-engine](https://github.com/pmariglia/poke-engine), a complete gen-9
forward model that can enumerate every outcome of a move pair with its likelihood, the
component the earlier phases were missing.

Development was staged, and each stage had to beat the previous stage's bot
head-to-head (500+ battles, measured with confidence intervals) before the project
advanced:

| Phase | What | Gate |
|---|---|---|
| 0 | Local Showdown server, poke-env harness, baseline bots, benchmark script | baselines measured |
| 1 | Classical engine: hand-crafted evaluation + lookahead search over damage-calculated outcomes | >70% win rate vs. max-damage over 500+ battles |
| 2 | First ML: learned win-probability model + move-prediction model, trained on millions of parsed human replays | beats the Phase-1 bot head-to-head |
| 3 | Reinforcement learning: PPO self-play, initialized from the Phase-2 policy | beats the Phase-2 bot head-to-head + a real ladder GXE |
| 4 | Stretch: C++ search core (MCTS/DUCT, PUCT with a trained PPO prior/value, compiled inference) | measured, not gated — exploratory by design |
| 5 | Stretch: comprehensive state-encoding rewrite, closing a diagnosed real-ladder gap | correctness harness passing, not a win-rate gate |
| 6 | **Final architecture:** search + opponent set prediction over a complete forward model; RL and the from-scratch C++ search both superseded | GXE above 50% on the real gen9ou ladder — **met, 68.2%** |

## Engineering constraints (on purpose)

- **Laptop-first.** Every training run has to complete on a MacBook Air (M4) in at
  most an overnight run, checked by timing one epoch before committing to a full one.
  Cloud GPUs are an optional accelerator, never a dependency.
- **Strength is the metric.** Loss, accuracy, and calibration are tracked, but no
  phase ships on them, only on head-to-head win rate.
- **Fair play.** The engine never gives live move recommendations to a human during
  ranked play, which Showdown's rules prohibit. Its analysis is post-hoc replay
  review; ladder play by the engine itself uses a registered bot alt.

## Status

### Phases 0–1: classical search

A 2-ply search bot (hand-crafted evaluation + expected-damage lookahead, verified
against Showdown's own simulator source) beat max-damage 84.8% and a scripted
heuristic bot 59.2% over 500 battles each (95% CI, both well past gate). The dominant
bug along the way was a unit mismatch — opponent HP is reported on a 0–100 percent
scale mid-battle, not in absolute points, which had made nearly every attack look like
a guaranteed faint. Fixing it moved the heuristic matchup from 39% to 59%. Stopped with
a documented, deliberate gap: the bot never uses non-damaging moves, since the
evaluation has no way to value them.

### Phase 2: first ML — gate not met

A learned win-probability model and a move-prediction model, both PyTorch MLPs trained
on ~30,000 ELO-filtered human replays via a shared two-adapter encoding pipeline (live
battles and parsed replays converging on one vector). After four rounds of iteration
the learned eval plateaued at 35.6% vs. the Phase-1 bot — read as a structural ceiling,
not a lingering bug: a learned evaluation bolted onto the same shallow 2-ply search
can't see switching's multi-turn value.

### Phase 3: reinforcement learning — gate met, later terminated

PPO self-play, warm-started from the Phase-2 models, sidesteps Phase 2's ceiling by
learning a policy directly instead of ranking states through fixed-depth lookahead.
After a 22M-step run:

- **69.6%** vs. the Phase-1 search bot (95% CI [65.4, 73.5])
- **69.2%** vs. the Phase-2 learned bot — the actual roadmap gate

It also played 45 real games on the live Showdown ladder: 16-29, a 35.6% win rate,
**GXE 26.3%** — a harder and more honest test than bot-vs-bot benchmarks, and the
number that became this project's baseline to beat. The RL track was terminated on
2026-08-30 after a subsequent state-encoding rewrite (Phase 5) invalidated every
trained checkpoint and a nine-run retrain never reproduced the gate; the numbers above
are historical, measured against the pre-rewrite encoder, not the current architecture.

### Phase 4: C++ search core — stretch, complete

Before a multi-week C++ build, a pure-Python prototype tested the actual premise: does
a real, branching search beat the existing 1-ply search over the same evaluation? It
didn't (26.7%, confirmed across a hyperparameter sweep, a 10x simulation-count
increase, and replay inspection). Rather than an implied strength bet, the phase
proceeded as an explicit exploration of the search side of the architecture on its own
terms: a battle-state representation, forward model, and open-loop MCTS/DUCT search in
C++, weight-export tooling, a neural-network forward pass reimplemented in C++
(verified against PyTorch to ~9e-5), and PUCT search using the trained network as a
per-node prior and leaf value:

- PUCT search (PPO prior + critic value) beats the same search using only the
  hand-crafted evaluation, **60.4%** [56.0, 64.6] — the learned prior genuinely helps.
- PUCT search still loses to the raw PPO policy alone, **35.8%** [31.7, 40.1] — search
  wrapped around a frozen policy/value net, never trained with search in the loop,
  doesn't close the gap to running the network directly.

The forward model both searches ran over turned out to be the real limit — see Phase 6.

### Phase 5: state-encoding rewrite — stretch, complete

A real-ladder diagnostic found the actual bottleneck wasn't search depth or model
architecture, it was the state representation: the encoder only ever captured a coarse
per-Pokémon move-type aggregate, with no per-move signal for type effectiveness. The
literal failure mode: one loss used Draco Meteor four turns in a row into a
Fairy-type immune to it. A four-stage rewrite followed (`VECTOR_LEN` 665 → 2156),
adding per-move type-effectiveness/secondary-effect signal, protect/recharge/recoil
handling, weather-conditional move behavior, and side-condition completeness. Closed
with a correctness harness; every existing trained checkpoint was invalidated by the
shape change.

### Phase 6: the final architecture — gate met

Two numbers ended the RL and from-scratch-search tracks and set this phase's direction:
the finished C++ search lost to the far simpler Phase-1 2-ply bot (32.4%), and the
trained PPO policy's real-ladder GXE was 26.3%. The diagnosis: Phase 4 built a sound
search over a forward model that couldn't represent the game — no stat boosts, items,
abilities, residual damage, weather, or Terastallization. A deeper search over an
incomplete model compounds its error at every ply; it should lose to a shallower
search, and it did.

The reference point is [Foul Play](https://github.com/pmariglia/foul-play), which
reaches 80% GXE and top 100 in gen9 OU with no reinforcement learning at all: search
guided by a hand-crafted evaluation, over a *complete* simulator, with unknown
opponent sets sampled from usage statistics. This project already had the search
architecture; it was missing the complete forward model and the opponent modeling.

Phase 6 built both, and the measured chain runs end to end:

- **Forward-model fidelity:** given a real replay's action, the bound gen-9 simulator
  is exactly right on 33.7% of turns and right-or-near-right on 54.1%, over 5,738
  real turns — but only 29.6% of turns are representable from revealed information
  alone, without predicting the opponent's hidden set.
- **Set prediction:** a Smogon usage-statistics prior lifts forward-model
  representability from 30.4% to 66.3% and eliminates a systematic HP-bulk bias.
  Conditioning species predictions on already-revealed teammates alone is worth +13.9
  points of species recall.
- **The player, local benchmark:** K sampled opponent states searched under a shared
  wall-clock budget, aggregated by summed visits. **95.4%** [93.2, 96.9] vs. the
  Phase-1 search bot, **99.0%** [97.7, 99.6] vs. the finished Phase-4 C++ search, each
  over a full 500-battle run.
- **The real ladder, the gate itself:** 50 games on a dedicated account, **34-16
  (68.0%), GXE 68.2%** — clears the 50% bar and more than doubles Phase 3's RL-era
  baseline, with no learned policy anywhere in the pipeline.

Read together: search over a *complete* forward model plus opponent-set prediction
beats both a hand-tuned search over an *incomplete* model and reinforcement learning
over the same incomplete model — measured, not assumed, at every step. The trained RL
checkpoints and the from-scratch C++ search stay in the tree as recorded work; they
are not maintained and are no longer the engine's architecture.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Local Showdown server (cloned into the repo, gitignored)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install

# The gen-9 poke-engine forward model (Rust). DO NOT `pip install poke-engine` -
# the published PyPI wheel is a gen-4 build with no error if used on gen-9 states.
brew install rustup && export PATH="/opt/homebrew/opt/rustup/bin:$PATH" && rustup default stable
git clone --depth 1 --branch v0.0.48 https://github.com/pmariglia/poke-engine.git poke-engine
./scripts/build_poke_engine.sh
```

## Run

```bash
# Terminal 1: start the local simulator
cd pokemon-showdown && node pokemon-showdown start --no-security

# Terminal 2: smoke test
.venv/bin/python scripts/smoke_test.py

# Benchmark two bots head-to-head (95% CI win rate)
# --p1/--p2: random, maxdamage, heuristic, search, learned, ppo, mcts, mcts_puct, setsearch
# --max-concurrent-battles N runs N battles at once - real speedup for lighter bots,
# no effect on setsearch specifically (its Rust search holds Python's GIL)
.venv/bin/python scripts/benchmark.py --p1 setsearch --p2 search --format gen9ou --n-battles 500

# Real Showdown ladder play with the final architecture, needs a registered alt account
.venv/bin/python scripts/ladder_setsearch.py --n-games 10

# Tests (integration tests auto-skip if the local server isn't running)
.venv/bin/pytest
```

## Built on

- [poke-env](https://github.com/hsahovic/poke-env) — Python/Gymnasium interface to Pokémon Showdown
- [poke-engine](https://github.com/pmariglia/poke-engine) — the complete gen-9 forward model the final architecture searches over
- [Foul Play](https://pmariglia.github.io/posts/foul-play/) — the reference architecture: search plus opponent set prediction over poke-engine, no reinforcement learning
- [Metamon](https://github.com/UT-Austin-RPL/metamon) — offline RL baselines and 3.5M+ parsed human replay trajectories, used in the earlier learning phases
