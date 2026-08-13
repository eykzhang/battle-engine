# battle-engine

An ML/search battle engine for competitive Pokémon: "Stockfish for Pokémon."

The engine plays and analyzes [Pokémon Showdown](https://pokemonshowdown.com/)
battles using game-tree search and machine learning, built in stages that mirror how
chess engines actually evolved: hand-crafted evaluation and search first, then a
learned evaluation trained on millions of human replays (the "NNUE moment"), then
reinforcement learning through self-play.

It's the intelligence layer behind [BattleBrain](https://github.com/eykzhang/battle-brain),
a native iOS app that surfaces the engine's per-turn win-probability analysis for
replay review, the same relationship Stockfish has to a chess GUI.

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

The engine reuses the official Pokémon Showdown simulator as its rules engine, via a
local server and [poke-env](https://github.com/hsahovic/poke-env), rather than
re-implementing gen-9 battle mechanics from scratch. Effort goes into the interesting
part: search, evaluation, and learning.

Development is staged, and each stage has to beat the previous stage's bot
head-to-head (500+ battles, measured with confidence intervals) before the project
advances:

| Phase | What | Gate |
|---|---|---|
| 0 | Local Showdown server, poke-env harness, baseline bots, benchmark script | baselines measured |
| 1 | Classical engine: hand-crafted evaluation + lookahead search over damage-calculated outcomes | >70% win rate vs. max-damage over 500+ battles |
| 2 | First ML: learned win-probability model + move-prediction model, trained on millions of parsed human replays | beats the Phase-1 bot head-to-head |
| 3 | Reinforcement learning: PPO self-play, initialized from the Phase-2 policy | beats the Phase-2 bot head-to-head + a real ladder GXE |
| 4+ | Stretch, underway: C++ search core (MCTS/DUCT + compiled inference), full gen9 OU support | exploratory |

The end-state architecture mirrors Stockfish and Leela Chess Zero: train in Python,
search and infer in C++.

## Engineering constraints (on purpose)

- **Laptop-first.** Every training run has to complete on a MacBook Air (M4) in at
  most an overnight run, checked by timing one epoch before committing to a full one.
  Cloud GPUs are an optional accelerator, never a dependency.
- **Strength is the metric.** Loss, accuracy, and calibration are tracked, but no
  phase ships on them, only on head-to-head win rate.
- **Fair play.** The engine never gives live move recommendations to a human during
  ranked play, which Showdown's rules prohibit. Its analysis is post-hoc replay
  review.

## Status

**Phase 0 and 1 gates met.** A 2-ply search bot (hand-crafted eval + expected-damage
lookahead, verified against Showdown's own simulator source) beats max-damage 84.8%
and a scripted heuristic bot 59.2%, over 500 battles each (95% CI, both well past the
gate). Getting there involved a real debugging arc: replay inspection first showed
the bot never switched proactively, which barely moved the win rate, and a follow-up
code review found the actual bug, an HP-percentage/absolute-HP unit mismatch that
made nearly every attack look like a guaranteed opponent faint. Fixing it took the
heuristic matchup from 39% to 59%.

**Phase 2 built end-to-end; win-rate gate not met.** A learned win-probability model
(PyTorch MLP) trained on ~30,000 ELO-filtered human replays from
[Metamon](https://github.com/UT-Austin-RPL/metamon), plus a separate move-prediction
model, both built on a shared state-encoding pipeline with two adapters (live
battles, parsed replays). Several rounds of code review caught real bugs along the
way (a live/replay encoding mismatch, a fixed-damage-move gap, feature bugs in
setup-move detection). After four rounds of iteration, the learned eval plateaued at
35.6% vs. the Phase-1 bot, short of the gate. Read as a real ceiling rather than a
lingering bug: a learned evaluation bolted onto the same shallow 2-ply search
structurally can't see switching's multi-turn value.

**Phase 3 gate met.** PPO self-play sidesteps Phase 2's ceiling by learning a policy
directly instead of ranking states through fixed-depth lookahead. Built and
independently reviewed: an action-space translation layer reconciling poke-env's
scheme with the project's own, masked PPO (illegal actions excluded at the
distribution level, after review found 56% of a fresh policy's actions were illegal
and being silently misattributed), and warm-starting the actor/critic from the
Phase-2 models. Replay inspection surfaced two real encoder gaps during
training, including the policy spamming Spikes for 32 turns against a Flying-type
opponent immune to it, both fixed. After those fixes, a 22M-step run gave:

- **69.6%** vs. the Phase-1 search bot (95% CI [65.4, 73.5])
- **69.2%** vs. the Phase-2 learned bot (the actual roadmap gate)

It also played 45 real games on the live Showdown ladder under an alt account: 16-29,
a 35.6% win rate, GXE 26.3%. An honest result, not a cherry-picked one: the trained
policy currently loses more than it wins against the live gen9ou ladder population,
a harder and more meaningful test than the local bot-vs-bot benchmarks above.

**Phase 4 (stretch: C++ search core) underway.** Before committing to a multi-week
C++ build, a pure-Python MCTS/DUCT prototype tested the actual premise: does a real,
branching search beat the existing 1-ply search over the same hand-crafted
evaluation? It didn't (26.7% vs. the Phase-1 bot, confirmed across a hyperparameter
sweep, a 10x simulation-count increase, and replay inspection that turned up no
bug). Taken honestly, that reframes Phase 4 as a deliberate C++-learning project
rather than an implied strength bet, and it's proceeding as one: the C++ toolchain
(CMake, pybind11, Catch2, ASan/UBSan-enabled debug builds) is set up, and the
battle-state representation and hand-crafted evaluation function are ported to C++
and passing a 32-test Catch2 suite alongside the existing 162 Python tests. Writing
the actual MCTS/DUCT tree search is next.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Local Showdown server (cloned into the repo, gitignored)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install
```

## Run

```bash
# Terminal 1: start the local simulator
cd pokemon-showdown && node pokemon-showdown start --no-security

# Terminal 2: smoke test
.venv/bin/python scripts/smoke_test.py

# Benchmark two bots head-to-head (95% CI win rate)
# --p1/--p2: random, maxdamage, heuristic, search (Phase 1), learned (Phase 2), ppo (Phase 3)
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500

# Phase 3: masked PPO self-play training, warm-started from the Phase-2 models
.venv/bin/python scripts/train_ppo.py --timesteps 500000

# Real Showdown ladder play, needs a registered alt account
.venv/bin/python scripts/ladder_ppo.py --n-games 40 --max-concurrent-battles 4

# Tests (integration test auto-skips if the local server isn't running)
.venv/bin/pytest
```

## Built on

- [poke-env](https://github.com/hsahovic/poke-env) — Python/Gymnasium interface to Pokémon Showdown
- [Foul Play](https://pmariglia.github.io/posts/foul-play/) — the strongest classical bot (Rust forward model, MCTS with DUCT for simultaneous moves)
- [Metamon](https://github.com/UT-Austin-RPL/metamon) — offline RL baselines and 3.5M+ parsed human replay trajectories
