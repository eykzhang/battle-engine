# battle-engine

**An ML/search battle engine for competitive Pokémon — "Stockfish for Pokémon."**

The engine plays and analyzes [Pokémon Showdown](https://pokemonshowdown.com/) battles
using game-tree search and machine learning, built in deliberate stages that mirror
how chess engines actually evolved: hand-crafted evaluation + search first, then a
learned evaluation trained on millions of human replays (the "NNUE moment"), then
reinforcement learning through self-play.

It is the intelligence layer behind
[**BattleBrain**](https://github.com/eykzhang/battle-brain), a native iOS companion
app that surfaces the engine's per-turn win-probability analysis for replay review —
the same relationship Stockfish has to a chess GUI.

## Why Pokémon is a hard AI problem

Chess engines get to assume a lot that Pokémon takes away. Every turn in a Pokémon
battle is:

- **Simultaneous** — both players commit moves at once, so there is no simple
  minimax alternation; reasoning about the opponent's concurrent choice is part of
  every decision (formally, each turn is a matrix game, not a tree node).
- **Stochastic** — damage rolls, critical hits, secondary effects, and accuracy
  checks mean identical decisions produce different futures; search must reason over
  outcome distributions, not lines.
- **Imperfect-information** — the opponent's movesets, items, abilities, and EV
  spreads are hidden and only revealed through play, so the engine must maintain
  beliefs about what it hasn't seen.

Any one of these breaks textbook chess techniques; Pokémon has all three at once.
That combination — and the fact that strong prior work exists to benchmark against —
is what makes it a genuinely interesting engine-building problem rather than a toy.

## Approach

One deliberate scoping decision up front: the engine **reuses the official Pokémon
Showdown simulator** as its rules engine (via a local server and
[poke-env](https://github.com/hsahovic/poke-env)) rather than re-implementing gen-9
battle mechanics — a multi-year project in itself, and the same choice made by
essentially all serious research in this space. Effort goes into the interesting
part: search, evaluation, and learning.

Development is staged, and each stage must **beat the previous stage's bot
head-to-head** (500+ battles, measured by a benchmark harness with confidence
intervals) before the project advances — objective strength gates, not vibes:

| Phase | What | Gate |
|---|---|---|
| **0** | The lab: local Showdown server, poke-env harness, baseline bots (random, max-damage), benchmark script | baselines measured |
| **1** | Classical engine: hand-crafted evaluation function + lookahead search over damage-calculated outcomes | >70% win rate vs max-damage over 500+ battles |
| **2** | First ML: a learned win-probability evaluation and a move-prediction (imitation) model, trained on millions of parsed human replays; the learned eval replaces the hand-crafted one inside the search | beats the Phase-1 bot head-to-head |
| **3** | Reinforcement learning: PPO self-play, initialized from the Phase-2 imitation policy | beats the Phase-2 bot head-to-head |
| **4+** | *Stretch*: a C++ search core — fast forward model + MCTS with an embedded, NNUE-style compiled inference of the learned eval, bound to Python via pybind11; gen9 OU support with opponent-set inference | ladder GXE |

The end-state architecture mirrors Stockfish and Leela Chess Zero exactly: **train in
Python, search and infer in C++.**

Format: `gen9randombattle` first (the standard bot-development ladder), gen9 OU as
stretch.

## Engineering constraints (on purpose)

- **Laptop-first.** Every training run must complete on a MacBook Air (M4) in at most
  an overnight run — measured by timing one epoch / 1k battles and extrapolating
  before committing. Cloud GPUs are an optional accelerator, never a dependency.
  Constraints like this force the parts that actually teach you something: careful
  state encoding, small models, and honest measurement.
- **Strength is the metric.** ML metrics (loss, accuracy, calibration) are tracked,
  but no phase ships on them — only on head-to-head win rate.
- **Fair play.** The engine never provides live move recommendations to a human
  during ranked play (against Showdown's rules). Its analysis is post-hoc replay
  review; ladder runs of the bot itself follow Showdown's bot etiquette.

## Status

**Phase 0 gate met.** Benchmark harness built (Wilson-interval win rates over N
battles); three baselines measured head-to-head over 500 battles each:

| Matchup | Win rate (95% CI) |
|---|---|
| max-damage vs random | 91.8% [89.1, 93.9] |
| heuristic vs random | 99.8% [98.9, 100.0] |
| heuristic vs max-damage | 90.0% [87.1, 92.3] |

**Phase 1 gates met.** `TwoPlySearchPlayer`: 2-ply lookahead (my move, opponent's assumed
best known reply) over a hand-crafted eval and an expected-damage calculator verified
against Showdown's own simulator source.

| Matchup | Win rate (95% CI) | Gate |
|---|---|---|
| search vs max-damage | 84.8% [81.4, 87.7] | ✅ >70% target |
| search vs heuristic | 59.2% [54.8, 63.4] | ✅ clear win |

Getting there took two rounds of evidence-driven debugging. Replay inspection first
showed the bot never switched proactively (fixed with a switch-urgency bonus), but that
barely moved the win rate (~38-39%) — the tell that something bigger was still wrong. An
independent Opus review (before anything was committed) then found the real bug:
`expected_damage` was dividing real damage by the opponent's `max_hp`, but poke-env
reports that as a 0-100 percent scale for the opponent's Pokemon mid-battle, not real HP
— so nearly every attack looked like a near-guaranteed opponent faint, drowning out any
other consideration (including switching). Fixed, verified independently, and confirmed
by rerunning the benchmark: heuristic win rate went 39.0% → 59.2%.

Remaining known gap: the bot still never uses non-damaging moves (status/setup/hazards),
since a 0-damage move can never outrank a damaging one in the current ranking —
`SimpleHeuristicsPlayer` has explicit logic for that. Left for Phase 2, whose learned eval
should pick this signal up from data rather than more hand-crafted logic.

Next: Phase 2 — learned win-probability eval + move-prediction model on human replay
data, swapped into the same search.

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
# Terminal 1: start the local simulator (first run builds; takes a minute)
cd pokemon-showdown && node pokemon-showdown start --no-security

# Terminal 2: smoke test — two random bots play three battles
.venv/bin/python scripts/smoke_test.py

# Benchmark two bots head-to-head with a 95%-confidence win rate
# --p1/--p2 choices: random, maxdamage, heuristic, search (our Phase-1 bot)
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500

# Tests (the harness integration test auto-skips if the server isn't running)
.venv/bin/pytest
```

## Prior art this builds on

- [poke-env](https://github.com/hsahovic/poke-env) — Python/Gymnasium interface to Pokémon Showdown
- [Foul Play](https://pmariglia.github.io/posts/foul-play/) — the strongest classical bot (Rust forward model, MCTS with DUCT for simultaneous moves)
- [Metamon](https://github.com/UT-Austin-RPL/metamon) — offline RL baselines and 3.5M+ parsed human replay trajectories
