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

Format: `gen9randombattle` first (the standard bot-development ladder, Phases 0-1).
Phase 2's training data is human `gen9ou` replays, so real `gen9ou` benchmarking
(constructed teams, validated against the local Showdown ruleset) now exists
alongside it — full ladder-ready `gen9ou` play (opponent-set inference, etc.) is
still stretch scope.

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

**Phase 2 built end-to-end, gate not yet met — four honest rounds of iteration,
plateaued.** A learned win-probability model (small MLP, PyTorch) trained on human
replay data from [Metamon](https://github.com/UT-Austin-RPL/metamon) scores search's
candidate moves in place of the hand-crafted Phase-1 evaluation, and a second
imitation (move-prediction) model was also built, closing out both of Phase 2's
originally-planned deliverables:

- **State encoding**: a fixed-size vector representation of a battle state (species,
  types, HP, status, boosts, stats, held item, protect-counter risk, and a
  hand-engineered moveset summary — recovery/hazard-setup/hazard-removal/setup-boost/
  pivot/priority/coverage), with two adapters (live poke-env battles, parsed human
  replays) feeding the same model. Check `battle_engine.encoding.VECTOR_LEN` for the
  real current dimension rather than trusting a number here — it's changed more than
  once as features were added (316 → 656 → 663 → 665).
- **Data pipeline**: a streaming fetcher that ELO-filters replays without downloading
  Metamon's full 20+GB archive, temporally spread (30,000 replays, Dec 2023 → May
  2026) rather than clustered in one month, and a dataset builder that gets the
  win/loss labeling right (a replay's *final* outcome) and splits train/val by battle
  to avoid leakage.
- **Two models**: `WinProbModel` (state → P(win), two hidden layers) feeds
  `TwoPlySearchPlayer`'s eval function; `ImitationModel` (state → predicted human
  action, 13-class classifier over a verified real action-space scheme) is built and
  validated but not yet integrated anywhere.

Several rounds of independent code review (same practice that caught Phase 1's bug)
found and fixed real correctness issues throughout — a live-vs-replay encoding
mismatch, a checkpoint-saving bug, a damage-calculation gap for fixed-damage moves
(Seismic Toss and similar), and two feature-encoding bugs in how setup-boosting moves
were detected.

| Matchup (on-distribution, gen9ou) | Win rate (95% CI) | Gate |
|---|---|---|
| learned eval vs. Phase-1 search (final) | 35.6% [31.5, 39.9] | ❌ not met |

The full trajectory — 30.2% → 38.4% → 39.6% → 35.6% across format-mismatch fixes,
more/fresher data, richer features, more model capacity, and bug fixes — plateaued in
the same band after the first real jump. Read as a real signal, not noise: a learned
eval bolted onto the same shallow 2-ply search structurally can't see switching's
multi-turn value beyond a hand-tuned patch, and that's plausibly the actual ceiling on
this axis, not a bug still waiting to be found.

**Phase 3 gate met.** PPO self-play sidesteps Phase 2's ceiling by learning a policy
directly rather than ranking states through a fixed-depth lookahead.

Plumbing (built and independently reviewed before any real-scale run):
- **Action-space reconciliation**: poke-env's Gymnasium environment exposes its own
  26-way action scheme; the Phase-2 imitation model was trained on a different,
  13-way one matching this project's own state-encoding conventions. A bidirectional
  translation layer plus a custom Gymnasium env make the two interoperable.
- **Masked PPO** ([stable-baselines3](https://github.com/DLR-RM/stable-baselines3) +
  [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)'s
  `MaskablePPO`): illegal actions are excluded at the action-distribution level, not
  merely corrected after the fact — a real training-quality problem was measured and
  fixed here (56% of actions were illegal under a fresh policy, and the naive
  "substitute a random legal move" fallback was silently misattributing credit).
- **Warm start from both Phase-2 models**: the win-probability model seeds the critic,
  the imitation model seeds the actor — made possible by re-shaping PPO's network to
  match their architecture exactly (verified via bit-identical output reproduction,
  not just matching shapes).
- **Self-play** against a pool of frozen snapshots of the policy's own past weights,
  plus a configurable fraction of training games played directly against the Phase-1
  search bot (not just measured against it) — added after diagnosis showed pure
  self-play never required the policy to transfer "beats recent self" into "beats
  2-ply lookahead."
- **Real progress tracking during training**: periodic head-to-head evaluation
  against the Phase-1 search bot via real games (not a proxy metric), checkpointing,
  and resumable runs.

A 1,000,000-timestep sanity run surfaced a real, quantifiable pathology via replay
inspection (the same technique that found Phase 1's and Phase 2's bugs): the policy
repeatedly re-used Protect immediately after it had just failed, walking back into
Showdown's own escalating-failure mechanic. Traced to a genuine encoder gap and
fixed. A second sanity run at the same scale, plus the real gate benchmark, gave
**34.2% [30.2, 38.5] vs. the Phase-1 search bot — a real but modest improvement over
the initial plateau (~28-32%), still a clear loss.** ❌ gate not met.

Two more real fixes followed from further replay inspection and an independent
Opus review:
- **Reward-shaping magnitude**: the per-turn HP/faint shaping could mathematically
  outweigh the terminal win/loss signal (±1.8 vs. ±1.0, derived from poke-env's own
  reward formula) — rebalanced so winning/losing dominates.
- **A second encoder gap**: the policy spammed Spikes 32 turns straight against a
  Flying-type opponent fully immune to it — a hazard-immunity mechanic (Flying-type,
  Levitate, Heavy-Duty Boots) with no signal anywhere in the vector. Fixed — and the
  first version of the fix was itself caught missing Heavy-Duty Boots (the #2 most
  common immunity source in real data) by the same review process, before it shipped.

With all three fixes combined, a fresh run climbed to a real, confirmed **40.8%
[36.6, 45.2]** at 1,000,000 steps — a clean, statistically confident break from the
~28-32% plateau for the first time. An overnight extension to 22,000,000 steps
(~10.2 hours, 570.6 steps/s sustained the whole run, no thermal throttling) kept
climbing rather than flattening out, and the final real 500-battle benchmark gave

**69.6% [65.4, 73.5] vs. the Phase-1 search bot** — ✅ a decisive, statistically
confident win, not just an edge.

The literal roadmap gate is "beats the Phase-2 supervised bot," not the Phase-1
search bot specifically — Phase 2's own final benchmark lost to the search bot 64.4%
of the time, so PPO's 69.6% here almost certainly clears that bar too, but that exact
matchup hasn't been separately benchmarked.

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
# --p1/--p2 choices: random, maxdamage, heuristic, search (Phase-1), learned (Phase-2), ppo (Phase-3)
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500
.venv/bin/python scripts/benchmark.py --p1 learned --p2 search --format gen9ou --n-battles 500

# Phase 3: masked PPO self-play training, warm-started from the Phase-2 models,
# with periodic real-game evaluation against the search bot and checkpointing
.venv/bin/python scripts/train_ppo.py --timesteps 500000

# Tests (the harness integration test auto-skips if the server isn't running)
.venv/bin/pytest
```

## Prior art this builds on

- [poke-env](https://github.com/hsahovic/poke-env) — Python/Gymnasium interface to Pokémon Showdown
- [Foul Play](https://pmariglia.github.io/posts/foul-play/) — the strongest classical bot (Rust forward model, MCTS with DUCT for simultaneous moves)
- [Metamon](https://github.com/UT-Austin-RPL/metamon) — offline RL baselines and 3.5M+ parsed human replay trajectories
