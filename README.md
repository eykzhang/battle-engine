# battle-engine

An ML/search battle engine for competitive Pokémon: "Stockfish for Pokémon."

The engine plays and analyzes [Pokémon Showdown](https://pokemonshowdown.com/)
battles using game-tree search, built in stages that mirror how chess engines actually
evolved: hand-crafted evaluation and search first, then a learned evaluation trained on
millions of human replays (the "NNUE moment"), then reinforcement learning through
self-play, then a compiled C++ search core.

The learning phases were built, measured, and are documented below — including the one
that met its gate. They are recorded work rather than the current architecture. As of
2026-08-30 the reinforcement-learning track is **terminated**, and development has
turned to search plus opponent set prediction over a complete forward model, with no
learned policy in the loop. Phase 6 below explains why, and what measurement forced it.

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
| 4 | Stretch, complete: C++ search core (MCTS/DUCT, PUCT with a trained PPO prior/value, compiled inference) | measured, not gated — exploratory by design |
| 5 | Stretch, complete: comprehensive state-encoding rewrite, closing the real-ladder gap | correctness harness passing, not a win-rate gate |
| 6 | Active: search + opponent set prediction over a complete forward model (the Foul Play architecture); RL track terminated | GXE above 50% on the real gen9ou ladder |

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

**Phase 3 gate met, then terminated (2026-08-30).** The result below is real and was
directly measured, but it was measured against the pre-rewrite 665-dim encoder. Phase
5's encoding rewrite invalidated every trained checkpoint, a nine-run retrain never
reproduced the gate, and Phase 6 ended the track rather than continuing it. Read the
numbers as a historical measurement, not as the engine's current strength.

PPO self-play sidesteps Phase 2's ceiling by learning a policy
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

**Phase 4 (stretch: C++ search core) complete.** Before committing to a multi-week
C++ build, a pure-Python MCTS/DUCT prototype tested the actual premise: does a real,
branching search beat the existing 1-ply search over the same hand-crafted
evaluation? It didn't (26.7% vs. the Phase-1 bot, confirmed across a hyperparameter
sweep, a 10x simulation-count increase, and replay inspection that turned up no
bug). Taken honestly, that reframed Phase 4 as a deliberate C++-learning project
rather than an implied strength bet, and it followed through as one: a hand-written
battle-state representation, forward model, and open-loop MCTS/DUCT search in C++,
weight-export tooling for the trained PPO checkpoint, a hand-written neural-network
forward pass (verified against PyTorch to ~9e-5), and PUCT search using that network
as a per-node prior and leaf value — 100+ Catch2 tests, 200+ Python tests, and real
500-battle measurements throughout:

- PUCT search (PPO prior + critic value) beats the hand-crafted-eval search **60.4%**
  [56.0, 64.6] — the neural prior genuinely helps.
- PUCT search still loses to the raw PPO policy alone, **35.8%** [31.7, 40.1] — the
  extra search doesn't yet beat just running the trained network directly.

Read honestly: the engine and the PUCT implementation are both real and correct, but
this configuration — a frozen policy/value net wrapped in search after the fact,
never trained with search in the loop — doesn't close the gap to pure PPO. That
tracks with why AlphaZero-style search needs the network trained on the search's own
output, not handed a frozen one; not attempted here.

**Phase 5 (stretch: state-encoding rewrite) complete.** A real-ladder diagnostic (20
games under an alt account, losses read turn-by-turn) found the actual bottleneck
wasn't search depth or model architecture — it was the state representation.
`battle_engine/encoding.py` used to encode only a coarse per-Pokémon move-type
aggregate, with no per-move signal for type effectiveness or the many other
mechanically relevant interactions a battle turns on. The result was a literal,
repeated failure mode: in one loss, Dragapult's Draco Meteor was used four turns in a
row into Clefable, immune every time — a species the policy trained against
directly, ruling out "never saw this matchup" as the whole story. A four-phase,
prioritized rewrite followed, grounded against Foul Play's own open-source
evaluation logic and this project's C++ movedex/pokedex data: a per-move-slot
feature block (type effectiveness/STAB/secondary effects, `VECTOR_LEN` 665 → 1953),
protect-family/charge/recharge/recoil/drain/self-KO signals (→ 2086),
weather/terrain-conditional move behavior (→ 2120), and side-condition completeness
— hazard stacking, status severity, tera-used (→ 2156, final). A closing
correctness harness confirms the fix: the exact Draco-Meteor-into-Clefable failure
now encodes as 0.0 effectiveness end to end, `dataset.py`/`rl_env.py` need no logic
changes (both consume `VECTOR_LEN`/`encode()` opaquely), and the C++ `encode_native()`
parity suite is honestly marked skipped (not silently broken) pending a future
re-port. **Every existing trained checkpoint (`win_prob.pt`, `imitation.pt`, PPO) is
invalidated by the shape change** — a full replay-dataset-rebuild → retrain pass
(the already-fetched replay sample is fully reusable; only `encode()` changed, not
the fetch/label logic) is the next stretch step, not yet executed.

**Phase 6 (search + set prediction over a complete forward model) in progress; the RL
track is terminated.** Retraining against the new 2156-dim encoder was the planned next
step. It isn't any more, and the reason is a number rather than a preference.

Two measurements framed it. The C++ MCTS/DUCT search loses to the much simpler Phase-1
2-ply bot, 32.4% [28.4, 36.6]. And the best asset the project has ever produced, the
trained PPO policy, went 16-29 on the real ladder for 26.3% GXE. Against real opponents,
every version of this engine has been an underdog.

The diagnosis is that Phase 4 didn't fail to build a search — it built a sound search
over a forward model that can't represent the game. `cpp/src/forward_model.cpp` mentions
status moves exactly once, as an early return of zero damage: no stat boosts, no items,
no abilities, no residual damage, no weather effects, no Terastallization. That was a
deliberate, documented scope cut, but it explains the result better than the original
"branching search doesn't help" reading did. A 2-ply search makes one forward-model
call; a depth-4 search compounds the same model error four times over. Deeper search
over a model that thin *should* lose to shallower search, and it does.

The reference point is [Foul Play](https://github.com/pmariglia/foul-play), which
reaches 80% GXE and top 100 in gen9 OU using no reinforcement learning at all: DUCT
search guided by a hand-crafted evaluation, over the near-complete
[poke-engine](https://github.com/pmariglia/poke-engine) simulator, with unknown opponent
sets sampled from usage statistics. Its author's stated headline lesson is that set
prediction matters as much as or more than the search itself. This project already had
the search — architecturally the same algorithm — and neither of the other two pieces.

So Phase 6 keeps the search shape, binds poke-engine as a complete gen9 forward model,
and adds the opponent modeling. The gate is GXE above 50% on the real ladder: better
than most human players, measured against real ones rather than against this project's
own bots.

The trained checkpoints, the PPO and PUCT code, and the C++ neural-network forward pass
stay in the tree as recorded, measured work. They are not maintained, and the Phase 3
win rates are only quotable against the pre-rewrite encoder they were measured on.

## Current focus

Phase 6 M1–M3: the gen9 poke-engine toolchain (done — with a guard test, because the
published wheel is a gen4 build that fails silently), a Showdown replay corpus with both
sides' actions parsed from the raw protocol, and a backend-pluggable fidelity harness
that measures per-turn divergence for the old and new forward models against the same
real battles.

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
