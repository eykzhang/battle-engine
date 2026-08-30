# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project

battle-engine is an ML/search Pokémon battle engine ("Stockfish for Pokémon") — the
intelligence layer behind the sibling `../battle-brain` iOS app. It plays and analyzes
Pokémon Showdown battles via game-tree search and machine learning, developed in
phased stages (classical search → supervised learning → RL → C++ core as stretch).

**This is explicitly a learning project** — the user has little prior ML experience
and wants to learn it here (and C++ in the stretch phase). Prefer explaining-while-
building over dropping in finished solutions: when introducing an ML concept
(loss functions, state encoding, PPO, etc.), explain what it is and why it's the right
tool before/while using it. Small, understood steps beat big opaque ones.

The full phased roadmap, decisions, and rationale are in
`/Users/edward/.claude/plans/completely-overhaul-the-battle-wondrous-pillow.md` — read
it before making architectural decisions. The iOS/AWS side is planned in
`/Users/edward/.claude/plans/battlebrain-an-happy-summit.md`.

## Status and history

Working history lives in `notes/`, not in this file — see
[[battle-engine/notes/index|notes/index]] for the full map.

Current state, as of 2026-08-30:

- **Phase 0** (harness, baselines) — gate met.
- **Phase 1** (classical search) — gate met. `TwoPlySearchPlayer` beats `MaxBasePowerPlayer`
  84.8% [81.4, 87.7] and `SimpleHeuristicsPlayer` 59.2% [54.8, 63.4]. Stopped with a documented
  gap: the bot never uses non-damaging moves.
- **Phase 2** (supervised) — win-probability and imitation models built. Milestone E gate **not
  met**: a learned eval bolted onto the same 2-ply search plateaued in a 35-40% band, judged a
  structural ceiling rather than a tuning problem.
- **Phase 3** (RL) — **gate met, directly measured, historically** (pre-rewrite encoding). PPO
  self-play beat the Phase-1 search bot 69.6% [65.4, 73.5] and the Phase-2 supervised bot 69.2%
  [65.0, 73.1] on a 665-dim encoding. The 2026-08-27 encoding rewrite (`VECTOR_LEN` 665 -> 2156)
  and same-day team-pool expansion (5 -> 26 teams) invalidated every trained checkpoint; a
  4-day, 9-run retrain attempt (2026-08-27 to 2026-08-30) never reproduced the gate and was
  **abandoned, not continued** — see
  [[battle-engine/notes/decision-phase-3-retrain-abandoned-for-phase-4-focus|the decision note]].
  Not being actively worked; historical result stands, frozen against the old encoding.
- **Phase 4** (C++ core) — **now the project's active focus.** M0-M7 complete as of 2026-08-26
  (MCTS/DUCT search, hand-crafted eval as the default leaf evaluator, PUCT search with the
  pre-rewrite PPO actor as an optional prior — real 60.4% [56.0, 64.6] vs. hand-crafted-eval-only
  search; see `overview.md`'s Measurable Outcomes). New goal as of 2026-08-30: get the engine to
  beat most players, not just this project's own internal bot roster — not yet turned into a
  measured, gated target (population, rating band, real match count all still to be decided).
  Does not depend on Phase 3 — modeled in part on
  [pmariglia/foul-play](https://github.com/pmariglia/foul-play), one of the top-ranked real
  Showdown bots, which uses search over a hand-crafted eval with no RL/self-play at all.

## Working notes

When a session solves something non-obvious, write it down in that session:

- **A transferable lesson** — a bug worth remembering, a design choice with a real rationale —
  gets its own `notes/gotcha-*.md`, `notes/decision-*.md`, or `notes/pattern-*.md`.
- **The session's own record**, with exact numbers and full context, gets one
  `notes/<slug>.md` with `type: log`.
- Add both to `notes/index.md`.

One idea per file. Do not append to a running log — this file was 1,309 lines before that habit
was undone on 2026-08-18, and nothing in it could be found by grep.

Frontmatter schemas and the vault-wide conventions are in the parent `../CLAUDE.md`.

## Hard rules

- **Laptop-first**: every training run must fit on the M4 MacBook Air (measure one
  epoch / 1k battles first, extrapolate, cut scope to keep runs ≤ overnight). Cloud
  GPUs (Colab, AWS spot) are optional accelerators, never dependencies.
- **Phase gates are head-to-head win rates** measured by the benchmark harness
  (500+ battles, report confidence intervals) — don't advance phases on vibes.
- **Integrity**: the engine never provides live move recommendations during ranked
  human play (cheating under Showdown rules). Ladder runs of the bot itself follow
  bot etiquette (alt account, register as bot where required). Analysis is post-hoc.
- **Evidence over assumption** (same rule as battle-brain): verify poke-env/Showdown/
  PyTorch APIs against docs or a small test before relying on them — poke-env's API
  has shifted across versions.

## Commands

```bash
# Start local Showdown server (required for any battles; first run builds)
cd pokemon-showdown && node pokemon-showdown start --no-security

# Smoke test (2 random bots, 3 battles)
.venv/bin/python scripts/smoke_test.py

# Benchmark harness: head-to-head win rate w/ 95% CI (needs the local server running)
# --p1/--p2 choices: random, maxdamage, heuristic, search (Phase-1 bot), learned (Phase-2 model + search)
# Default --format is gen9randombattle; pass --format gen9ou to benchmark "learned"
# on-distribution (the format its training data is actually in) - see
# notes/milestone-e-learned-eval-plateau.md for why that distinction matters.
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500
.venv/bin/python scripts/benchmark.py --p1 learned --p2 search --format gen9ou --n-battles 500

# Tests (integration test auto-skips if the local server isn't running)
.venv/bin/pytest

# Pull a small ELO-filtered sample of Metamon's gen9ou replay dataset (streams the
# 20+GB archive, stops after --n replays; never downloads it whole). Needs the `ml`
# extra installed: .venv/bin/pip install -e ".[ml,dev]"
# --accept-probability thins acceptances to force deeper (temporally wider) archive
# scanning at the cost of more bandwidth - see notes/milestone-e-learned-eval-plateau.md,
# the current 30k-replay dataset needed this to not be 100% one month.
.venv/bin/python scripts/fetch_replay_sample.py --n 200 --min-elo 1200
.venv/bin/python scripts/fetch_replay_sample.py --n 30000 --min-elo 1200 --accept-probability 0.05

# Build the cached (vector, win/loss-label) dataset from fetched replays
.venv/bin/python scripts/build_dataset.py --replay-dir data/replays_raw --out-dir data/dataset

# Train the win-probability MLP on the cached dataset; saves the best-val-loss
# checkpoint (not the final epoch) to data/models/win_prob.pt
.venv/bin/python scripts/train_win_prob.py

# Build the cached (vector, action-label) dataset for the imitation model - same
# replays, same train/val split as build_dataset.py, different label
.venv/bin/python scripts/build_action_dataset.py --replay-dir data/replays_raw --out-dir data/dataset

# Train the imitation (move-prediction) MLP; saves to data/models/imitation.pt
.venv/bin/python scripts/train_imitation.py

# Validate a gen9ou team against the local Showdown checkout's actual current
# ruleset (banlist isn't static - don't trust a team's legality from its source,
# see battle_engine/teams.py's module docstring)
node pokemon-showdown validate-team gen9ou < packed_team.txt

# Phase 3: masked PPO training (sb3-contrib's MaskablePPO) against self-play
# (frozen snapshots of the trainee's own past weights - default) or a fixed
# RandomPlayer (--opponent random, for quick smoke tests). Warm-starts from
# data/models/{imitation,win_prob}.pt by default (--no-warm-start to skip).
# Needs the local server running and stable-baselines3 + sb3-contrib installed
# (both in the `ml` extra). --n-steps controls the rollout length before each
# policy update (small here for a quick feasibility check; SB3's own default is
# 2048); --snapshot-interval/--max-snapshots tune the self-play pool.
# --eval/--checkpoint (both on by default) periodically benchmark the live
# policy against TwoPlySearchPlayer (real games, not a proxy metric) and
# periodically save the model so a long run survives an interruption -
# --resume-from continues from a saved checkpoint instead of starting cold
# (--timesteps then means MORE steps, not a new absolute target).
.venv/bin/python scripts/train_ppo.py --timesteps 2000 --n-steps 256
.venv/bin/python scripts/train_ppo.py --resume-from data/models/checkpoints/ppo_500000_steps.zip --timesteps 500000

# Diagnostic (not a benchmark tool): plays a few real games between a saved
# PPO checkpoint and the search bot, logging each turn's chosen order plus
# HP/protect_counter - the "watch real replays" technique that found the
# protect-spam pathology behind Phase 3's win-rate plateau
# (see notes/pattern-watch-real-replays-not-just-metrics.md).
.venv/bin/python scripts/inspect_ppo_replays.py --n-battles 6

# Plays the trained checkpoint on the REAL Showdown ladder (not the local dev
# server) for a real GXE number - the roadmap's other Phase 3 gate component.
# Needs an already-registered account (an alt, not personal - see the
# script's own docstring for why, and for what Showdown's actual current
# bot/alt policy does and doesn't require). Password via POKE_SHOWDOWN_PASSWORD
# or a secure prompt, never a CLI arg. Real, real-time games against real
# humans - start with a small --n-games.
export POKE_SHOWDOWN_PASSWORD=...
.venv/bin/python scripts/ladder_ppo.py --username my-bot-alt --n-games 10

# Phase 4: build the C++ extension (cpp/) - Debug by default (ASan/UBSan on,
# see cpp/CMakeLists.txt's comment on why this matters for hand-written
# tree/pointer code); --release for an optimized, sanitizer-free build.
# Needs cmake (brew install cmake) and Xcode's clang (C++20) - both verified
# present, see plans/precious-crafting-bachman.md. Output lands directly in
# battle_engine/_native*.so - no reinstall step (PEP 660 editable install).
./scripts/build_cpp.sh

# Run pytest when the native extension is part of what's under test - plain
# `.venv/bin/pytest` will crash on collection (`Fatal Python error: Aborted`,
# no useful traceback) importing a Debug/ASan-built _native*.so without the
# ASan runtime preloaded first. This wrapper sets DYLD_INSERT_LIBRARIES
# before the interpreter starts (can't be done from conftest.py - has to
# happen pre-launch). Confirmed real during M1 bring-up, not a hypothetical.
./scripts/pytest_native.sh

# Run the Catch2 C++ unit test suite once cpp/tests/ has real tests (M4+)
ctest --test-dir cpp/build
```

The venv is `.venv/` (Python 3.13); `pokemon-showdown/` is a gitignored local clone.
`data/` (fetched replay samples, trained models) is gitignored too — regenerate via
the fetch/train scripts above rather than committing it. Add new commands here as the
harness/training scripts land — don't leave this stale.

## Layout

- `battle_engine/` — the package: bots (`search.py`), eval (`evaluation.py`,
  `damage.py`), state encoding (`encoding.py`), dataset building (`dataset.py`),
  models (`win_prob.py`, `imitation.py`), benchmark harness (`benchmark.py`), team
  pool for gen9ou (`teams.py`); Phase 3: action-space translation (`action_space.py`),
  the PPO-facing Gymnasium env (`rl_env.py`), imitation/win-prob weight transplant
  (`ppo_warm_start.py`), self-play (`self_play.py`), periodic real-game eval callback
  + benchmark-facing PPO loader (`ppo_eval.py`); Phase 4: pure-Python MCTS/DUCT
  validation prototype (`mcts_prototype.py`, throwaway, see Status), compiled C++
  extension lands here as `_native*.so` (gitignored)
- `cpp/` — Phase 4's C++ engine (M1 toolchain + M4 `BattleState`/hand-crafted-eval
  port done, see Status; `include/be/`, `src/`, `bindings/module.cpp`, `tests/` per
  plans/precious-crafting-bachman.md's repo layout — user-implemented per this
  project's Phase 4 hard rule, Claude scaffolds headers/stubs/tests only; `build/`
  and `.cache/` gitignored)
- `scripts/` — runnable entry points (smoke test, benchmarks, replay fetching,
  dataset building, training, PPO training, PPO replay diagnosis
  `inspect_ppo_replays.py`, real-ladder play `ladder_ppo.py`, C++ build
  (`build_cpp.sh`) and its ASan-aware pytest wrapper (`pytest_native.sh`))
- `tests/` — pytest (state encoding, damage calc, dataset/action-label logic, model
  training loops, harness determinism w/ seeded RNG, action-space translation, the
  PPO env — including one real-server integration test — PPO warm-start weight
  transplant, self-play, PPO eval/benchmark loading, native-extension bindings)
- `pokemon-showdown/` — local simulator checkout (gitignored)
- `data/` — gitignored: `replays_raw/` (fetched replays), `dataset/` (cached train/val
  arrays for both the win-prob and action-label datasets), `models/` (trained
  checkpoints: `win_prob.pt`, `imitation.pt`, and PPO checkpoints once
  `train_ppo.py --save` is used) — see commands above

## Git workflow

The user commits and pushes themselves — do not run `git commit` or `git push` unless
explicitly asked in the moment. No `Co-Authored-By` trailers, ever.

## Notion sync

Project page: https://app.notion.com/p/3a1fe25f150b8106bfdef912a19dc33f (see parent `../CLAUDE.md` "Notion sync"
for the rule: update Status + "Recent activity" once per session with meaningful
progress).
