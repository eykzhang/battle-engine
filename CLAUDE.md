# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project

battle-engine is a search-based Pokémon battle engine ("Stockfish for Pokémon") — the
intelligence layer behind the sibling `../battle-brain` iOS app. It plays and analyzes
Pokémon Showdown battles by game-tree search over a battle simulator, guided by a
hand-crafted evaluation function.

It got here through phased stages (classical search → supervised learning → RL → C++
core), and the machine-learning phases are recorded work rather than the live
architecture: **the reinforcement-learning track was terminated on 2026-08-30** in
favor of the Foul Play architecture — search plus set prediction over a complete
forward model, no learned policy. See the Status section for what that means for the
existing checkpoints and code.

**This is explicitly a learning project** — the user had little prior ML experience
and wanted to learn it here (and C++ in the later phases). Prefer explaining-while-
building over dropping in finished solutions: when introducing a concept (search,
state representation, set prediction, PPO in the historical code), explain what it is
and why it's the right tool before/while using it. Small, understood steps beat big
opaque ones.

Plans, in roadmap order (read the active one before making architectural decisions):

- Phase 4, the C++ search core: `/Users/edward/.claude/plans/precious-crafting-bachman.md`
- Phase 4 close-out (M7 + enhancement track): `/Users/edward/.claude/plans/zesty-zooming-wozniak.md`
- **Phase 6, active** — the Foul Play architecture:
  `/Users/edward/.claude/plans/phase-6-foul-play-architecture.md`

(The original roadmap plan file no longer exists on disk; `notes/index.md` plus the
Status section below are the surviving record of phases 0-3.)

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
- **Phase 3** (RL) — **terminated 2026-08-30.** The gate was met historically and directly
  measured against the pre-rewrite 665-dim encoding: PPO self-play beat the Phase-1 search bot
  69.6% [65.4, 73.5] and the Phase-2 supervised bot 69.2% [65.0, 73.1]. The 2026-08-27 encoding
  rewrite (`VECTOR_LEN` 665 -> 2156) and same-day team-pool expansion (5 -> 26 teams)
  invalidated every trained checkpoint; a 4-day, 9-run retrain attempt never reproduced the gate
  and was abandoned — see
  [[battle-engine/notes/decision-phase-3-retrain-abandoned-for-phase-4-focus|the decision note]].
  Phase 6 makes that permanent rather than paused: **no learned policy is on the critical path
  any more.** The PPO/PUCT code, `cpp/src/mlp.cpp`, and the trained checkpoints stay in the tree
  as recorded work — not maintained, not retrained, and **not to be described as the engine's
  current architecture.** The 69.6%/69.2% numbers remain quotable only with their
  "against the old encoding, historical" qualifier attached.
- **Phase 4** (C++ core) — complete as of 2026-08-26. M0-M7: MCTS/DUCT search, hand-crafted eval
  as the default leaf evaluator, PUCT search with the pre-rewrite PPO actor as an optional prior
  (real 60.4% [56.0, 64.6] vs. hand-crafted-eval-only search). The search is sound; the honest
  result is that it **loses to the Phase-1 2-ply bot, 32.4% [28.4, 36.6]** — see
  [[battle-engine/notes/phase-4-m7-and-enhancement-track|the close-out note]] for the full
  inventory.
- **Phase 5** (state-encoding rewrite) — complete. `VECTOR_LEN` 665 -> 2156, per-move-slot
  feature blocks, correctness harness passing. Numbered as Phase 5 in `README.md`'s public phase
  table; note that some existing notes use "Phase 5" for plan-local milestone numbering inside
  the Phase 4 plan instead.
- **Phase 6** (the Foul Play architecture) — **now the project's active focus**, started
  2026-08-30. Diagnosis: Phase 4 did not fail to build a search, it built a good search over a
  forward model that cannot represent the game (`forward_model.cpp` mentions `Status` exactly
  once, an early `return 0.0f` — no boosts, items, abilities, residuals, weather effects, or
  Tera). Phase 6 binds [poke-engine](https://github.com/pmariglia/poke-engine) as a complete
  gen9 forward model, adds usage-stats-based opponent set prediction, and searches under a
  wall-clock budget with root parallelism — the architecture
  [pmariglia/foul-play](https://github.com/pmariglia/foul-play) uses to reach 80% GXE and top
  100 in gen9 OU with no RL at all. **Gate: GXE above 50% on the real gen9ou ladder**, against
  Phase 3's measured 26.3% baseline. Full plan:
  `/Users/edward/.claude/plans/phase-6-foul-play-architecture.md`.

  Milestone state as of 2026-08-30:

  - **M1** (gen9 toolchain) — built and verified; `scripts/build_poke_engine.sh` plus the
    gen9 guard test.
  - **M2** (replay corpus + protocol parser) — done. 300 rating-filtered gen9ou replays in
    `data/replays_showdown/`, `battle_engine/replay_log.py`.
  - **M3** (translation layer + fidelity harness) — done, with one scope cut recorded in
    [[battle-engine/notes/phase-6-m3-fidelity-harness|the log note]]: only the poke-engine
    backend is scored, not our own `forward_model.cpp`, because the integration decision it
    fed was already made on throughput grounds. Measured over 5,738 turns of real gen9ou:
    **only 29.6% of turns are representable from revealed information alone** — on the other
    70.4%, a search cannot even express the move that was actually played. Given the action,
    poke-engine is exactly right on 33.7% of turns and right-or-near-right (only error an HP
    figure within 10%) on **54.1%**. Residual error is dominated by damage numbers, and
    supplying every ability the battle eventually reveals barely moves them — the gap is EV
    spreads and items, which no replay ever shows.
  - **M4** (set prediction) — next, and now scoped against measured numbers rather than
    Foul Play's assertion. The M3 result says it must predict **spreads**, not just species,
    items and abilities: a filler that supplies only the latter leaves most of the damage
    error on the table. Note also that a battle eventually reveals just 40.5% of abilities and
    26.8% of items, so in-battle inference cannot be the primary source — usage statistics are.
  - **M5** (the player) and **M6** (real-ladder GXE) — not started. M5 is blocked on M4 more
    strictly than the plan assumed: an unrevealed slot's placeholder species id is `none`,
    which is also poke-engine's "do nothing" action string, so an unrevealed slot cannot be
    switched into at all
    ([[battle-engine/notes/gotcha-poke-engine-addresses-actions-by-name-not-index|note]]).

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

# Phase 6: build the gen9 poke-engine Python bindings. DO NOT `pip install
# poke-engine` - the published wheel is a gen4 build (generation is a
# compile-time Cargo feature) and would simulate gen9ou states under gen4
# mechanics with no error at all. See
# notes/gotcha-poke-engine-pypi-wheel-is-gen4-not-gen9.md for the full finding,
# the three checks that confirm a build really is gen9, and the three
# silent-failure footguns in its Python API.
# cargo is NOT on the default PATH - brew's rustup formula is keg-only.
brew install rustup && export PATH="/opt/homebrew/opt/rustup/bin:$PATH" && rustup default stable
git clone --depth 1 --branch v0.0.48 https://github.com/pmariglia/poke-engine.git poke-engine
./scripts/build_poke_engine.sh

# Phase 6 M2: fetch real gen9ou replays from Showdown's own replay API, into
# data/replays_showdown/ (gitignored). Unlike the Metamon corpus in
# data/replays_raw/, these carry BOTH players' actions per turn, which is what
# M3's fidelity harness needs. Resumable and idempotent - files already on disk
# count toward --n and are never re-downloaded. Rate-limited by default (--delay);
# be polite, this is a free public API.
.venv/bin/python scripts/fetch_showdown_replays.py --n 50 --min-rating 1300

# Parse those replays into (state_before, p1_action, p2_action, state_after)
# turn transitions. Note the module's central rule: anything not observable from
# the log is the UNKNOWN sentinel, and bool(UNKNOWN) raises rather than being
# falsy - see notes/decision-unknown-is-a-sentinel-that-refuses-to-be-falsy.md.
.venv/bin/python -c "from battle_engine.replay_log import parse_replay_file; \
  r = parse_replay_file('data/replays_showdown/<id>.json'); print(len(r.transitions))"

# Phase 6 M3: score the forward model against the replay corpus. Drives a real
# poke-env Battle from each replay log, stops at every turn boundary, asks the
# model what both players' actual actions will do, and diffs its answer against
# the log. Reads cached JSON off disk - no Showdown server, no network.
# Runs two conditions by default and prints the delta between them: the
# "action-oracle" supplies only the move/switch/Tera the turn needs, the
# "hindsight-oracle" also supplies every ability and move the battle will
# eventually reveal. That delta is what set prediction is worth, and it is the
# number M4 is scoped against. Full results and method:
# notes/phase-6-m3-fidelity-harness.md.
.venv/bin/python scripts/fidelity_harness.py
.venv/bin/python scripts/fidelity_harness.py --limit 50 --condition action --json data/fidelity.json
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
  + benchmark-facing PPO loader (`ppo_eval.py`); Phase 6: Showdown replay-log parser
  (`replay_log.py` — protocol log to turn transitions, with an explicit UNKNOWN
  sentinel for anything the log does not observe) and the poke-env -> poke-engine
  state translator (`poke_engine_state.py` — validates every species/move/item id
  against the built extension's own vocabulary, and returns a provenance ledger
  separating what was observed from what an injectable `UnknownFiller` assumed;
  M4's set prediction plugs into that seam without touching the module), and the
  forward-model fidelity harness (`fidelity.py` — replay log -> poke-env
  `Battle` -> poke-engine `State`, one turn simulated, diffed against what the
  log says happened; `ForwardModelBackend` is a real seam but only the
  poke-engine backend is implemented, see the module docstring for why);
  Phase 4:
  pure-Python MCTS/DUCT
  validation prototype (`mcts_prototype.py`, throwaway, see Status), compiled C++
  extension lands here as `_native*.so` (gitignored)
- `cpp/` — Phase 4's C++ engine (M1 toolchain + M4 `BattleState`/hand-crafted-eval
  port done, see Status; `include/be/`, `src/`, `bindings/module.cpp`, `tests/` per
  plans/precious-crafting-bachman.md's repo layout — user-implemented per this
  project's Phase 4 hard rule, Claude scaffolds headers/stubs/tests only; `build/`
  and `.cache/` gitignored)
- `scripts/` — runnable entry points (smoke test, benchmarks, replay fetching,
  dataset building, training, PPO training, PPO replay diagnosis
  `inspect_ppo_replays.py`, real-ladder play `ladder_ppo.py`, Showdown replay
  fetching `fetch_showdown_replays.py`, forward-model fidelity scoring
  `fidelity_harness.py`, C++ build (`build_cpp.sh`) and its
  ASan-aware pytest wrapper (`pytest_native.sh`))
- `tests/` — pytest (state encoding, damage calc, dataset/action-label logic, model
  training loops, harness determinism w/ seeded RNG, action-space translation, the
  PPO env — including one real-server integration test — PPO warm-start weight
  transplant, self-play, PPO eval/benchmark loading, native-extension bindings;
  Phase 6: the replay-log parser, the poke-env -> poke-engine translator, and the
  fidelity harness's own judgment calls — which turns it excludes and why)
- `pokemon-showdown/` — local simulator checkout (gitignored). `sim/SIM-PROTOCOL.md`
  is the authoritative spec for Phase 6's replay-log parser
- `poke-engine/` — Phase 6's Rust forward model, pinned checkout at v0.0.48
  (gitignored, built via `scripts/build_poke_engine.sh` — never `pip install`ed, see
  Commands). `src/state.rs`'s `State::deserialize` doctest is the authoritative state
  format; `poke-engine-py/src/lib.rs` is the exposed Python API surface
- `data/` — gitignored: `replays_raw/` (Metamon parsed replays, supervised datasets
  only), `replays_showdown/` (Phase 6: raw Showdown replay JSON, both sides'
  actions), `dataset/` (cached train/val
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
