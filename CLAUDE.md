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

## Status

Phase 0 (harness & baselines) gate met — see git history for details.

Phase 1 (classical search) stopped here, with its primary gate met and a known,
documented gap: `TwoPlySearchPlayer` (`battle_engine/search.py`) does 2-ply lookahead
(my action, opponent's assumed best known-move reply) over a hand-crafted eval
(`battle_engine/evaluation.py`: HP/alive-count/type-matchup/status/hazards/speed) and
an expected-damage calculator (`battle_engine/damage.py`: gen-9 formula with roll/crit/
accuracy/multi-hit folded into one expected value, verified against Showdown's own
`sim/battle-actions.ts` crit table and a hand-derived damage example).

Gate results (500 battles each): beats `MaxBasePowerPlayer` 84.8% [81.4, 87.7] — clears
the >70% target. Beats `SimpleHeuristicsPlayer` 59.2% [54.8, 63.4] — a real, statistically
clear win (whole CI above 50%), though not as dominant as the max-damage matchup.

Getting there took two rounds of evidence-driven debugging, not guessing:
1. Replay inspection showed the bot originally never switched proactively (tanking
   repeated hits it should've pivoted away from). Fixed with a switch-urgency bonus in
   `search.py`, scaled by how bad the current type matchup is (a 2-ply search can't see
   switching's multi-turn payoff on its own). Confirmed via replay that real mid-battle
   switches started happening — but win rate barely moved (~38-39%), which was the clue
   something bigger was still wrong.
2. An independent Opus review (before anything was committed) found the actual dominant
   bug: `expected_damage()` was dividing real damage by `defender.max_hp`, but for the
   *opponent's* Pokemon mid-battle, poke-env reports `max_hp` on a 0-100 percent scale,
   not real HP points (their exact HP pool isn't known any more than their EVs/nature
   are). That silently inflated every opponent-side damage projection several-fold — the
   bot effectively saw almost every attack as a near-guaranteed opponent faint, which
   explains both the original never-switch bug (nothing can outrank a phantom KO) and
   why fixing switching alone didn't help. Fixed via `estimate_stat(defender, "hp")`
   (same known-vs-estimated-from-base-stats pattern already used for atk/def/spa/spd/
   spe) instead of raw `.max_hp`. A second, related bug from the same review (the
   projection charged full retaliation damage even when my move already fainted the
   opponent) was fixed alongside it. Verified independently before trusting the review's
   claim, and confirmed by rerunning the benchmark: heuristic win rate went 39.0% → 59.2%.

The remaining known gap: the bot still never uses non-damaging moves (status, setup,
hazards) — `expected_damage` returns 0 for them, so they can never outrank a damaging
move in the ranking. `SimpleHeuristicsPlayer` has explicit logic for this that we don't.

Decision: stop Phase 1 here rather than build status/setup-move modeling now — both
gates are now genuinely met. Phase 2's learned eval is expected to pick up on setup/
status value that the hand-crafted eval structurally can't express.

Next: Phase 2 (first ML) — learned win-probability eval + move-prediction model trained
on Metamon's replay dataset, swapped into the same search shape.

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
# --p1/--p2 choices: random, maxdamage, heuristic, search (our Phase-1 bot)
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500

# Tests (integration test auto-skips if the local server isn't running)
.venv/bin/pytest
```

The venv is `.venv/` (Python 3.13); `pokemon-showdown/` is a gitignored local clone.
Add new commands here as the harness/training scripts land — don't leave this stale.

## Layout

- `battle_engine/` — the package (bots, eval, encoding, search)
- `scripts/` — runnable entry points (smoke test, benchmarks, training)
- `tests/` — pytest (state encoding, damage calc, harness determinism w/ seeded RNG)
- `pokemon-showdown/` — local simulator checkout (gitignored)

## Git workflow

The user commits and pushes themselves — do not run `git commit` or `git push` unless
explicitly asked in the moment. No `Co-Authored-By` trailers, ever.

## Notion sync

Project page: https://app.notion.com/p/3a1fe25f150b8106bfdef912a19dc33f (see parent `../CLAUDE.md` "Notion sync"
for the rule: update Status + "Recent activity" once per session with meaningful
progress).
