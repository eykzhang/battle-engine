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

Phase 0 (harness & baselines) in progress: repo scaffolded, venv + poke-env 0.15
installed, local Showdown server cloned, smoke test (random vs random) verified.
Next: max-damage bot + benchmark harness (see roadmap Phase 0 gate).

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

# Tests
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
