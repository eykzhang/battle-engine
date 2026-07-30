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

Phase 2 (first ML) started: state encoding is built and tested —
`battle_engine/encoding.py` maps either a live poke-env battle or one turn of a
Metamon parsed replay into the same fixed-size (316-dim) vector via two adapters
(`battle_view_from_poke_env`, `battle_view_from_replay_state`) over a shared
`BattleView`/`PokemonView` intermediate representation, `tests/test_encoding.py`
covers shape, edge cases (fainted/unknown slots), and adapter parity on an
equivalent state. The replay schema was verified against real downloaded data, not
assumed from metamon's docs — those describe a `ParsedReplay`/`turnlist` shape that
doesn't match the actual `.json.lz4` files, which are `{"states": [...], "actions":
[...]}` with each state already flattened/POV-relative. `scripts/
fetch_replay_sample.py` streams Metamon's gen9ou.tar.gz (20+GB on Hugging Face) and
decompresses it member-by-member, stopping once it's collected N ELO-filtered
replays, so a laptop-sized dev sample never requires the full download.

Known, accepted scope limit in the v1 encoder: opponent-side detail beyond the
current active Pokemon is a single "fraction remaining" scalar, not per-mon slots —
a replay's single state doesn't carry stats for opponent Pokemon that aren't
currently active, only a remaining-count. Revisit if the win-prob model's accuracy
looks bottlenecked on this.

Since then: added ability handling to the matchup-score dimension — known
type-immunity abilities (Levitate, Water Absorb, Flash Fire, Wonder Guard, ...) are
folded into `_type_multiplier` via `_TYPE_IMMUNITY_ABILITIES` rather than given their
own dimension (gen 9 has hundreds of abilities; a learned embedding is the right tool
for full ability identity, once an actual model exists to attach one to). Caught a
real test bug while building this: an early test used Rotom to demonstrate "ability
unrevealed vs. revealed", but Rotom only has one possible ability in-game, so poke-env
correctly auto-fills it as public knowledge — silently making the comparison a no-op.
Fixed by switching to Bronzong (genuinely ambiguous: Levitate/Heatproof/Heavy Metal).

Dataset pipeline (Phase 2 milestone C): `battle_engine/dataset.py` / `scripts/
build_dataset.py` build cached `(vector, label)` arrays (`data/dataset/{train,val}
.npz`) from fetched replays, split by *battle* (not file, not turn — see below).
Label is each replay's *final* outcome (filename's WIN/LOSS suffix, cross-checked
against the last state), not the per-state `battle_won`/`battle_lost` flag, which is
`False`/`False` for nearly the whole game and would otherwise train the model toward
"is the game already over and won this turn" instead of a real win-probability
signal.

Win-probability model (Phase 2 milestone D) is built and trained once:
`battle_engine/win_prob.py` (`WinProbModel`: `316(ish) → hidden → ReLU → Dropout →
1`, `train()`, `accuracy()`/`calibration_error()`) and `scripts/train_win_prob.py`.
Two independent Opus reviews (same practice that caught the Phase-1 HP-percent-scale
bug) found and fixed real bugs across milestones B–D. The pattern is consistent
enough to call out on its own: **every review so far has found something real by
checking actual downloaded data and library source, not by trusting docstrings** —
why "evidence over assumption" stays a hard rule here, not a nice-to-have.

Bugs found and fixed, first review (milestones A/B — `encoding.py`/`fetch_replay_sample.py`):
- Fainted teammates silently vanished from the replay-derived encoding (Metamon's
  `available_switches` never lists them) but not the live-battle one (poke-env keeps
  them in `battle.team`). Fixed with `battle_views_from_replay()` — needs a whole
  replay's turn sequence, not one state, to notice a teammate disappeared.
- Fixing that surfaced its own bug: tracking identity by Pokemon "name" breaks on
  in-battle form changes (Terapagos → `terapagosterastal` on Tera, Minior →
  `miniormeteor`). Fixed by keying on `base_species` instead (also used for stable
  bench-slot ordering on both adapters — replay bench order wasn't stable turn to
  turn either).
- Live-battle hazard encoding reported every simultaneously-active condition; real
  replay data's hazard field is single-valued/overwritten. Narrowed the live side to
  match (a deliberate fidelity trade, the user's call, not a bug by itself).
- `fetch_replay_sample.py`: added a request timeout (was unbounded), reports
  achieved date-range (an early run was found to be temporally biased — 30/30 files
  from one month), added `--accept-probability` to trade bandwidth for a wider
  spread.

Bugs found and fixed, second review (milestone D, plus a re-check of the first
review's fixes against a much larger real sample — 2,060 replays instead of 30):
- **`scripts/train_win_prob.py` was saving the *final*-epoch model while printing
  "best val_loss at epoch k"** — since this model reliably overfits well before
  training ends, the saved checkpoint was the single worst one of the run, silently.
  `train()` now tracks and returns the best-val-loss state dict directly.
- **Train/val leakage**: ~2.3% of Metamon's archived battles are stored from both
  players' POV as separate files sharing one battle id; splitting by file let some
  land with one POV in train and the mirrored POV (same game, inverted label) in val.
  `split_replays` now groups by battle id.
- The hazard-narrowing fix above was incomplete: Aurora Veil and Tailwind weren't in
  the vocabulary at all (a real miss, not the documented tradeoff), and `battle_field`
  (terrain) has the identical single-valued-masking issue (Trick Room/Gravity could
  mask an active terrain). Both fixed the same way, now sourced from poke-env's own
  `STACKABLE_CONDITIONS` rather than a hand-maintained guess at which conditions are
  turn- vs. count-tracked. The original "no removal signal, so narrowing is the only
  option" reasoning also turned out shakier than assumed — the field does revert to
  `noconditions` on a clean sweep — so fuller hazard reconstruction is a real,
  deliberately-deferred option now, not a closed question.
- `calibration_error` silently dropped predictions of exactly 1.0 (fell outside
  every `[lo, hi)` bin) — latent today, but exactly the failure mode a longer/bigger
  training run reaches. `predict_proba` left the model permanently in `eval()` mode
  instead of restoring the caller's prior mode.
- `battle_view_from_poke_env` would raise a confusing bench-overflow assertion at
  team preview (no active Pokemon yet, but `battle.team` already has all 6) — now a
  clear, intentional `ValueError` instead.
- `fetch_replay_sample.py`'s "idempotent" re-run skip check didn't count existing
  files toward the target, so re-running with the same `--n` always added N *more*
  files instead of topping up to N total — fixed, and now skips the network
  entirely if the target's already met.

Current real numbers (2,060 replays, ~2,013 distinct battles): 55,446 train / 6,406
val states, best val_loss 0.664 at epoch 2 of 30 (val_acc 0.633, cal_err 0.104),
overfitting sets in almost immediately after. Real but modest signal — nowhere near
wired into search yet.

Next: milestone E — swap the trained model into `search.py`'s scoring (behind
`evaluate()`'s interface) and re-benchmark against the Phase-1 bot. That head-to-head
win rate is the actual Phase 2 gate, not any of the training metrics above.

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

# Pull a small ELO-filtered sample of Metamon's gen9ou replay dataset (streams the
# 20+GB archive, stops after --n replays; never downloads it whole). Needs the `ml`
# extra installed: .venv/bin/pip install -e ".[ml,dev]"
.venv/bin/python scripts/fetch_replay_sample.py --n 200 --min-elo 1200

# Build the cached (vector, win/loss-label) dataset from fetched replays
.venv/bin/python scripts/build_dataset.py --replay-dir data/replays_raw --out-dir data/dataset

# Train the win-probability MLP on the cached dataset; saves the best-val-loss
# checkpoint (not the final epoch) to data/models/win_prob.pt
.venv/bin/python scripts/train_win_prob.py
```

The venv is `.venv/` (Python 3.13); `pokemon-showdown/` is a gitignored local clone.
`data/` (fetched replay samples) is gitignored too — regenerate via the fetch script
above rather than committing it. Add new commands here as the harness/training
scripts land — don't leave this stale.

## Layout

- `battle_engine/` — the package (bots, eval, encoding, search)
- `scripts/` — runnable entry points (smoke test, benchmarks, replay fetching, training)
- `tests/` — pytest (state encoding, damage calc, harness determinism w/ seeded RNG)
- `pokemon-showdown/` — local simulator checkout (gitignored)
- `data/` — gitignored: `replays_raw/` (fetched replays), `dataset/` (cached train/val
  arrays), `models/` (trained checkpoints) — see commands above

## Git workflow

The user commits and pushes themselves — do not run `git commit` or `git push` unless
explicitly asked in the moment. No `Co-Authored-By` trailers, ever.

## Notion sync

Project page: https://app.notion.com/p/3a1fe25f150b8106bfdef912a19dc33f (see parent `../CLAUDE.md` "Notion sync"
for the rule: update Status + "Recent activity" once per session with meaningful
progress).
