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

### Milestone E: multiple rounds, gate still not met, plateaued ~35-40%

`win_prob.make_eval_fn` wires the trained model into `TwoPlySearchPlayer`'s scoring
(`scripts/benchmark.py --p1 learned --p2 search`) — same search shape as Phase 1, only
the eval function changes. First real result exposed a format mismatch that had been
sitting undocumented: the default `--format` is `gen9randombattle` (Phase 0/1's
format), but the model trains on `gen9ou` human replays (constructed teams) —
`gen9randombattle` has no team-building infra, so a `learned` benchmark there tests
the model on team compositions/movesets it never saw in training. Metamon's own replay
corpus turned out to have *no* random-battle data at all, for any generation — checked
via the HF dataset's file listing, not assumed — so this wasn't a fixable oversight in
Phase 0/1's original format choice, it was a wrong premise.

Fixed by building real `gen9ou` benchmarking: `battle_engine/teams.py` holds 5 real
Smogon sample-team exports (`RandomTeamFromPool`, a poke-env `Teambuilder`) — sourced
from a public forum thread, then validated (not just trusted) against the local
`pokemon-showdown validate-team gen9ou` checkout, which caught 2 of the originally-
fetched teams actually failing on this ruleset (one used the since-banned Tera Blast,
one used Baxcalibur — tagged Uber here — and Spore, banned by this ruleset's Sleep
Moves Clause) — both swapped for validator-clean teams from the same thread.
`scripts/benchmark.py` now threads `team=` through for any non-`gen9randombattle`
format.

First on-distribution gate result: **30.2% [26.3, 34.4]** — a clear, confident loss.
Diagnosed via the same replay-inspection technique that found Phase 1's bugs: at
critically low HP against a healthy opponent, the learned eval ranked *staying in and
attacking* above every switch option, including switches to the team's own designated
safe pivots — the same "shallow 2-ply search can't see switching's multi-turn payoff"
blind spot Phase 1's switch-urgency patch exists to fix for `evaluate()`, silently
unfixed for the learned eval (`switch_urgency_weight=0.0` there, correctly - a weight
tuned for `evaluate()`'s [-4,4] scale would swamp a [0,1] probability output - but
nothing built to replace it).

Three fixes, each verified independently before trusting the next:
1. **Data was 100% December 2023** (checked via a temporal breakdown of the fetched
   sample) — a ~2.5-year-stale metagame snapshot relative to the 2026 sample teams
   used for benchmarking. `fetch_replay_sample.py --accept-probability` (already
   existed, was just unused at scale) forces deeper archive scanning for temporal
   spread; scaled to 30,000 replays spanning Dec 2023 → May 2026 — cheap on this
   machine (~15 min total for fetch + rebuild + retrain, ~3,000 members/sec scan
   rate).
2. **`expected_damage()` silently returned 0.0 for fixed-damage moves** (Seismic
   Toss, Night Shade: damage = attacker's level; Dragon Rage, Sonic Boom: a flat
   number) — poke-env's `Move.base_power` is 0 for these (they use a separate
   `Move.damage` field the code wasn't reading), so they fell through the same path
   as true status moves. Found via the switch-ranking diagnostic: Blissey's Seismic
   Toss scored identically to Soft-Boiled. `damage.py`'s `_fixed_damage()` now
   handles both cases, respecting type immunity but correctly skipping STAB/crit/roll
   (real Showdown mechanics for these moves). HP-dependent variants (Super Fang,
   Final Gambit, Endeavor — poke-env marks these `damageCallback`, no plain `damage`
   value) deliberately left out, same "named simplification" convention as the rest
   of this module.
3. **`switch_urgency_weight` swept** (0.0-0.2, 80 battles/point) for the learned eval
   specifically — 0.08 came out best and is now `scripts/benchmark.py`'s
   `LEARNED_SWITCH_URGENCY_WEIGHT` constant.

Result after all three: **38.4% [34.2, 42.7]** — real, clear improvement over 30.2%,
still a confident loss.

### Encoding gap found: items and movesets were never read

Replay inspection during the above diagnosis kept surfacing the same pattern: the
model didn't seem to know a switch-in resisted hazards or could heal. Checked against
real downloaded data (not assumed): both live poke-env `Pokemon` objects and Metamon's
replay `mon` dicts carry full `item` and `moves` (name/type/category/base_power/
priority/PP per move) — none of it was in the encoder. `encoding.py`'s `PokemonView`
gained:
- `item`: one-hot over a data-driven top-20 vocabulary (verified against real replay
  frequency) + an "other known item" bucket, `None` folding together "no item" and
  "not yet revealed" (same convention already used for `ability`).
- `MoveSummary`: hand-engineered features per known moveset, sourced from Showdown's
  own movedex flags via `_MOVES_DEX` (not hand-guessed) — `has_recovery` (flags.heal),
  `has_hazard_setup` (sideCondition), `has_hazard_removal` (a short hardcoded list,
  same reasoning as `_HAZARD_TOKENS` — Showdown has no data-level flag for this),
  `has_setup_boost`, `has_pivot` (flags.selfSwitch), `has_priority`, `max_base_power`,
  and move-type coverage (distinct from the mon's own types).

Vector grew 316 → 656 dims (`320` was the last exact number before this — the module
docstring's own "check the real shape, don't trust a number here" warning applies to
whatever's current). Naively retraining the *old* single-64-unit-layer model on the
bigger vector made things *worse* (val_acc 0.651→0.636, benchmark 38.4%→32.0%) — more
input dimensions, same fixed capacity, more room to overfit, not more signal used.
Fixed by bumping `WinProbModel` to two hidden layers (128→64, dropout 0.3, both now
`hidden_sizes: Sequence[int]` instead of a single `hidden_size: int` — checkpoint
format changed accordingly). Result: **39.6% [35.4, 44.0]** — real but marginal over
38.4%, and notably achieved with *lower* val_acc (0.638-0.640) than the pre-encoding
model's 0.651, confirming aggregate val accuracy is not a reliable proxy for
head-to-head strength on this project (a smaller intermediate 150-battle sample had
shown 46.0%, which looked like a breakthrough — the full 500-battle rerun walked that
back to 39.6%; small-N benchmark reads misled once here, worth remembering).

### Independent review (Sonnet, scoped to this session's diff, 2026-07-31)

Same "verify against real data/library source, don't trust the code's own comments"
practice as every prior review, run on `damage.py`, `encoding.py`, `win_prob.py`,
`teams.py`, `benchmark.py`, `train_win_prob.py`, and their tests. Two real bugs found,
both in `has_setup_boost`:
- **False positive**: gated only on "any positive value in a move's `boosts` dict",
  which doesn't check *who* the boost applies to — Swagger/Flatter have positive
  `boosts` values but `target: "normal"` (they buff the *opponent*, while confusing/
  taunting them). Fixed by gating on `target == "self"`.
- **False negative**: Belly Drum and Acupressure are real, competitively significant
  self-buff moves implemented via Showdown's `onHit` simulator logic instead of a
  declarative `boosts` field, so they weren't flagged at all. Fixed with a short
  hardcoded `_ONHIT_SETUP_MOVES` list, same pattern as `_HAZARD_REMOVAL_MOVES`. Curse
  deliberately excluded even though it's also `onHit`/`boosts=None`: its effect is
  genuinely type-dependent (self-buff for non-Ghost users, no buff for Ghost users)
  and can't be resolved without the user's type at lookup time — the conservative
  "not a setup move" default is a documented choice, not a gap.

Also fixed two minor findings: `_ITEM_VOCAB`'s 20th slot (`blackglasses`) wasn't
actually the 20th-most-common item on a larger sample (`blacksludge` is, swapped), and
`benchmark.py`'s team-assignment check was a negative check (`!= "gen9randombattle"`)
rather than an explicit allowlist (tightened — not a live bug given this CLI's actual
usage, but a real footgun for any future format).

Rebuilt dataset + retrained after these fixes: **35.6% [31.5, 39.9]** — statistically
indistinguishable from 39.6% (heavy CI overlap). The bug fixes were correct and worth
making on their own merits, but didn't move the win rate. A follow-up
`switch_urgency_weight` re-sweep (6 points, 100 battles each) against this final model
also found no distinguishable winner among 0.0-0.20 — all CIs overlap heavily, 0.08
remains the best point estimate, already the configured default.

**Honest read of the whole sequence** (30.2% → 38.4% → 39.6% → 35.6%): after the first
real jump, three more rounds of legitimate improvements (richer features, more
capacity, real bug fixes) plateaued in the same 35-40% band. That's a real pattern,
not noise from any single run — most likely evidence that a learned eval bolted onto
the same shallow 2-ply search (which still can't see multi-turn value beyond a
hand-tuned switch-urgency patch) has hit a structural ceiling on this axis, not that
another eval-quality tweak is waiting to unlock it. **Milestone E gate: not met.**
Open options going forward: deeper search (real cost/complexity increase, edges into
Phase 4 MCTS/DUCT territory), more data (already tried once, diminishing so far,
though only within Metamon's own archive - true out-of-Metamon sources untried), or
moving on to Phase 3 (RL bypasses this specific ceiling by learning a policy directly
rather than ranking projected states through a fixed-depth lookahead - a genuinely
different mechanism, not just another eval swap).

### Imitation model built (2026-07-31) — Phase 2's originally-planned second milestone

The roadmap's Phase 2 section always specified *two* deliverables: the win-probability
model above, and a "move-prediction / imitation model (state → human's move) ...
opponent-modeling prior and later RL policy init" — never built until now, closing out
Phase 2's actual planned scope rather than leaving it implicitly Phase 3 prep.

Metamon's replay files carry a per-state `actions` field (`{"states": [...],
"actions": [...]}`, same top-level shape already documented above) that was never
inspected before — no metamon package is installed locally and its schema isn't
documented anywhere accessible, so the label scheme was reverse-engineered from real
data: integers -1 to 12, confirmed by correlating each value against a state's own
`can_tera`/`forced_switch`/`available_switches` fields. Resulting scheme
(`battle_engine/dataset.py`'s `ACTION_SPACE_SIZE = 13`): 0-3 move slot, 4-8 switch,
9-12 move slot while terastallized (100% of these states have `can_tera=True`), -1
missing/no ground truth (excluded from training). This is Metamon's own scheme, not
poke-env's Gymnasium `SinglesEnv` action space (a different 26-way encoding: 0-5
switch-by-team-index, 6-9 move, 10-25 mega/z-move/dynamax/tera variants gen9ou doesn't
use) — reconciling the two is a real, deliberately deferred step for whenever this
feeds a Phase 3 PPO policy init, not attempted now.

Switch-action labels needed remapping before they were usable: a raw switch action is
a *position* within that turn's live `available_switches` list, which (like the bench
slots documented earlier) isn't stable turn to turn. Measured on real data: 331/424
(78%) of switch actions in a 50-replay sample would be mislabeled if used directly.
Remapped to the same species-sorted stable order `encoding.py`'s bench slots already
use (`_replay_switch_slot_order` in `dataset.py`) — provably equivalent to that
function's bench-key construction by a set-identity argument (documented in the
code), not just asserted, since building the full `PokemonView`s just to recover
species identity would've been wasteful.

`battle_engine/imitation.py` (`ImitationModel`: same two-hidden-layer/dropout/Adam
architecture family as `WinProbModel`, swapped to a softmax head over
`ACTION_SPACE_SIZE` classes and `CrossEntropyLoss`) + `scripts/build_action_dataset.py`
/ `scripts/train_imitation.py`, mirroring the win-prob pipeline's structure
(best-val-loss checkpointing, same train/val battle-id split). No legal-action masking
at train or predict time — the model has to learn legality from data patterns alone,
same as a typical first-pass imitation setup.

Real numbers (30,000 replays, same set as the win-prob model): 707,273 train / 79,099
val labeled states (fewer than the win-prob dataset's 802,233/89,827 — ~12% of states
have a `-1`/missing action and are dropped). Best val_loss at epoch 26/30
(val_top1_acc 0.245) — notably, unlike every win-prob training run so far, this one
did *not* overfit early: train/val loss tracked closely the whole 30 epochs. 24.5%
top-1 accuracy over 13 classes is a real signal (uniform-random baseline ~7.7%,
always-guess-most-common-class baseline ~16.6%), modest but not spectacular — no
legal-action masking and no temporal/sequence context (single-state features only)
are the likely ceilings, both real, deliberately-deferred next steps rather than
issues fixed here. Not yet integrated anywhere (no opponent-modeling or Phase 3 use
yet) — this session's scope was building and validating it, not deploying it.

Next: open decision between (a) more Phase 2 tuning on the milestone-E axis (deeper
search, data beyond Metamon), or (b) moving to Phase 3 groundwork (PPO self-play,
now with both a value-function warm-start candidate (`win_prob.py`) and a policy
warm-start candidate (`imitation.py`) available, once the action-space reconciliation
above is resolved).

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
# on-distribution (the format its training data is actually in) - see Status above
# for why that distinction matters.
.venv/bin/python scripts/benchmark.py --p1 search --p2 maxdamage --n-battles 500
.venv/bin/python scripts/benchmark.py --p1 learned --p2 search --format gen9ou --n-battles 500

# Tests (integration test auto-skips if the local server isn't running)
.venv/bin/pytest

# Pull a small ELO-filtered sample of Metamon's gen9ou replay dataset (streams the
# 20+GB archive, stops after --n replays; never downloads it whole). Needs the `ml`
# extra installed: .venv/bin/pip install -e ".[ml,dev]"
# --accept-probability thins acceptances to force deeper (temporally wider) archive
# scanning at the cost of more bandwidth - see Status above, the current 30k-replay
# dataset needed this to not be 100% one month.
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
```

The venv is `.venv/` (Python 3.13); `pokemon-showdown/` is a gitignored local clone.
`data/` (fetched replay samples) is gitignored too — regenerate via the fetch script
above rather than committing it. Add new commands here as the harness/training
scripts land — don't leave this stale.

## Layout

- `battle_engine/` — the package: bots (`search.py`), eval (`evaluation.py`,
  `damage.py`), state encoding (`encoding.py`), dataset building (`dataset.py`),
  models (`win_prob.py`, `imitation.py`), benchmark harness (`benchmark.py`), team
  pool for gen9ou (`teams.py`)
- `scripts/` — runnable entry points (smoke test, benchmarks, replay fetching,
  dataset building, training)
- `tests/` — pytest (state encoding, damage calc, dataset/action-label logic, model
  training loops, harness determinism w/ seeded RNG)
- `pokemon-showdown/` — local simulator checkout (gitignored)
- `data/` — gitignored: `replays_raw/` (fetched replays), `dataset/` (cached train/val
  arrays for both the win-prob and action-label datasets), `models/` (trained
  checkpoints: `win_prob.pt`, `imitation.pt`) — see commands above

## Git workflow

The user commits and pushes themselves — do not run `git commit` or `git push` unless
explicitly asked in the moment. No `Co-Authored-By` trailers, ever.

## Notion sync

Project page: https://app.notion.com/p/3a1fe25f150b8106bfdef912a19dc33f (see parent `../CLAUDE.md` "Notion sync"
for the rule: update Status + "Recent activity" once per session with meaningful
progress).
