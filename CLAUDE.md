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

### Phase 3 started (2026-07-31): action-space reconciliation, then PPO/SB3 wiring

Decision: (b) above — moving to Phase 3 groundwork rather than further Phase 2
tuning, per the roadmap's own order (PPO via stable-baselines3 first, "working
plumbing before hand-rolling anything").

**Action-space reconciliation** (`battle_engine/action_space.py`): translates between
`dataset.py`'s 13-way Metamon action scheme (what the imitation model was trained on,
and what this project's own state encoding's species-sorted bench ordering matches)
and poke-env's native 26-way Gymnasium `SinglesEnv` scheme (0-5 switch by raw
`battle.team` position, 6-9 move, 10-25 four gimmick blocks — mega/z-move/dynamax
none of which exist in gen9 OU, plus tera). Both directions needed, not just
metamon→poke-env: `battle_engine/rl_env.py`'s `MetamonActionSinglesEnv` (a
`SinglesEnv` subclass exposing `Discrete(13)` instead of the native `Discrete(26)`,
chosen over the native space because `PokeEnv.embed_battle` is abstract regardless of
action-space choice, so this project's own encoding — species-sorted bench, no raw
team-position feature — is used either way, and native 26-way switch actions would be
an unlearnable target against that observation) overrides
`action_to_order`/`order_to_action`/`get_action_mask`/`get_action_space_size`.

Two independent reviews (same "verify against real library source, not comments"
practice as every prior milestone) found and fixed real bugs:
- **First review**: `order_to_action` was missing entirely (inherited `SinglesEnv`'s
  native 26-way version) — poke-env's `SingleAgentWrapper` calls it on the
  *opponent's* real move to feed back through this env's own `step()`, so without an
  override the opponent's actual actions were silently corrupted one step later. Also
  fixed: the `battle._wait` mask fallback was inferred unsoundly (now explicit), a
  `-1`/`-2` sentinel collision, `get_action_space_size` still reporting 26.
- **Second review** (after `embed_battle`/`calc_reward`/`observation_spaces` and
  `scripts/train_ppo.py` were built): `strict=False` (used so an undertrained PPO
  policy's illegal picks don't crash training) silently substitutes a *different,
  randomly chosen* move — measured 56.2% of sampled actions illegal under a fresh
  near-uniform policy on real gen9ou states, meaning PPO was crediting the action it
  picked with the outcome of a different action poke-env actually played. Real fix
  (masking) deliberately deferred to the very next session, not patched in place. Also
  fixed: this project's own translation-layer `ValueError`s bypassed the
  strict/non-strict contract entirely (always raised, now respect `strict` like every
  other illegal-action case); a dead `team=` arg on the `SingleAgentWrapper` opponent
  that leaked a real, unclosed, never-actually-used websocket connection per
  `build_env()` call (`SingleAgentWrapper` never lets `opponent` actually play — it
  only calls `opponent.choose_move`/`teampreview` as pure decision functions over the
  battle `env.agent2` is really playing — fixed via `start_listening=False`); a
  docstring with the SB3 `CombinedExtractor` concatenation order backwards
  (`action_mask` first at columns 0:13, `observation` second at 13:669 — Gymnasium's
  `spaces.Dict` sorts keys alphabetically — matters for the deferred imitation-weight
  transplant).

**PPO wiring** (`scripts/train_ppo.py`): `MetamonActionSinglesEnv` +
stable-baselines3's `PPO("MultiInputPolicy", ...)` via poke-env's `SingleAgentWrapper`,
training against a fixed `RandomPlayer` (not self-play yet). Laptop feasibility
measured (per the hard rule below): ~660 steps/s, ~108-117 steps/battle (~6
battles/sec) — puts "hundreds of thousands of battles" at roughly 14 hours, inside an
overnight run, on a single environment with no parallelism.

**Action masking** (same session, right after the second review, since the review
found illegal-action substitution was a real training-quality problem — see above):
switched from plain `PPO` to `sb3-contrib`'s `MaskablePPO` + `ActionMasker`, wired to
`MetamonActionSinglesEnv.get_action_mask` via a new `sb3_contrib_action_mask_fn` in
`rl_env.py` (kept in the package, not the script, for the same reason everything else
here is: so it has real unit-test coverage instead of only being exercised by manual
script runs). Verified concretely, not just "should work": the pre-masking run logged
dozens of `Invalid action ... Defaulting to random move` warnings per few hundred
steps; the post-masking run logs zero across the same test, and throughput is
unaffected (~660 steps/s either way — masking is cheap, just a logit adjustment
before sampling).

**Imitation/win-prob weight init** (`battle_engine/ppo_warm_start.py`, same session):
masking made this possible — since legality is now enforced at the action-distribution
level, the network no longer needs `action_mask` as an input *feature*, so
`ObservationOnlyExtractor` drops it, making PPO's trunk exactly `VECTOR_LEN`-shaped
again (not `VECTOR_LEN + ACTION_SPACE_SIZE` via `CombinedExtractor`). Paired with
`net_arch=[128, 64]` and `activation_fn=nn.ReLU` (matching both Phase-2 checkpoints
exactly, verified against the real saved `state_dict`s, not assumed), PPO's actor
trunk is now literally the same shape as `ImitationModel`'s and the critic trunk the
same shape as `WinProbModel`'s, so `load_warm_start_weights` copies `Linear` weights
directly rather than approximating across a mismatch: `imitation.pt`'s trunk+output →
`policy.mlp_extractor.policy_net` + `policy.action_net`, `win_prob.pt`'s → `.value_net`
equivalents. Verified end-to-end, not just shape-checked: the warm-started policy
reproduces bit-identical logits/values to the two source checkpoints on the same input
(`tests/test_ppo_warm_start.py`). `WinProbModel`'s output is a win-probability logit,
not a calibrated return estimate — a directionally-reasonable value-function start,
left unrescaled since PPO's own value loss corrects scale through training and
guessing at a rescaling without a real run to check it against would be exactly the
kind of unverified assumption this project avoids. `scripts/train_ppo.py --warm-start`
is now the default (`--no-warm-start` to train from scratch); a quick real run showed
the qualitative difference immediately — `explained_variance` starts at ~0.65 instead
of a random-init run's negative value, and episode reward is already positive in the
second rollout instead of consistently negative.

**Self-play** (`battle_engine/self_play.py`, same session — the last of the three
review-motivated follow-ups): `FrozenPolicyPlayer` (a poke-env `Player`) samples a
new frozen snapshot of the trainee's own past weights once per battle (tracked via
`battle.battle_tag`, not per turn — a policy that switched personality mid-battle
wouldn't be a coherent opponent), from a small pool (`max_snapshots`, default 5) kept
fresh by `SelfPlaySnapshotCallback` pushing a copy of the live policy's weights every
`snapshot_interval` timesteps during `model.learn()`. Buildable directly on top of the
warm-start work: the frozen opponent is just another `MaskableActorCriticPolicy` built
with the same `warm_start_policy_kwargs()`, so any snapshot of the trainee's
`state_dict()` loads into it with no translation needed. Seeded with the trainee's own
starting weights (so battle 1 has a real opponent, not an empty pool) —
`scripts/train_ppo.py --opponent {self-play,random}` (self-play is now the default;
`random` kept for quick smoke tests). Verified with a real run against the local
server (`--snapshot-interval 256 --max-snapshots 3` over 1024 timesteps): snapshots
pushed and sampled correctly, zero illegal-action warnings (masking still applies
regardless of opponent), no errors.

This closes out all three review-motivated follow-ups from this session (masking,
warm-start, self-play) — 138 tests passing.

### Third review (2026-08-01, scoped to masking/warm-start/self-play)

Same practice, verified against real library source and real instrumented training
runs rather than re-reading prior summaries. Findings and fixes:
- **`--no-warm-start` was silently changing the architecture, not just the init** —
  gating `policy_kwargs` itself on `--warm-start` meant the `--no-warm-start` path got
  SB3's *defaults* (`net_arch` `{64,64}` per head, `Tanh`, sees `action_mask` as an
  input feature, ~95k params) instead of the warm-start shape (`[128,64]`, `ReLU`,
  observation-only, ~186k params) — confounding architecture with initialization for
  any future "does warm-start help" comparison (a real Phase-3 gate input). Fixed:
  `warm_start_policy_kwargs()` now always applies; `--no-warm-start` only skips
  `load_warm_start_weights`.
- **The value/actor gradient-coupling risk `ppo_warm_start.py` flagged as a
  possibility was measured, not just theorized**: stable-baselines3 applies one
  global gradient-norm clip across the whole policy, so `WinProbModel`'s uncalibrated
  win-probability-logit scale (vs. the discounted-return scale PPO's critic actually
  predicts) does throttle the warm-started actor's effective gradient — 2.8x smaller
  on the very first update, measured on a real instrumented run with a
  matched-architecture random-init control to isolate this from the confound above.
  Converges to the control's magnitude by about update 4 (~1k timesteps into a run
  planned for hundreds of thousands) — real and measured, but self-correcting early
  enough not to be worth guessing at a rescaling factor for. `ppo_warm_start.py`'s
  docstring now states this with numbers instead of as a hedge.
- **A self-play test was vacuous** (`test_choose_move_only_resamples_a_snapshot_between_battles_not_within_one`,
  now `test_choose_move_resamples_between_battles_but_not_within_one`): it seeded only
  one snapshot, so "resampling didn't change anything" held whether or not resampling
  actually happened — proven by mutation testing (swapping the `battle_tag` check for
  an unconditional resample still passed the old test). Fixed by seeding two
  distinguishable snapshots and asserting both directions: unchanged within a
  `battle_tag`, and a real weight change observable across enough distinct ones.
  Re-verified the fixed test actually fails against the same mutation before trusting
  it.
- **The websocket-leak pattern the second review fixed in `build_env()` had crept back
  into `tests/test_rl_env.py`'s real-server integration test** (predated that fix, was
  never updated) — same `start_listening=False`, no `team=` fix applied there too.

Checked and confirmed correct, no bug: the full masking chain through
`DummyVecEnv`→`Monitor`→`ActionMasker`→`SingleAgentWrapper` (verified `gymnasium`
1.3.0's `Wrapper` has no `__getattr__`, but `get_wrapper_attr` recurses correctly
regardless — mask matched the live observation byte-for-byte across a full real run);
no concurrency hazard between `FrozenPolicyPlayer`'s inference and the trainee's
gradient updates (both confirmed `MainThread`-only via real instrumentation); the two
policies' `state_dict()` key sets are identical; the warm-start-then-seed ordering in
`train_ppo.py` is correct as written; snapshot independence/eviction both hold.

Not yet done: an actual real-scale training run (the small feasibility runs above are
all far too short to show real learning) and the eventual Phase 3 gate check
(RL-tuned policy beats the Phase-2 supervised bot head-to-head, plus a real ladder GXE
number).

### Sanity run (2026-08-01): plumbing proven, plateau found, real behavioral bug diagnosed

`scripts/train_ppo.py` gained `--resume-from` (loads a `--checkpoint` `.zip` and continues
training via SB3's `reset_num_timesteps=False`, `--timesteps` meaning "more steps," not
a new absolute target; forces `--warm-start` off with a visible message, since the
resumed weights already *are* the trained state). Verified against a real checkpoint,
not just constructed: `num_timesteps`/`n_steps`/`policy_kwargs` (including
`ObservationOnlyExtractor`) all round-trip correctly through `MaskablePPO.load`. A
separate small verification run (`--resume-from` the 500,000-step checkpoint,
`--eval-interval 1024`) confirmed `EvalVsOpponentCallback`/`SelfPlaySnapshotCallback`
(both keyed on `self.num_timesteps`, which correctly continues from 500,000 rather than
resetting to 0) fire on the *original absolute* schedule: the first eval landed at
500,736 — the next multiple of 1,024 counting from 0 (500,736 = 489 × 1,024), i.e. the
schedule behaves as if training had never stopped, not as if a new count started at the
resume point. (`CheckpointCallback` specifically does NOT get this property — see
`scripts/train_ppo.py`'s comment, fixed after an independent review, 2026-08-01, caught
an earlier version of this note wrongly claiming it did; harmless for a resume landing
on a round multiple of both intervals, as this project's actual resumed run did.)

A planned 1,000,000-timestep self-play run (warm-started, masked, `--eval-interval
20000 --eval-battles 20` against `TwoPlySearchPlayer`) was deliberately stopped at
560k to save usage, then resumed from the 500k checkpoint and run to completion —
47 eval points total, clean finish, zero errors, zero leaked connections, unbroken
checkpoint chain 100k→1,000,000, final model saved to `data/models/ppo.zip`.

**Result: a real, honest plateau, not a run that needed more time.** Win rate vs.
`TwoPlySearchPlayer` climbed from ~20% early to a ~28-32% band by mid-run and stayed
there — the second half of the run (500k-1,000,000, i.e. doubling total training)
showed no further improvement over the first half. Decision: **do not commit to the
planned overnight run as-configured** — doubling steps already failed to move the
needle, so scaling to ~27x more compute (an "overnight" budget at this throughput)
on the identical setup is a bad bet. This mirrors Phase 2's own milestone-E lesson:
diagnose before scaling compute blindly.

**Diagnosis, ranked**: (1) self-play's opponent pool was too narrow/stale relative to
total training length (`max_snapshots=5`, `snapshot_interval=4096` — a rolling ~2%
window of training history); (2) the trainee never once played `TwoPlySearchPlayer`
during training, only itself — self-play converging to "beats recent self" doesn't
have to transfer to "beats 2-ply lookahead"; (3) reward-shaping weights and PPO's own
hyperparameters were never tuned past their initial defaults.

**A fourth, more concrete cause was found via qualitative diagnosis** (`scripts/
inspect_ppo_replays.py`, new diagnostic script — same "watch real replays" technique
that found Phase 1's switch bug and Phase 2's damage-calc bugs): the trained policy
repeatedly re-used a protect-counter move (Protect, Endure, ...) immediately after it
had just failed, walking back into Showdown's own escalating-failure mechanic (each
consecutive use drops success chance geometrically) and taking large, predictable HP
losses as a direct, quantified result — verified turn-by-turn against real
`Pokemon.protect_counter`/HP data, not just repeated-action-name pattern-matching: HP
crashing from 1.00 to 0.54 immediately after a 2nd consecutive Protect, in a battle
the policy went on to lose, the same pattern twice in one game. Also found (a
separate, not-yet-addressed issue): genuine multi-dozen-turn stalling switch-loops
between the trainee and the search bot, matching the "auto-tie at turn 1000" server
warnings observed during the sanity run's own eval battles.

**Root cause of the protect-spam pathology: a real encoder gap, not a training-config
problem.** poke-env already tracks the exact mechanic (`Pokemon.protect_counter`) but
`encoding.py` never read it — no model trained on the old vector could structurally
learn "don't protect again, it's about to fail," the information was invisible to it.
Fixed: `PokemonView.protect_counter` (`encoding.py`), exact via `mon.protect_counter`
on the live adapter, reconstructed from replay turn-sequence history on the other (no
equivalent field exists in Metamon's per-state schema — a verified, named
simplification: increments on consecutive same-side use of a protect-counter move,
resetting on a different move or an active-species change, without attempting to
detect success/failure specifically, which would need inferring "no incoming damage"
from hp_pct deltas — confounded by residual status damage ticking through a
*successful* Protect. Checked against real data before accepting the trade: a
300-replay/8,815-state sample shows protect-counter moves in only 1.35% of states,
longest observed streak 3, and every observed streak-of-2-or-more instance ended in
that Pokemon fainting the same turn — independent confirmation, on real human games,
that the simplified streak still captures the real pattern). `VECTOR_LEN` 656 → 663.

Datasets rebuilt and both Phase-2 models retrained on the new vector (sample counts
match the pre-change run exactly, confirming nothing else broke): `win_prob.pt`
val_acc 0.634 (consistent with the historical ~0.63-0.64 band — one added scalar
among 663 dims isn't expected to move aggregate win-prediction accuracy much; the
real payoff is expected in the RL policy's live decisions, not this number).
`imitation.pt` val_top1_acc 0.245 (matches the prior number exactly). 150 tests
passing (7 new, covering both adapters' exact/reconstructed semantics and the
encoding's clamp/normalization).

**Important, expected side effect**: all pre-existing PPO checkpoints (`ppo.zip`,
`data/models/checkpoints/*.zip` from this sanity run) are now shape-incompatible
with the new 663-dim observation space — `--resume-from` on any of them will fail.
Kept on disk as a historical record of the plateau result, not resumable.

### Fourth review (2026-08-01, scoped to this session's resume/protect_counter/retrain work)

Same practice, verified against real data and library source rather than the prior
session's own summary. One real HIGH-severity bug found in the just-built feature
itself, plus a serious verification-integrity finding, a flaky test, and low-severity
issues — all fixed and re-verified, not just noted.

- **`_replay_protect_streaks` over-counted streaks — a real bug in the reconstruction
  logic, not the documented success/failure simplification.** `player_prev_move`/
  `opponent_prev_move` are NOT "the move used on this transition" — they're "the last
  move this side has ever used," carried forward byte-for-byte (name AND `current_pp`
  AND every other field) across states where that side didn't actually act (Metamon
  emits extra decision states, e.g. for the opponent's forced switch after a KO, where
  the player's own side did nothing). The original version had no way to distinguish a
  genuine second consecutive protect from a stale carried-over one — both showed the
  name "protect" — so it kept incrementing an already-resolved streak. Measured against
  real per-move PP as independent ground truth: **59.4% of reconstructed streak-2
  values and 41.7% of streak-3 values were wrong.** Fixed by comparing each side's
  whole `prev_move` dict to the previous state's: identical means nothing happened for
  that side this transition, so the streak carries forward unchanged rather than being
  read as a fresh use — a signal available directly from `states`, no need to thread
  the replay's separate action labels through. Re-verified post-fix against the same
  independent ground truth: 1,238/1,242 (99.7%) of streak≥2 reconstructions now
  correspond to a real PP drop, across a 3,000-replay sample.
- **The original "verification" numbers in `encoding.py`'s docstring and this file were
  themselves corrupted by the bug they were meant to validate** — a serious catch, since
  the whole point of quoting them was to demonstrate the simplification was checked
  against real data, not asserted. "Longest observed streak 3" was false (the actual
  pre-fix dataset contains streak values corresponding to 4 and 5). "Every streak-2+
  instance ended in a faint" was false (only 28-39% did, depending on sample) — worse,
  the reasoning was circular: the bug's own post-faint phantom state was *creating*
  those streak≥2-at-faint instances, so the "confirmation" was reading the artifact of
  the bug as evidence the approximation was sound. Post-fix, re-measured honestly: a
  nonzero streak occurs on 1.34% of (state, side) pairs, longest streak seen is 2 (not
  3+), and streak≥2 correlates with a same-turn faint only 11.4% of the time (70
  instances, 8 at a faint) — genuine multi-use protect streaks are just rare in human
  play, which is a fine, honest characterization; "always ends badly" was never true.
- **`tests/test_rl_env.py`'s real-server integration test was flaky**: measured 7/40
  (17.5%) failures hitting its 200-step cap before a battle terminated — random-vs-random
  gen9ou battles regularly run long (this session's own diagnosis found real
  multi-dozen-turn stalling switch-loops), and Showdown itself only force-ends via
  auto-tie at turn 1000. Fixed by raising the cap to 1050 — safely past that
  server-enforced hard limit, so every real battle deterministically terminates before
  the cap rather than merely being statistically likely to; re-run 5/5 clean afterward.
- **Two low-severity issues, both fixed**: `scripts/inspect_ppo_replays.py` leaked both
  websocket connections on any exception (no `try/finally` around the battle loop,
  unlike `ppo_eval.py`'s own `EvalVsOpponentCallback`, which gets this right) — fixed.
  `scripts/train_ppo.py`'s resume comment claimed `CheckpointCallback`'s save interval
  is keyed on absolute `num_timesteps` like this project's own two callbacks
  (`EvalVsOpponentCallback`, `SelfPlaySnapshotCallback`) — false, verified against
  installed SB3 source: `CheckpointCallback` uses `self.n_calls`, which resets to 0
  every fresh `main()` invocation regardless of resume, so its save *cadence* is
  calls-since-this-run-started, not the absolute schedule (the saved filename's own
  timestep number is still absolute and correct). Harmless for a resume landing on a
  round multiple of both intervals (as the actual sanity-run resume did), but the
  comment was wrong and is now fixed. `README.md` still said "656-dim vector" —
  updated to point at `VECTOR_LEN` instead of a hardcoded number, the same
  "don't trust a stale dimension in a comment" rule this project has stated (and
  violated, then caught) more than once now.
- Checked and confirmed correct, no bug: `_PROTECT_COUNTER_MOVES`'s transcription
  (exact set-equality match against poke-env's real private constant), opponent-side
  switch-detection symmetry, first-state (i=0) handling, `_species_key`/`base_species`
  form-change safety, `_encode_pokemon`'s field placement, `PokemonView.unknown()`/
  fainted-teammate defaults, dataset/checkpoint freshness (all four `.npz` files and
  both `.pt` checkpoints confirmed 663-wide via direct inspection, not just mtimes),
  and `--resume-from`'s actual n_steps/policy_kwargs/num_timesteps round-trip.

Datasets rebuilt and both models retrained a second time on the corrected encoder
(sample counts unchanged, confirming nothing else broke): `win_prob.pt` val_acc 0.644,
`imitation.pt` val_top1_acc 0.243 — both consistent with the historical band. 151 tests
passing.

Next: rerun the sanity-run methodology (self-play + eval-vs-search-bot tracking) with
the now-correctly-reconstructed protect_counter feature in place to see whether it
moves the plateau, before deciding on self-play-pool-widening/mixed-opponent-training
(still real, still not done) or an overnight run.

### Self-play pool widened + search-bot mixing added (2026-08-09)

Before committing to another expensive full-length run, addressed the two remaining
open items from the sanity-run plateau diagnosis (the `protect_counter` encoder gap
was already fixed and retrained above, but never actually re-run) — rather than
re-running with only that one fix and possibly still seeing the plateau for a
reason already suspected but untouched:

- **Self-play pool was a rolling ~2% window of a long run** (`DEFAULT_MAX_SNAPSHOTS=5`
  at `DEFAULT_SNAPSHOT_INTERVAL=4096`, over a 1,000,000-timestep run). Widened to 20
  (~8.2% of the same run) — cheap: each snapshot is one ~186k-param MLP state_dict
  (~750KB).
- **The trainee never once played `TwoPlySearchPlayer` during training, only itself**
  — self-play converging to "beats recent self" was never required to transfer to
  "beats 2-ply lookahead," and `--eval`'s periodic real games against it only ever
  *measured* that gap, never trained against it. Fixed with a new
  `battle_engine/self_play.py::MixedOpponentPlayer` (delegates each battle, resampled
  once per `battle_tag` like `FrozenPolicyPlayer` already does, to a weighted set of
  sub-players) and `scripts/train_ppo.py --search-bot-fraction` (default 0.2,
  self-play only; 0.0 reproduces the old pure-self-play behavior exactly). Wired so
  `SelfPlaySnapshotCallback`/pool-seeding still target the inner `FrozenPolicyPlayer`
  directly even when it's wrapped in a `MixedOpponentPlayer`.

Verified with two real short runs against the local server (not just unit tests) — a
50/50 self-play/search-bot mix and a pure-self-play (`--search-bot-fraction 0.0`) run
for backward compatibility — both completed cleanly, zero illegal-action warnings,
zero errors. 4 new tests in `tests/test_self_play.py` (delegation, zero-weight
sub-players never selected, per-battle-not-per-turn resampling using the same
mutation-tested shape as `FrozenPolicyPlayer`'s own resampling test). 155 tests
passing.

Not yet done: an actual real-scale run with all three fixes together
(`protect_counter`, widened pool, search-bot mixing) — this session's scope was
building and unit/smoke-verifying the fixes, not spending the wall-clock on a new
sanity run. That combined run is the next real step before judging whether the
plateau moves. PPO/reward hyperparameter tuning (the diagnosis's fourth, vaguest
item) remains deliberately deferred until after that run.

### 500k sanity run + reward rebalance + hazard-immunity encoding fix (2026-08-09)

The combined run above (protect_counter + widened pool + search-bot mixing) was run
for real: 500,000 timesteps, `--eval-interval 20000 --eval-battles 20`. Periodic
eval-vs-`TwoPlySearchPlayer` climbed noisily to a final point of 45.0% (9/20), and a
resume to 1,000,000 total steps kept climbing (final point 55.0% at 920k, ending at
40.0%). Pooling properly rather than trusting individual n=20 points: first half
(20k-500k) 122/500 = 24.4%, second half (500k-1,000,000) 163/500 = 32.6% [28.5%,
36.7%] — at the *top edge* of the old ~28-32% plateau band, not a clean break from it.
The real, gate-standard 500-battle benchmark on the resulting checkpoint
(`scripts/benchmark.py --p1 ppo --p2 search --format gen9ou --n-battles 500`) gave
**171/500 = 34.2% [30.2%, 38.5%]** — a real but modest improvement over the historical
band, and still a clear loss (PPO still loses to the search bot roughly 2-to-1).
**Phase 3 gate: still not met.**

Two more fixes followed, both from real diagnosis, not guesses:

**Reward rebalance** (`battle_engine/rl_env.py`): `_FAINTED_VALUE`/`_HP_VALUE` 0.15 ->
0.05, `_VICTORY_VALUE` unchanged at 1.0. Derived from `PokeEnv.reward_computing_helper`'s
actual source (each side's 6 Pokemon contribute up to ±value to the state value, so
worst-case per-battle shaping is `12*value` total, not "per side" as an earlier version
of this note said - `12*0.15 = 1.8` against the terminal `±1.0` victory term, confirming
shaping could outweigh winning/losing itself; `12*0.05 = 0.6` keeps a real dense signal
while staying clearly subordinate).

**Hazard-immunity encoding gap** (`battle_engine/encoding.py`): found via
`scripts/inspect_ppo_replays.py` on the reward-rebalanced-only checkpoint - a real,
severe pathology in real gameplay, not a training artifact. Two separate instances:
- A trained policy spammed Spikes 32 consecutive turns against a Flying-type opponent
  fully immune to it (turns 114-145 of one real battle) - the vector had no signal at
  all for hazard-immunity by type/ability, a mechanic entirely separate from the
  existing type-chart-based matchup-score dimension.
- A second, distinct pattern (Stealth Rock recast 8 turns straight while the user's own
  Pokemon's HP drained 58%->6% with no other action taken) turned out NOT to be the
  same kind of gap on inspection: hazard *presence* was already visible in the vector
  (`opp_hazards` containing `"stealthrock"`), so this reads as the policy not yet having
  learned to act on an already-available signal (a training-insufficiency pattern) not
  a missing-feature one - no encoding fix attempted for this specific case, flagged
  instead as a candidate that more training/reward-shaping might address, not more
  features.

Fixed the first (real, structural) gap: `_is_hazard_immune` (`my_active_hazard_immune`/
`opp_active_hazard_immune`, 2 new scalar dims, `VECTOR_LEN` 663 -> 665) - true for
Flying-type, Levitate ability, or Heavy-Duty Boots (see the function's own docstring for
the full accounting and what's deliberately still out of scope: Iron Ball/Gravity/
Magnet Rise/Ingrain/Smack Down/Roost, all measured rarer than Boots on real data).

An Opus review of this session's full diff (reward rebalance, self-play pool widening,
search-bot mixing, hazard-immunity encoding), run in parallel with a second, independent
"look through more replays" pass, both against real library source and real data rather
than the diff's own comments - found:
- **The hazard-immunity feature's first version was itself incomplete**: it only
  checked Flying-type/Levitate, originally named `_is_grounded`. Measured on real replay
  data (300 replays, 20,610 active-mon states): Heavy-Duty Boots is the #2 most common
  immunity source (5.19% of states, behind Flying's 17.0%, ahead of Levitate's 2.46%)
  and the *only* one of the three that also blocks Stealth Rock - the original version
  confidently mislabeled every boots holder as vulnerable, sitting right next to the raw
  item feature (`_ITEM_VOCAB` already includes `"heavydutyboots"`) without using it. Real
  catch: this directly undercuts the fix's own purpose (a boots-holding opponent, ~1 in
  20 states, would still trigger the same 32-turn-spam pathology). Fixed and renamed
  before any training used it - datasets rebuilt and both Phase-2 models retrained a
  third time on the corrected values (not just re-widened dimensions this time - the
  boots fix changes real feature *values* for ~5% of states, not just vector width).
- `--search-bot-fraction` outside `[0.0, 1.0]` failed silently rather than raising
  (`random.choices` only rejects a non-positive *weight total*, and self-play's weight
  here is always >= 0) - e.g. `1.5` silently resolved to "always pick the search bot,"
  making a typo read as a successful run of a different experiment. Fixed with an
  explicit range check.
- A pre-existing (not introduced this session) websocket leak on `--resume-from`
  failure: `build_env()` opens real connections before `MaskablePPO.load` is attempted,
  and any load failure (e.g. a shape mismatch from resuming across an encoding change)
  left both open with no path to `env.close()`. Fixed with a try/except around the load.
- Stale dimension numbers in `README.md` and `ppo_warm_start.py`'s docstring (still said
  "663", now "665") and two inaccuracies in this file's own prior section (said "5 new
  tests" for `tests/test_self_play.py`, actually 4; a test count that was accurate for
  its own point in time but easy to misread as current) - both fixed.
- **Flagged, not yet acted on** (plausible but explicitly not confirmed by the reviewer):
  cutting `_FAINTED_VALUE`/`_HP_VALUE` 3x may be making the warm-started critic's initial
  fit *worse*, not just smaller - `WinProbModel`'s output scale doesn't move with the
  reward rebalance, so shrinking return variance without rescaling the critic could widen
  the mismatch between them. Measured on 5 short instrumented runs per setting:
  first-update `explained_variance` was negative in 4/5 runs at 0.05 (one run -2.54) vs.
  negative in 3/4 at the old 0.15 too - noisy, mostly-negative either way, and this
  project's own earlier "explained_variance starts at ~0.65 with warm-start" claim
  (`ppo_warm_start.py`) does not reproduce under either setting on the current encoder.
  Decision: keep 0.05 for the next run anyway - the mathematical problem it fixes
  (shaping outweighing the terminal win/loss signal) is confirmed and structural, the
  warm-start-quality concern is unconfirmed and, per the original gradient-coupling
  study, this class of effect has previously self-corrected within ~4 of ~244 updates in
  a 500k-step run - but explicitly flagged here as a real open question to revisit if the
  next run underperforms again, not silently resolved.
- Everything else in the diff checked and confirmed correct, no bug: `MixedOpponentPlayer`
  weighted selection/battle_tag caching, `TwoPlySearchPlayer`'s use as a non-connected
  decision function, `self_play_opponent` reference-correctness through the
  `MixedOpponentPlayer` wrapper, `--resume-from`'s interaction with the new opponent code,
  `VECTOR_LEN` arithmetic and `encode()`'s shape assertion, the `[-1]`->`[-3]` test
  reindex, and that the new tests are non-vacuous (mutation-tested the same way the
  2026-08-01 review required of `FrozenPolicyPlayer`'s own resampling test).

Also separately bumped `--n-steps` 256 -> 2048 (stable-baselines3's own default,
restored after being left at an 8x-smaller dev-speed shortcut through every real
training run so far, including both plateaus) - `batch_size` stays at SB3's default
(64), so this is still a clean 32 minibatches/epoch at `n_envs=1`.

Next: the actual combined run (reward rebalance + hazard-immunity fix, corrected for
boots + n_steps=2048) - not yet done as of this note. `--eval-interval 20000` recommended
over the default 4096 (measured: the default schedule would add ~2,440 real eval battles
on top of a 500k run, likely dominating its wall-clock).

### Combined run + overnight extension: Phase 3 gate met (2026-08-09/10)

The combined run (reward rebalance + boots-corrected hazard-immunity fix + `n_steps=2048`,
all three together for the first time) ran as a fresh 500k-step sanity check
(`--eval-interval 1000000 --eval-battles 20` after the first check confirmed `n_steps=2048`
throughput ~642 steps/s, no regressions): pooled 147/500 = 29.4% over its first 500k -
ambiguous relative to the 34.2% pre-boots-fix benchmark (which had double the training),
roughly in the old plateau band. Resumed to 1,000,000 total steps: second-half pooled
234/500 = 46.8%, a real jump within the run itself (individual points up to 75%, the
highest single point this project to that date). The real 500-battle gate benchmark at
1,000,000 steps: **204/500 = 40.8% [36.6%, 45.2%]** - CI entirely clear of the historical
~28-32% plateau band (confident, real escape from it), and higher than the prior 34.2%
result though with a thin CI overlap (36.6-38.5%) so not itself airtight on its own.

Given a real, improving trend and cheap-so-far run costs (~15-20 min per 500k on this
laptop), committed to a ~10-hour overnight extension: resumed from the 1,000,000-step
checkpoint to 22,000,000 total (21,000,000 more steps), `--eval-interval 1000000
--eval-battles 20 --checkpoint-interval 250000`. System kept awake via `caffeinate -s
-w <training PID>` (no `-d`, so the display could still sleep normally) rather than
GUI tools, specifically because it ties caffeinate's lifetime directly to the training
process - it exits automatically the moment training stops, whether that's early
termination or natural completion, with no separate cleanup step needed. Monitored via
hourly scheduled check-ins (not continuous polling) with explicit instructions not to
react to any single flat/bad-looking hour - only a sustained multi-hour trend or a real
pathology (errors, throughput collapse, degenerate behavior) would have warranted early
termination. Neither occurred: all 10 hourly checks showed healthy, steady throughput
(570-579 steps/s the entire night, no thermal-throttle signs on this fanless laptop) and
a win rate that stayed consistently well above the pre-overnight baseline throughout,
trending upward in the second half if anything (individual eval points reached 90% and
80% in the final two checkpoints).

The run completed the full 22,000,192 steps cleanly (570.6 steps/s sustained average,
~10.2 hours), `caffeinate` confirmed self-released on process exit as designed. Pooling
all 21 periodic eval points from the overnight run (2,000,000 through 22,000,000 steps,
20 battles each): 256/420 = 61.0% - first half (2M-12M) pooled 54.5%, second half
(13M-22M) pooled 68.0%, a real continued climb within the overnight run itself, not
just a one-time jump. The real 500-battle gate benchmark on the final checkpoint:

**348/500 = 69.6% [65.4%, 73.5%] vs `TwoPlySearchPlayer`.**

CI entirely clear of both the 40.8% pre-overnight benchmark and the original ~28-32%
plateau - the first time this project has produced a real, statistically confident,
decisive win (not just an edge) for PPO over the Phase-1 search bot.

**Direct Phase-2-bot benchmark (2026-08-11)**: `--p1 ppo --p2 learned --format gen9ou
--n-battles 500` (same final checkpoint) - **346/500 = 69.2% [65.0%, 73.1%] vs the
Phase-2 learned bot**, confirming the roadmap's Phase 3 gate ("RL-tuned policy beats
the Phase-2 supervised bot head-to-head") directly rather than by the transitivity
inference this note originally relied on. Near-identical to the 69.6% vs-`search`
number above, consistent with Phase 2's own bot having lost to `search` 64.4% of the
time (35.6% win rate) - PPO dominates both Phase-1 and Phase-2 bots by roughly the
same margin. **Phase 3 gate: met, directly measured.**

**Replay-inspection re-check (2026-08-11)**: `scripts/inspect_ppo_replays.py
--n-battles 8` against the same final checkpoint, to verify (not just infer from the
aggregate win rate) that the previously-diagnosed pathologies are actually gone.
Confirmed clean on all three: protect-spam gone (7 total protect uses across 8
battles, `protect_counter` never exceeded 2, always followed by a different move -
none of the old escalating-failure streaks); Stealth Rock recast gone (2 uses total,
no repeat pattern); Spikes-vs-hazard-immune-opponent not observed at all. No
stalling/switch loops either (all 8 battles ended in 15-43 turns, far under the old
multi-dozen-turn loops and the 1000-turn auto-tie). One long same-move streak (8x
Roost) was inspected directly rather than assumed pathological - HP trended upward
(0.18->0.60) against a wall it couldn't break, i.e. legitimate stall-recovery play,
not a failure loop.

**Ladder script built (2026-08-11)**: `scripts/ladder_ppo.py` plays the trained
checkpoint on the real Showdown server (`ShowdownServerConfiguration`, not the local
dev server every other script defaults to) via `Player.ladder()`, for the roadmap's
other Phase 3 gate component - a real ladder GXE number. Checked poke-env's actual
client source (0.15.0) before building it: `log_in()` only does a real
password-authenticated login when a password is supplied (no bot-account-creation
call exists anywhere in the client - a real, already-registered account is a
prerequisite, not something this script can set up). Also checked Showdown's own
rules page, FAQ, and its linked Bot FAQ gist: **no codified "battle bots must
register/use an alt account" policy was found** - the alt-account approach from this
project's Hard Rules is a self-imposed precaution (keeping an experimental bot's
record off any personal identity), not a documented compliance requirement. Password
handling avoids the plaintext-CLI-arg pitfall (shell history/process listing
exposure): reads `POKE_SHOWDOWN_PASSWORD` or falls back to a secure `getpass` prompt.
Not yet run - needs a real registered alt account, which is a manual step for the
user (poke-env can't create one), and real ladder games are real-time against real
humans (materially slower than every other benchmark in this project), so `--n-games`
should start small.

**First real runs (2026-08-11), and a genuine unresolved stall**: registered
`battle-engine-test`, credentials stored in a gitignored `.env.local` (auto-loaded by
the script - never in memory/CLAUDE.md, and never as a CLI arg). A 5-game serial run
(`--max-concurrent-battles 1`, the default) completed cleanly: 2/5 won. A follow-up
40-game run at `--max-concurrent-battles 5` (poke-env's own real
`Player(max_concurrent_battles=...)`, verified safe to raise for this script's
specific setup - `FrozenPolicyPlayer.choose_move` has no `await` in it and
`load_ppo_player` seeds exactly one policy snapshot, so concurrent battles can't
corrupt each other's move computation or mix up which weights are used) showed real
initial progress (the account's live rating moved: Elo 1068->1000, GXE 27.2%->25.0%
over the first ~40 min) then went **completely flat for 25+ minutes** (identical
Elo/GXE/deviation across two checks) with near-zero CPU time, while the process
stayed alive with its one real connection to `sim4.psim.us` still established.

Initial diagnosis (this session, no stack trace available - `py-spy dump` needs root
on macOS and no TTY was available anywhere in the tooling to supply a `sudo`
password) was inconclusive. A follow-up Opus code review (given the exact symptom
and the poke-env source snippets already found) resolved it further, reading
poke-env's actual source plus - critically - the local `pokemon-showdown` checkout's
real server-side code, not just the client library:

1. **A real, confirmed bug in poke-env's own `_ladder`**: it fires the *next* ladder
   search before checking whether it's already at `max_concurrent_battles` capacity
   (`player.py`'s loop calls `search_ladder_game` before the `while
   self._battle_count_queue.full()` check, not after) - so at
   `--max-concurrent-battles 5`, once 5 battles are live, it still searches for a 6th.
2. **The real Showdown server enforces a hard 5-concurrent-battles-per-IP limit**
   (confirmed in the local checkout's `server/monitor.ts`,
   `countConcurrentBattle`) - a 6th search gets silently rejected via a `|popup|`
   message that poke-env only logs as a warning (`ps_client.py`) and never retries.
   `_ladder`'s wait on the next battle-start signal has no timeout, so a
   silently-rejected search is a **permanent, zero-CPU stall with the connection
   still alive** - exactly what was observed.

Together: running at exactly the server's own concurrency ceiling (5) very likely
triggered poke-env's off-by-one bug to push one search past that ceiling, which the
server silently dropped with no recovery path on the client side. Also flagged by the
review as plausible but explicitly *not* provable from source alone: a possible
lock-holding interaction between `_battle_start_condition` and `_battle_end_condition`
in the same `_ladder` loop body. Checked and reasonably ruled out: our own
`FrozenPolicyPlayer.choose_move` (fully synchronous, no `await`) can't itself cause a
stall like this, though an uncaught exception inside it would produce an
indistinguishable zero-CPU signature - not confirmed either way here.

**Practical takeaway**: stay at `--max-concurrent-battles 1` (proven clean twice) for
the actual GXE-gathering runs. Real concurrency support would need fixing poke-env's
search-before-capacity-check ordering and handling `popup`/`updatesearch` messages it
currently ignores - real, deferred work, not needed for this gate. **This project had
never used `max_concurrent_battles > 1` anywhere before this run** - every existing
script/test uses the default of 1, so this was genuinely uncharted territory, not a
previously-trusted path breaking.

Also discovered and fixed along the way: the empty-looking log during the entire
stall turned out to be a red herring, not evidence either way - Python fully buffers
stdout when piped through `tee` (confirmed the hard way: even the final SIGTERM'd
exit left the piped log completely empty, since SIGTERM skips normal interpreter
shutdown/flush). Fixed properly, not worked around: `_instrument_progress()` in
`ladder_ppo.py` now prints a flushed, real-time line on every battle start (wraps
`choose_move`, same technique `inspect_ppo_replays.py` already uses) and every
battle finish (overrides `Player._battle_finished_callback` - a real, public no-op
hook meant to be overridden, not a private/internal detail), so any future run is
self-diagnosing live instead of requiring this whole external rating-polling/`py-spy`
detour again.

Root cause confirmed (Opus review, reading poke-env's source plus the local
`pokemon-showdown` checkout's real server code, not just the client library): a real
poke-env bug (`_ladder` searches for the next battle before checking capacity)
combined with the real server's hard 5-concurrent-per-IP limit - running exactly at
that ceiling let the off-by-one push one search past it, which the server silently
rejected with no retry path on poke-env's side. Since the overshoot is always exactly
+1, `--max-concurrent-battles 4` is safe from this specific mechanism (overshoot lands
at 5, still within the server's limit) without needing to patch poke-env itself -
confirmed by then actually running at 4 with the new live logging: clean, steady
progress the entire run, zero stalls.

**Retest at `--max-concurrent-battles 4` (2026-08-11) - real ladder GXE obtained,
Phase 3's second gate component now directly measured, not just planned.** Two runs,
exact counts both times thanks to the new logging: the original 5-game serial test
(2-3-0) plus a clean 40-game/4-concurrent run (14-26-0, 35.0%). Combined: **16-29-0
across 45 real games, 35.6% win rate [23.2%, 50.2%] (Wilson 95% CI)**, cross-validated
by the account's own converging rating (deviation tightened ±52 -> ±39 as more games
landed; final GXE 26.3%, Glicko 1305 ± 39). The CI's upper edge barely touches 50%,
so this isn't airtight proof of a sub-50% true rate, but every point estimate across
both runs (40.0%, 35.0%, 35.6% combined) independently landed in the same 35-40%
band - a real, converging signal, not noise. Also directly benchmarked the roadmap's
other Phase 3 gate wording ("beats the Phase-2 supervised bot head-to-head") rather
than relying on the earlier transitivity inference: **346/500 = 69.2% [65.0%, 73.1%]
vs the Phase-2 learned bot** (`--p1 ppo --p2 learned --format gen9ou`), confirming it
directly.

**Honest read**: the trained PPO policy decisively beats both prior phases' bots in
local head-to-head benchmarks (69.6% vs Phase-1 search, 69.2% vs Phase-2 learned) but
is currently the underdog against the real gen9ou ladder population it was matched
against (35.6%, GXE 26.3%) - a genuinely different and harder test than bot-vs-bot
benchmarking, and exactly the kind of gap this deliverable exists to surface rather
than paper over. Not yet diagnosed *why* it's losing on the real ladder specifically
(no replays saved this run, `--save-replays` exists but wasn't used) - a real,
deliberately-deferred next step if this axis is revisited, not attempted here.

**Phase 3 status: complete.** Both roadmap gate components now have real, directly-
measured numbers (not inferred, not aspirational): beats the Phase-2 bot head-to-head
(69.2%) and a real ladder GXE number (26.3%) exist and are recorded above. Moving to
Phase 4 (stretch: C++ search core) next.

ppo.zip and the checkpoint chain from this run are the current, non-archived
`data/models/` state as of this note.

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
# protect-spam pathology behind Phase 3's win-rate plateau (see Status above).
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
  + benchmark-facing PPO loader (`ppo_eval.py`)
- `scripts/` — runnable entry points (smoke test, benchmarks, replay fetching,
  dataset building, training, PPO training, PPO replay diagnosis
  `inspect_ppo_replays.py`, real-ladder play `ladder_ppo.py`)
- `tests/` — pytest (state encoding, damage calc, dataset/action-label logic, model
  training loops, harness determinism w/ seeded RNG, action-space translation, the
  PPO env — including one real-server integration test — PPO warm-start weight
  transplant, self-play, PPO eval/benchmark loading)
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
