"""Fixed-size battle-state encoding — the core Phase-2 lesson (see project
roadmap): design a vector representation of a battle state, unit-test it hard.

Two adapters map either a live poke-env `AbstractBattle` or one turn's
`state` dict from a Metamon parsed replay into a common intermediate
`BattleView` (this project's own small dataclasses, not poke-env's or
metamon's), then `encode()` walks that one shape into a fixed-length numpy
vector. Same "one function, two adapters" shape `search.py`'s
`_project_after_action` already uses for live-vs-simulated states.

The replay schema below was verified against a real downloaded file (see
scripts/fetch_replay_sample.py), not assumed from metamon's documented
UniversalState/ReplayState field names — those turned out not to match the
raw .json.lz4 layout 1:1 (e.g. no top-level `turnlist`/`winner`; a replay
file is actually `{"states": [...], "actions": [...]}`, each state already
POV-relative and pre-flattened: `hp_pct` not raw HP, `base_atk`/`atk_boost`
as separate scalars, space-separated type strings with a literal "notype"
placeholder for single-typed Pokemon).

Corrections from an independent review (2026-07-30), after this module had
already been built and unit-tested once — both caught by checking real
downloaded replay data and poke-env's actual source, not by inspection alone:

- **My-side bench, fainted teammates**: Metamon's per-state
  `available_switches` silently drops fainted teammates entirely (verified:
  0 of 3247 real switch-list entries are fainted) — Showdown just stops
  offering them as a switch target. A single replay `state` therefore *looks*
  identical whether a teammate fainted or was simply never brought, which is
  wrong: `battle_view_from_replay_state` (single-state, kept for
  ad-hoc/debugging use) still has this gap, but `battle_views_from_replay`
  (state**s**, plural — use this for real work, e.g. `dataset.py`) tracks a
  running per-replay roster and reconstructs fainted teammates as
  known-but-fainted slots, matching what `battle_view_from_poke_env` already
  gets for free from poke-env's `battle.team` (which keeps fainted mons).
- **Bench slot ordering**: both adapters now sort bench slots by species name.
  Replay data's `available_switches` order isn't stable — measured 10.1% of
  consecutive same-replay states reordering a teammate with no faint
  involved — so a fixed slot index carried no consistent identity before
  this; poke-env's own team-dict order isn't guaranteed stable either.
  Sorting by species name (available on both sides, doesn't change turn to
  turn) fixes both.

Named simplifications (same convention as damage.py/evaluation.py):
- Opponent bench detail: only the opponent's current active Pokemon is
  encoded in full; the rest of their team is summarized as a single "fraction
  still alive" scalar (`opponents_remaining` in the replay data), not
  per-mon slots. Unlike the my-side fainted-teammate gap above, this one
  isn't reconstructable even with the full turn sequence: a replay state
  doesn't carry species/stats for opponent Pokemon that aren't currently
  active, only a remaining-count and a teampreview species list. A live
  poke-env battle actually knows more than this (accumulates revealed
  opponent-team detail turn over turn), so this is a real, deliberate
  asymmetry, not a wash.
- Hazards report only the single most-recently-changed condition per side,
  not every condition simultaneously active, and not stack count. This was
  originally meant as "presence-only, for parity between the two adapters" —
  review found that claim was actually false: real replay data's
  `player_conditions`/`opponent_conditions` field is single-valued and
  overwritten on every new hazard/screen event (verified: 267 states show
  `stealthrock`, 86 show `spikes`, zero show both, despite that being a
  completely ordinary simultaneous board state). `_poke_env_hazards` now
  deliberately narrows the live side to match — real fidelity a live battle
  could otherwise provide, given up on purpose so training data and
  live-inference data mean the same thing. A second review pass fixed a
  real (not deliberate) gap in this: Aurora Veil and Tailwind weren't in
  `_HAZARD_TOKENS` at all, so both adapters silently ignored them even
  though poke-env tracks their turn number exactly like Stealth Rock/
  Reflect/etc. That same pass also found the *original* justification for
  narrowing at all — "no removal signal for Defog/Rapid Spin/screen-expiry"
  — is factually shaky: the field does revert to `noconditions` on a clean
  sweep (e.g. `stealthrock→noconditions` 277×, `auroraveil→noconditions`
  25× across a 600-replay sample), so an add-on-new-token /
  clear-on-noconditions reconstruction may be more tractable than assumed
  when this simplification was first chosen. Not attempted here — revisit
  if it turns out to matter for the trained model, since it's a real
  question, not something to quietly redecide unilaterally.
- Terrain has the same single-valued masking as hazards, for the same
  reason (`battle_field` is one token, and its real vocabulary includes
  non-terrain entries: `trickroom`, `gravity`). `_poke_env_terrain` ranks
  all of `battle.fields` by turn and only reports the winner if it's
  actually a terrain, matching what a replay-derived state can show.
  Trick Room/Gravity themselves aren't modeled as their own feature -
  same "not attempted, revisit if it matters" status as the hazard question
  above, not a design decision made either way yet.
- Level is omitted: every replay here is a standard, fully-leveled (100)
  format, so it carries no signal.
- Abilities aren't a dimension of their own. Ability identity is asymmetric
  info in real play (always known for your own Pokemon, `None` until
  revealed for the opponent's — poke-env already models this via
  `mon.ability`), and gen 9 has a few hundred distinct abilities, too many
  for a one-hot in a hand-built vector. Rather than an identity feature, the
  known subset of *type-immunity* abilities (Levitate, Water Absorb, Flash
  Fire, Wonder Guard, ...) is folded directly into the active-vs-active
  matchup-score dimension via `_TYPE_IMMUNITY_ABILITIES` — the same
  "hand-engineered prior over raw features" reasoning already used for the
  type chart itself. Abilities with non-type-effect consequences (Intimidate,
  Speed Boost, Protean, Multiscale, Unaware, ...) aren't modeled at all;
  representing those well is a job for a learned ability embedding once an
  actual model exists, not something to hand-derive here.
- `my_active_hazard_immune`/`opp_active_hazard_immune` (`_is_hazard_immune`,
  2026-08-09): whether each active Pokemon is immune to ground-based entry
  hazards (Flying-type, Levitate, and Heavy-Duty Boots are immune to
  Spikes/Toxic Spikes/Sticky Web; Boots additionally blocks Stealth Rock —
  independent of the type chart / `_TYPE_IMMUNITY_ABILITIES` above, which is
  about damaging-move effectiveness, not hazards). Added after real replay
  inspection of a trained PPO policy (see CLAUDE.md's Phase 3 status) caught
  it spamming Spikes 32 turns straight against a Flying-type opponent it
  could never affect — nothing in the vector could tell it why. Originally
  Flying/Levitate-only (`_is_grounded`); an independent review the same day
  measured Heavy-Duty Boots as the #2 most common immunity source (5.19% of
  a 20,610-state sample, ahead of Levitate's 2.46%) and the only one that
  also blocks Stealth Rock — the first version confidently mislabeled every
  boots holder as vulnerable. Fixed and renamed (see `_is_hazard_immune`'s
  own docstring for the full accounting, including what's still deliberately
  out of scope — Iron Ball/Gravity/Magnet Rise/Ingrain/Smack Down/Roost).
  Hazard stack count (see the hazards bullet above) is a separate, still-open
  gap this doesn't address — the observed pathology was 100% about immunity
  (the very first Spikes use already targeted an immune opponent), not the
  cap.

Real gap found 2026-08-01, diagnosing Phase 3's PPO win-rate plateau against
the search bot (replay inspection again, same technique that found the
Phase-1 switch bug and Phase-2 damage-calc bugs): a real, quantifiable
pathology, not just noise — the trained policy repeatedly re-used a
protect-counter move (Protect, Endure, ...) right after it had just failed,
walking back into Showdown's own escalating-failure mechanic (each
consecutive use drops the success chance geometrically) and taking large,
predictable HP losses as a direct result (verified turn-by-turn: HP crashing
from 1.00 to 0.54 immediately after a 2nd consecutive Protect, in a battle
the policy went on to lose). This is a genuine encoder gap, not a training-
config problem: poke-env already tracks the exact mechanic per-Pokemon
(`Pokemon.protect_counter`, incremented on a successful protect-counter
move, reset to 0 on a failed one, a non-protect move, or a switch-out) but
nothing here was reading it, so no model trained on this vector could ever
learn "don't protect again, it's about to fail" — the information needed to
make that call was structurally invisible to it.

`PokemonView.protect_counter` (an `int`, encoded as a single normalized
scalar — see `_PROTECT_COUNTER_SCALE`) fixes this:
- Live adapter: `mon.protect_counter` directly — exact, poke-env's own
  tracked value, correct for active and benched Pokemon alike (poke-env
  itself already resets it to 0 on switch-out, so reading it off a bench
  slot needs no special-casing).
- Replay adapter: no equivalent field exists in a Metamon replay state (only
  `player_prev_move`/`opponent_prev_move` — see below for what these
  actually mean) — reconstructed across a whole replay's turn sequence the
  same way fainted teammates already are (`battle_views_from_replay`, not
  the single-state function, which can't do this and documents 0 as its
  gap the same way it already does for fainted teammates). A deliberate,
  verified simplification versus poke-env's exact semantics: increments on
  any consecutive same-side use of a protect-counter move by the same
  active Pokemon (tracked via species-identity continuity, resetting the
  streak on any switch), without attempting to detect success vs. failure.
  Faithfully reconstructing the real failure-reset would need inferring
  "did this Protect actually block the hit" from hp_pct deltas alone, which
  is confounded by residual status damage (burn/poison/toxic keep ticking
  through a *successful* Protect) — trading one approximation for a
  noisier one, so deliberately not attempted.

  **A real bug in this reconstruction, found by independent review
  (2026-08-01), after the feature had already been built, tested, and used
  to retrain both Phase-2 models once**: `player_prev_move`/
  `opponent_prev_move` are NOT "the move used on the transition into this
  state" — they're "the last move this Pokemon has ever used," carried
  forward byte-for-byte (name AND current_pp AND every other field)
  unchanged across states where that side didn't actually act (Metamon
  emits extra decision states, e.g. for the opponent's forced switch after
  a KO, where the player's own side did nothing at all). The original
  version had no way to tell a genuine second consecutive protect from a
  stale carried-over one — both showed the name "protect" — so it kept
  incrementing an already-resolved streak. Measured against real per-move
  PP as independent ground truth (a move's own `current_pp` only drops on
  an actual use): 59.4% of reconstructed streak-2 values and 41.7% of
  streak-3 values were wrong as a direct, quantified result. Fixed by
  comparing each side's whole `prev_move` dict to the previous state's: if
  identical, nothing happened for that side this transition, so the
  PREVIOUS streak value carries forward unchanged (neither incremented nor
  reset) rather than being read as a fresh use - a signal available
  directly from `states` with no need to thread the replay's separate
  per-state action labels through this function. Re-verified against
  independent moveset-PP ground truth after the fix: 1,238/1,242 (99.7%)
  of streak>=2 reconstructions now correspond to a real PP drop (up from
  the pre-fix ~41-59% error rate) across a 3,000-replay sample.

  The original numbers quoted here as "verification" were themselves
  corrupted by this bug and have been corrected: re-run after the fix,
  across a 3,000-replay/91,385-state sample, a nonzero streak occurs on
  1.34% of (state, side) pairs (matches the original 1.35% claim, though
  that number was actually the nonzero-streak rate, not "protect-counter
  moves as a fraction of states," a mismatch the review also caught), the
  longest streak seen is now 2 (not 3, and the false 4s/5s the bug produced
  in the actual training dataset are gone), and streak>=2 correlates with
  that Pokemon fainting the same turn only 11.4% of the time (70 total
  instances, 8 at a faint) — NOT "every instance," which was the review's
  most important catch: the original "always ends in a faint" claim was
  reading its own bug's artifact (the post-faint phantom state) as
  independent confirmation, not measuring the real underlying pattern at
  all. The feature is still worth having — genuine multi-use protect
  streaks are real and rare in human play, matching the live adapter's
  exact semantics — but the original "validates itself" framing was
  circular, not evidence.

Real gap found via a real-ladder diagnostic (20 games, 2026-08-26 - see
notes/): the trained PPO policy repeatedly used moves that were type-immune
against the opponent (Dragapult using Draco Meteor into Clefable four turns
straight, immune every time). `MoveSummary`'s aggregate move-type coverage
told the model "this moveset covers Dragon type" but never "is THIS specific
move, right now, actually going to do anything to what's in front of it" -
that per-slot signal didn't exist anywhere in the vector. `MoveView` (below)
and `_move_slots_vector` fix this: each known Pokemon (not just the active
one) gets up to `MAX_MOVES` per-move-slot feature blocks, computed inside
`encode()` (not at PokemonView-construction time, since a move's real-time
effectiveness needs the DEFENDING side's current types/ability/item too -
`_encode_pokemon` now takes a `defender: PokemonView` argument for exactly
this reason). `MoveSummary`/`_move_summary_features` are left as-is
alongside this, not replaced - they're still a real, distinct switch-safety
signal (has_recovery/has_hazard_setup/etc.), and Phase 1's scope is fixing
the missing per-slot gap, not re-deriving the existing aggregate from it.

`_type_multiplier` is generalized (still the single source of truth for
type-chart+ability immunity, now also handling two item-based exceptions)
rather than duplicated - the same "generalize an existing, already-verified
function to a new call site" approach the plan for this rewrite calls out
explicitly. Air Balloon (Ground immunity while held) is a direct addition:
`defending_item == "airballoon"` zeroes out a Ground-type attacker's
multiplier, approximated as "currently holding" with no turn-scoped
"already popped this turn" tracking (`PokemonView` has no such state) -
documented the same way `_is_hazard_immune`'s own Iron Ball/Gravity
exclusions are.

Ring Target needed real verification before encoding (flagged as this
phase's own Uncertainty note) - "cancels a type immunity" is easy to get
subtly wrong. Read Showdown's own current sim source directly
(`sim/items.ts`'s `ringtarget` entry, `sim/pokemon.ts`'s `runImmunity`/
`isGrounded`, `sim/dex.ts`'s `getEffectiveness`), not assumed from memory:
- Ring Target's `onNegateImmunity: false` handler makes
  `Pokemon.runImmunity`'s `negateImmunity` true for its holder, which
  bypasses ONLY the hard-coded `hasType('Flying')` check inside
  `isGrounded` (Ground-vs-Flying-typing) and, for every other attacking
  type, skips straight past `dex.getImmunity`'s type-chart-based immunity
  gate. It does NOT touch the separate Levitate/Air-Balloon/Magnet-Rise
  checks in the same function (none of those branches read
  `negateImmunity` at all) - so Ring Target cancels TYPE-CHART immunities
  only (Ghost's immunity to Normal/Fighting, Flying's to Ground, Steel's to
  Poison, Dark's to Psychic, Normal's to Ghost, ...), never
  ability-granted ones (Levitate, Water Absorb, Wonder Guard, ...) - those
  are separate code paths Ring Target's handler never touches. (Air Balloon
  and Ring Target can never co-occur on one Pokemon regardless - both are
  held items, one slot.)
- `dex.getEffectiveness` computes each defending type's contribution
  independently and multiplies them together (verified empirically too:
  `PokemonType.damage_multiplier(d1, d2, ...)` on real poke-env data equals
  `damage_multiplier(d1) * damage_multiplier(d2)` for every real
  attacking/defending-pair combination, 0 mismatches across the full type
  chart) - and a type-chart "immune" (0x) contribution is stored as
  exponent 0 (neutral), the SAME representation as an ordinary neutral
  match. This confirms the plan's own Uncertainty note precisely: Ring
  Target turns one type's own 0x contribution into a neutral 1x, while the
  OTHER defending type's real resistance/weakness still multiplies in
  unaffected - it does not create a new weakness, and a dual-typed immune
  Pokemon doesn't become "generically hittable," only that one type's
  immunity is gone. Implemented in `_type_multiplier` by computing each
  defending type's multiplier separately and only overriding a 0.0 result
  to 1.0 when Ring Target is held, then multiplying the (possibly
  overridden) per-type results together - not by calling
  `damage_multiplier(d1, d2)` as one combined lookup, which has no way to
  attribute the 0x to one type or the other.

`_active_matchup_score` (the pre-existing active-vs-active aggregate
dimension) is updated to pass each side's item through the same
generalized `_type_multiplier` too, not just the new per-move code path -
no currently-passing test exercises Ring Target/Air Balloon there, so this
changes no existing test's result, and leaving the older call site on a
stale, less-correct version of the same shared function while the new one
uses the fixed version would be an inconsistency with no upside.

Phase 2: `MoveView` gains four more movedex-static fields (`bypasses_protect`,
`recoil_fraction`, `drain_fraction`, `is_self_ko`), and `PokemonView` gains
three runtime-state fields (`preparing`, `semi_invulnerable`,
`must_recharge`) - the move-mechanical properties that change whether using
a move right now is a good idea, beyond raw damage/type effectiveness (the
same axis Phase 1's per-move type-effectiveness gap was on, just a
different failure mode - a Pokemon dodging behind Fly/Dig invulnerability,
or one that's about to be stuck recharging, is exactly as invisible to a
2026-08-26-diagnosed model as an immune move target was).

Each new `MoveView` field was verified against `GenData.from_gen(9).moves`
before coding, not assumed:
- `bypasses_protect = not bool(entry["flags"].get("protect"))` - Feint's
  real flags dict (`{'failcopycat': 1, 'mirror': 1, 'noassist': 1}`) lacks
  the `protect` key entirely; Tackle's has it. A self/side-targeted move
  (Agility, Baton Pass, ...) also lacks the key - correctly reads as
  bypasses_protect=True too (Protect genuinely can't block a move that
  never targets the opponent), not a bug, same "real static signal even for
  a non-opponent-directed move" tier as `is_contact` etc.
- `recoil_fraction`/`drain_fraction`: real dex shape confirmed as
  `[numerator, denominator]` - `flareblitz.recoil == [33, 100]`,
  `gigadrain.drain == [1, 2]`, `tackle` has both `None`. Stored as
  `num/denom`, already a natural ~0..1 fraction, no extra scaling needed.
- `is_self_ko`: the plan flagged this as possibly needing a hardcoded list
  (same pattern as `_HAZARD_REMOVAL_MOVES`) - it doesn't. Real dex entries
  carry a declarative `selfdestruct` field (`"always"` for
  Explosion/Self-Destruct, `"ifHit"` for Memento/Final Gambit/Healing
  Wish/Lunar Dance), verified directly - `is_self_ko =
  bool(entry.get("selfdestruct"))`.

Semi-invulnerable-vs-merely-charging classification, verified against the
real local `pokemon-showdown/data/moves.ts` checkout (not poke-env's
trimmed `GenData`, which strips the `condition`/`onTryMove`
simulator-logic fields this needs): Fly/Dig/Dive/Bounce/Phantom
Force/Shadow Force each carry a `condition: { duration: 2,
onInvulnerability: ... }` block; Solar Beam/Sky Attack/Skull Bash/Freeze
Shock/Ice Burn/Meteor Beam/Electro Shot/Geomancy/Razor Wind carry no
`condition` key at all. Phantom Force/Shadow Force's `onInvulnerability:
false` is a literal boolean, not a callback like the other four's (which
have named exceptions - Earthquake still hits Dig, Surf still hits Dive,
Gust/Twister/Thunder/Hurricane/Smack Down/Thousand Arrows still hit
Fly/Bounce) - traced `sim/battle.ts`'s `runEvent`: a non-function handler
value is used as the event's return value directly (`else { returnVal =
handler.callback; }`), so `onInvulnerability: false` unconditionally
returns `false` (invulnerable, no exceptions) against every incoming move -
confirming Phantom Force/Shadow Force genuinely are semi-invulnerable, in
fact stricter than the other four. Confirms the plan's exact six-move list
with no exceptions found; hardcoded as `_SEMI_INVULNERABLE_CHARGE_MOVES`,
same short-verified-list pattern as `_HAZARD_REMOVAL_MOVES`/
`_ONHIT_SETUP_MOVES`.

`PokemonView.preparing`/`semi_invulnerable`/`must_recharge` are runtime
state, not movedex-static, so they live on `PokemonView` (not `MoveView`)
per this phase's own scope. Placed in `_encode_pokemon`'s concatenation
BEFORE `protect_counter`, not after, specifically so `protect_counter`
stays the LAST field of a Pokemon's block -
`test_protect_counter_encodes_as_a_normalized_scalar_and_clamps` asserts
`vec[-1]` directly (a documented layout contract from Phase 0, not
incidental), and this ordering choice means that already-passing test
needs no changes. Live adapter reads `mon.preparing` (poke-env's own public
property, already correctly `bool(preparing_target) or
bool(preparing_move)` - used directly rather than hand-rolling the same OR
from the two private-backed properties), `mon.preparing_move.id in
_SEMI_INVULNERABLE_CHARGE_MOVES` (guarded on `preparing_move is not None`),
`mon.must_recharge` directly - all three exact, no reconstruction needed,
unlike `protect_counter`.

Replay-adapter parity was an open plan uncertainty - resolved, not left
open. Checked a real downloaded replay sample directly (not assumed from
Metamon's docs), per this project's evidence-over-assumption rule:
- No top-level replay-state field resembles `preparing_move`/
  `must_recharge` (the real field set is `available_switches,
  battle_field, battle_lost, battle_won, can_tera, forced_switch, format,
  opponent_active_pokemon, opponent_conditions, opponent_prev_move,
  opponent_teampreview, opponents_remaining, player_active_pokemon,
  player_conditions, player_prev_move, weather`).
- Each side's `*_active_pokemon` dict does carry an `effect` field
  (single-valued, same single-valued-masking convention as the existing
  hazards/terrain fields) - its full vocabulary was scanned across a
  400-replay/~20,000-sample pass (real volatile statuses like
  `quarkdrivespe`/`protect`/`fallen`/`saltcure`/`substitute`, nothing
  charge/invulnerability/recharge-shaped) and again across 3,000 replays
  hunting specifically for `recharge`/`fly`/`dig`/`dive`/`bounce`/
  `shadowforce`/`phantomforce`/`solarbeam`/`twoturn`/`invuln`/etc.
  substrings - zero matches either time.
- Went one step further than a vocabulary scan: found real replay states
  where `player_prev_move`/`opponent_prev_move` shows a genuine charge-move
  use (`meteorbeam`, `phantomforce`, `dig`) and inspected `effect` on that
  exact state and the following one - `noeffect` in every case. The gap
  isn't "rare enough to miss in a sample," it's structural: Metamon's
  replay schema does not track this mechanic at all.
- Documented as a live-adapter-only feature, defaulting to
  `False`/`False`/`False` on `battle_view_from_replay_state`,
  `battle_views_from_replay`, and `PokemonView.unknown()` alike - same
  explicit-gap convention this module already uses for opponent bench
  detail and hazard recency.

Phase 3 (2026-08-26): weather/terrain-conditional move behavior - the
subset of weather/terrain interactions that change whether a move is good
RIGHT NOW, cross-referencing individual moves/abilities against the
battle's actual current weather/terrain (already encoded at the side level
via `_WEATHER_NAMES`/`_TERRAIN_NAMES`, per this phase's own scope, just not
cross-referenced against anything before now). Every fact below was
verified against the real local pokemon-showdown/data/moves.ts and
data/abilities.ts checkout (poke-env's trimmed GenData strips the
onModifyType/onBasePower/onTryMove/onModifySpe callback fields this needs,
same reason Phase 2's semi-invulnerable-move verification needed the same
checkout), not assumed from memory or poke-env's dex alone:

- Weather Ball (`_WEATHER_BALL_TYPE_BY_WEATHER`): real dex entry
  (`onModifyType`/`onModifyMove`) is Normal/50BP by default, switching to
  Fire/Water/Rock/Ice and doubling base power under sun/rain/sand/snow
  respectively (verified directly - poke-env's GenData entry only shows the
  static Normal/50 fallback, confirming this needs a short hardcoded table,
  same _HAZARD_REMOVAL_MOVES-style pattern). Computed in `_move_slot_vector`
  (needs live weather, like effectiveness/STAB already need live defender
  state) as an override of the move's effective type/power BEFORE the
  existing type one-hot/STAB/effectiveness/base_power scalars are computed
  from it - no new vector dimensions, this phase's weather table just
  changes what those existing fields compute.
- Solar Beam/Solar Blade (`_SUN_CHARGE_SKIP_MOVES`, the new
  `MoveView.is_charge_move` static field, and the new `needs_charge_turn`
  per-move-slot scalar): both moves' real `onTryMove` checks
  `['sunnyday', 'desolateland'].includes(...)` and skips the charge turn
  entirely - verified as the ONLY two charge moves with this check (grepped
  every `desolateland` occurrence in moves.ts; the other hits are
  Growth/Moonlight/Morning Sun/Synthesis, unrelated weather-scaled
  effects, out of this phase's scope). `desolateland` (Primal Groudon's
  harsh sunlight) doesn't exist in this project's real format pool, so only
  `sunnyday` matters, matching the already-established `_WEATHER_NAMES`
  vocabulary. `is_charge_move` is movedex-static (`flags.get("charge")`,
  same boolean-flag-read pattern as is_contact/is_sound/etc.);
  `needs_charge_turn` is the context-dependent scalar DW-3.1 asks for -
  True for every charge move except when it's Solar Beam/Solar Blade AND
  the current weather is sun.
- Thunder/Hurricane/Blizzard (`_WEATHER_ACCURACY_OVERRIDES`): real dex
  `accuracy` field is a flat 70 for all three (verified directly - no
  declarative weather-conditional field exists) with the actual weather
  logic in `onModifyMove`: Thunder/Hurricane always-hit in rain, 50% in
  sun; Blizzard always-hits in hail/snow (gen 9 has no `hail`, only
  `snowscape` - the existing `"snow"` weather token already covers this).
  Applied as a direct override of the existing per-move accuracy scalar in
  `_move_slot_vector`, not a new dimension - `MoveView.accuracy` itself
  stays the static dex value, matching the "context-dependent numbers
  computed at slot-vector time, not MoveView-construction time" convention
  Phase 1 already established for effectiveness/STAB.
- Terrain power boost (`_TERRAIN_BOOST_TYPE`, `_TERRAIN_POWER_MULTIPLIER`):
  Electric/Grassy/Psychic Terrain each boost their own type's moves by
  5325/4096 (~1.3x, the exact real chainModify fraction from moves.ts) for
  a GROUNDED attacker - Misty Terrain deliberately excluded from this table
  (verified: unlike the other three, its real condition block has no
  onBasePower boost for Fairy-type moves at all, only a Dragon-type 0.5x
  weakening against a grounded DEFENDER, a different mechanic out of this
  phase's scope). Applied the same way as Weather Ball - modifies the
  effective base_power feeding the existing scalar, using the (possibly
  weather-ball-overridden) effective type, so the two stack coherently the
  same way they would in a real damage calculation.
- Terrain status immunity (`_terrain_sleep_immune`/`_terrain_status_immune`,
  `_SLEEP_BLOCKING_TERRAINS`/`_STATUS_BLOCKING_TERRAINS`): verified against
  each terrain's real condition block - Electric Terrain's `onSetStatus`
  blocks sleep specifically (`status.id === 'slp'`) for a grounded, non-
  semi-invulnerable target; Misty Terrain's `onSetStatus` blocks ANY status
  unconditionally (no status-id check at all) plus confusion, for the same
  grounded target. Encoded as two side-level booleans (per active Pokemon,
  same tail placement as the ability-speed-boost boolean below) rather than
  one collapsed flag, since Electric Terrain's real scope (sleep only) is a
  strict subset of Misty Terrain's (all status) - collapsing them would
  misrepresent Electric Terrain as blocking burn/poison/paralysis, which it
  doesn't.
- Ability weather/terrain speed-doubling (`_WEATHER_SPEED_ABILITIES`,
  `_TERRAIN_SPEED_ABILITIES`): Swift Swim/Chlorophyll/Sand Rush/Slush Rush/
  Surge Surfer's exact poke-env-normalized ability ids and their exact
  matching weather/terrain verified directly against abilities.ts's
  `onModifySpe` handlers (`raindance`/`sunnyday`/`sandstorm`/
  `['hail','snowscape']`/`electricterrain` respectively) - no
  groundedness requirement on any of the five (confirmed: none of their
  onModifySpe handlers check isGrounded, unlike the terrain mechanics
  above), so this is a pure ability+weather/terrain lookup, independent of
  `_is_grounded`.

Grounded-ness (`_is_grounded`, per this phase's own Edge Cases note):
factored out of `_is_hazard_immune`'s inline Flying/Levitate check into its
own function, now the shared implementation for both real call sites that
need a groundedness predicate (`_type_multiplier`'s Air Balloon handling
stays separate - it's a per-attack-type-immunity concern, not a
groundedness predicate itself). Verified against the real
`Pokemon.isGrounded` in pokemon-showdown/sim/pokemon.ts: Flying-type,
Levitate, and Air Balloon (while held) all block groundedness; Iron
Ball/Gravity/Magnet Rise/Ingrain/Smack Down force or restore it - all five
of those remain deliberately unmodeled, same named-simplification
convention as `_is_hazard_immune`'s own prior accounting (PokemonView
tracks none of the underlying volatile/field state). A real, incidental
behavior change from this factoring, not previously true: `_is_hazard_immune`
now also reads an Air Balloon holder as hazard-immune (Air Balloon
genuinely blocks groundedness in real Showdown, confirmed in the same
isGrounded read - the prior version never checked for it at all, a real
pre-existing gap this phase's reuse instruction happened to surface, not
something introduced to change hazard behavior for its own sake). Defaults
to grounded=True when types are unknown - the direction that keeps
`_is_hazard_immune`'s already-anchored "unknown Pokemon must not read as
hazard-immune" test passing unchanged: an information gap must never grant
an unearned immunity, on either feature that now depends on this helper.

New global (not per-Pokemon) scalars - `_ability_speed_doubled`,
`_terrain_sleep_immune`, `_terrain_status_immune`, one pair each (my/opp
active) - are inserted into `encode()`'s concatenation BEFORE
`_active_matchup_score`, not after `my_active_hazard_immune`/
`opp_active_hazard_immune`, specifically so those two already-anchored
tail tests (`vec[-3]`/`vec[-2]`/`vec[-1]`) need no changes - same ordering
discipline `PokemonView.protect_counter`'s "stays last" placement already
established for the per-Pokemon block.

Phase 4 (2026-08-27): side-condition completeness, hazard stacking, status
severity, tera-used - the last phase of this rewrite that changes
`VECTOR_LEN`. Every fact below was verified against real poke-env source
(`inspect`-read, not memory) and a 30,000-replay real-data vocabulary scan
before coding - see `.code-foundations/build/2026-08-26-battle-engine-
encoding-rewrite-phase-4-discovery.md` for the full verification trail.

- **Safeguard**: folded straight into the existing `_HAZARD_TOKENS`/
  `_HAZARD_SIDE_CONDITIONS` mechanism (8 -> 9 tokens) - identical shape to
  Reflect/Light Screen (turn-tracked, not in poke-env's own
  `STACKABLE_CONDITIONS`), so both adapters get it for free with no new
  code path. Confirmed real in the replay vocabulary (17 occurrences/30,000
  replays) - rare but real.
- **Hazard stack count** (`BattleView.my_spikes_layers`/
  `my_toxic_spikes_layers`/`opp_spikes_layers`/`opp_toxic_spikes_layers`):
  additive alongside the existing presence-based `my_hazards`/`opp_hazards`
  sets, not a replacement. Live-exact via
  `side_conditions.get(SideCondition.SPIKES, 0)` (poke-env's own
  `STACKABLE_CONDITIONS` stores the real incrementing layer count there,
  verified via `_side_start`'s source). Replay: **live-adapter-only**,
  always 0 - the real replay vocabulary was scanned for a
  `"spikes2"`/`"spikes3"`-style token and found none; the field
  structurally only ever holds the bare condition name, not a count.
  Normalized/clamped by the real Showdown stack caps (Spikes 3, Toxic
  Spikes 2), same `min(x, scale) / scale` convention as
  `_PROTECT_COUNTER_SCALE`.
- **Toxic-counter severity** (`PokemonView.toxic_counter`): live-exact via
  `mon.status_counter if mon.status == Status.TOX else 0` (poke-env's own
  tracked value, correct for a benched Pokemon too - it resets to 0 on
  switch-out on its own, no special-casing needed, same convention as
  `protect_counter`). Replay: **live-adapter-only**, always 0 - the real
  per-mon replay dict's field set has no turn-count equivalent (only a
  current `status` token), and unlike `protect_counter` there's no cheap
  "did this really happen" signal to hang a streak-style reconstruction
  on, so none was attempted. Normalized/clamped against
  `_TOXIC_COUNTER_SCALE` (16 - the real turn count at which badly-poisoned
  damage itself stops growing, even though the raw counter can keep
  climbing past it).
- **Leech Seed/Substitute/Confusion** (`PokemonView.has_leech_seed`/
  `has_substitute`/`is_confused`): both adapters compute these **exactly**,
  no reconstruction needed. Live via `Effect.X in mon.effects`
  (independently - deliberately NOT narrowed to "at most one," unlike the
  hazards/terrain precedent below). Replay via `mon["effect"] ==
  "leechseed"`/`"substitute"`/`"confusion"` (the field itself is
  single-valued, so at most one is ever true on that side already).
  A deliberate departure from the hazards/terrain "narrow live to match
  replay" precedent: true three-way simultaneity is rare here (Leech Seed
  can't even be applied to a Substitute-protected target), and there's no
  local Metamon parser source to verify what it would report if more than
  one were genuinely active - inventing an unverified tie-break rule to
  force live-side narrowing would violate evidence-over-assumption harder
  than leaving the live side with its real, richer information. False for
  an off-field (bench/fainted/unknown) Pokemon on both adapters - poke-env
  itself clears `effects` on switch-out (`Pokemon.switch_out`, verified via
  source), matching real Showdown rules, and a fainted replay slot's stale
  `"effect"` snapshot is explicitly overridden to False rather than trusted
  (see `_replay_pokemon_view_fainted`).
- **`used_tera`** (`BattleView.my_used_tera`/`opp_used_tera`, side-level
  like weather/terrain): live-exact both sides via
  `any(mon.is_terastallized for mon in battle.team.values())` /
  `.opponent_team.values()` - whole team, not just the active mon (a mon
  that terastallized then fainted or got switched out still counts).
  Replay: `my_used_tera` is **exact** via `not state["can_tera"]`
  (`can_tera` verified monotonic across a real replay - `True` until the
  turn tera is used, `False` for every state after, never reverts - so a
  single state is sufficient, no history needed). `opp_used_tera` is
  **live-adapter-only**, always False on replay - no `opponent_can_tera`
  field exists, and a type-divergence heuristic (comparing the opponent's
  current `types` against first-seen) was considered and rejected: real
  non-Tera mechanics (Soak, Trick-or-Treat, Forest's Curse, Reflect Type)
  also change `types` mid-battle and would produce false positives, and a
  same-type Tera would produce a false negative - simpler and more honest
  to document the gap than ship an unverified heuristic.

New per-Pokemon fields (`toxic_counter`, `has_leech_seed`, `has_substitute`,
`is_confused`) are inserted into `_encode_pokemon`'s concatenation BEFORE
the Phase 2 `[preparing, semi_invulnerable, must_recharge]` block, not just
before `protect_counter` in isolation - `preparing` is the EARLIEST
already-anchored tail scalar in that concatenation (`vec[-4]` per
`test_DW_2_1`), so this ordering keeps `vec[-4]`/`vec[-3]`/`vec[-2]`/
`vec[-1]` all unchanged. Likewise, the new global scalars (hazard-layer
counts, `used_tera`) are inserted into `encode()`'s concatenation BEFORE
the Phase 3 6-scalar block, not just before `_active_matchup_score` -
`_ability_speed_doubled`'s pair is the earliest already-anchored tail
scalar there (`vec[-9]` per `test_DW_3_2`), so `vec[-9]` through `vec[-1]`
all stay unchanged too. Same ordering discipline this module has used at
every prior phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData

_TYPE_CHART = GenData.from_gen(9).type_chart
_ALL_TYPES = list(PokemonType)

_STATUSES = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
_STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]
_BOOST_NAMES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
_BASE_STAT_SCALE = 255.0  # Blissey's base HP, the real max base stat in the dex

MAX_BENCH = 5

# Verified against the real replay vocabulary across all currently
# downloaded replays (review, 2026-07-30): these 8 are every non-empty
# value player_conditions/opponent_conditions ever takes. An earlier version
# of this list was missing auroraveil/tailwind entirely, a real gap (not a
# documented tradeoff) - they were silently invisible to the live adapter
# too, since _poke_env_hazards only ever looked at the other 6.
#
# Phase 4 (2026-08-27) added a 9th: "safeguard" - verified real in a
# 30,000-replay vocabulary scan (17 occurrences, rare but real) and
# identical in shape to Reflect/Light Screen (turn-tracked, not in
# poke-env's own STACKABLE_CONDITIONS) - see module docstring.
_HAZARD_TOKENS = [
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "reflect", "lightscreen", "auroraveil", "tailwind", "safeguard",
]
_HAZARD_SIDE_CONDITIONS: Dict[str, SideCondition] = {
    "stealthrock": SideCondition.STEALTH_ROCK,
    "spikes": SideCondition.SPIKES,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
    "stickyweb": SideCondition.STICKY_WEB,
    "reflect": SideCondition.REFLECT,
    "lightscreen": SideCondition.LIGHT_SCREEN,
    "auroraveil": SideCondition.AURORA_VEIL,
    "tailwind": SideCondition.TAILWIND,
    "safeguard": SideCondition.SAFEGUARD,
}

# Real Showdown stack caps (Spikes: 3 layers, Toxic Spikes: 2) - used to
# normalize/clamp BattleView.*_spikes_layers/*_toxic_spikes_layers, same
# min(x, scale) / scale convention as _PROTECT_COUNTER_SCALE below.
_SPIKES_MAX_LAYERS = 3.0
_TOXIC_SPIKES_MAX_LAYERS = 2.0

# Real badly-poisoned damage caps at 1/16 max HP by the 16th consecutive
# turn - Pokemon.status_counter itself can keep incrementing past that, but
# the damage stops growing, so 16 is the meaningful normalization/clamp
# scale for PokemonView.toxic_counter (see module docstring).
_TOXIC_COUNTER_SCALE = 16.0

_WEATHER_NAMES = ["sandstorm", "raindance", "sunnyday", "snow"]
_WEATHER_FROM_POKE_ENV: Dict[Weather, str] = {
    Weather.SANDSTORM: "sandstorm",
    Weather.RAINDANCE: "raindance",
    Weather.SUNNYDAY: "sunnyday",
    Weather.SNOWSCAPE: "snow",
}

_TERRAIN_NAMES = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
_TERRAIN_FROM_POKE_ENV: Dict[Field, str] = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

# Abilities that grant a hard immunity to one attacking type (multiplier -> 0),
# keyed the same way poke-env/Metamon both already normalize ability names
# (lowercase, no spaces/punctuation - poke-env via to_id_str, Metamon's replay
# data the same way natively). Not exhaustive of every ability that touches
# type effectiveness (partial resistances like Thick Fat, move-flag immunities
# like Soundproof/Bulletproof are out of scope) - see module docstring.
_TYPE_IMMUNITY_ABILITIES: Dict[str, PokemonType] = {
    "levitate": PokemonType.GROUND,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS,
    "flashfire": PokemonType.FIRE,
    "eartheater": PokemonType.GROUND,
    "wellbakedbody": PokemonType.FIRE,
}
_UNKNOWN_ABILITY_TOKEN = "unknownability"  # Metamon's placeholder for "not yet revealed"

# Moves that increment poke-env's own Pokemon.protect_counter (verified
# against poke_env.battle.move's private _PROTECT_MOVES/_PROTECT_COUNTER_MOVES
# constants directly, not guessed - not importable, since they're private,
# so the exact name set is duplicated here rather than reconstructed by hand).
# Mat Block is a real protect-like move but deliberately excluded, matching
# poke-env's own _PROTECT_COUNTER_MOVES (side-protect moves wideguard/
# quickguard DO increment the counter; matblock does not).
_PROTECT_COUNTER_MOVES = {
    "protect", "detect", "endure", "spikyshield", "kingsshield", "banefulbunker",
    "burningbulwark", "obstruct", "maxguard", "silktrap", "wideguard", "quickguard",
}
_PROTECT_COUNTER_SCALE = 5.0  # real streaks are rare/short (see module docstring); a loose ceiling

# Real gap found 2026-07-31, after milestone E's first gate result (30.2%,
# format-mismatch-corrected) came in well below the >50% target: replay
# inspection during diagnosis showed the trained model ranking a switch into
# a healthy Heavy-Duty-Boots/Roost/Defog pivot *below* staying in with an
# about-to-faint attacker - exactly the kind of decision that needs item and
# moveset knowledge (hazard immunity, recovery, hazard removal) this encoder
# was never giving it. Both adapters' raw data actually carry this (verified:
# a real downloaded replay state's per-mon dict has 'item' and a full 'moves'
# list with name/type/category/base_power/priority; poke-env's live
# Pokemon has the same via .item and .moves), it just wasn't being read.
#
# Move *identity* isn't one-hot encoded (same "too many distinct values for a
# hand-built vector" reasoning as abilities) - instead, each mon's known
# moveset is summarized into a handful of hand-engineered features via
# Showdown's own movedex flags (_MOVES_DEX), the same "verify against real
# data/library source, don't hand-guess" approach _HAZARD_TOKENS and
# _TYPE_IMMUNITY_ABILITIES already use:
# - has_recovery: any known move has movedex flags.heal (Roost, Recover, ...)
# - has_hazard_setup: any known move sets a hazard side condition
#   (sideCondition in _HAZARD_SETUP_CONDITIONS - Stealth Rock/Spikes/Toxic
#   Spikes/Sticky Web, not screens, which don't affect switch-safety the
#   same way)
# - has_hazard_removal: known move name in _HAZARD_REMOVAL_MOVES - unlike the
#   features above, Showdown's movedex has no data-level flag for this (it's
#   coded as onHit simulator logic, not declared data), so this is a short,
#   verifiable hardcoded list, same spirit as _HAZARD_TOKENS itself
# - has_setup_boost: any known *self*-targeted move's movedex entry has a
#   positive `boosts` value (Swords Dance, Calm Mind, Dragon Dance, ...).
#   Gated on target == "self" - a real false-positive bug caught by review
#   (2026-07-31): Swagger/Flatter also have positive top-level `boosts`
#   values but apply them to the *opponent* (target: "normal"), so an
#   earlier version of this check flagged them as setup moves too. Still
#   imprecise for the rare opponent-debuff move with a *negative* boosts
#   entry (Screech), which correctly does NOT set this flag, an acceptable
#   miss. Belly Drum and Acupressure are real self-buffing moves with no
#   declarative `boosts` field at all (implemented via onHit simulator
#   logic instead, same reason has_hazard_removal needs a hardcoded list) -
#   _ONHIT_SETUP_MOVES below, same short-verifiable-list pattern.
#   Curse is deliberately left off that list even though it's also
#   onHit/boosts=None: its target is "normal" and its effect is genuinely
#   type-dependent (stat boost + self-damage for non-Ghost users, target-
#   damage + no boost for Ghost users) - can't be resolved without the
#   user's type at lookup time, so treating it as "not a setup move" is the
#   conservative, defensible choice, not a gap.
# - has_pivot: any known move has movedex flags.selfSwitch (U-turn, Volt
#   Switch, Parting Shot, ...) - relevant to switch-safety reasoning
#   specifically, distinct from has_priority below
# - has_priority: any known move has priority > 0
# - max_base_power: the highest base_power among known damaging moves,
#   normalized - a coarse "how hard can this thing hit" signal
# - move type coverage: multi-hot over types appearing among known moves -
#   distinct from the mon's own types (used for STAB/matchup), this is about
#   what a switch-in can threaten back with
_MOVES_DEX = GenData.from_gen(9).moves
_HAZARD_SETUP_CONDITIONS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_HAZARD_REMOVAL_MOVES = {"rapidspin", "defog", "courtchange", "tidyup", "mortalspin"}
_ONHIT_SETUP_MOVES = {"bellydrum", "acupressure"}
_MAX_BASE_POWER_SCALE = 250.0  # Explosion/Self-Destruct-class outliers, a loose ceiling

# Phase 2: charge moves that ALSO grant semi-invulnerability during their
# charge turn, distinct from a charge move that merely skips an action
# (Solar Beam et al) - Showdown has no data-level flag for this split (it's
# simulator condition/onInvulnerability logic, not declared data), so this
# is a short, verified hardcoded list, same pattern as _HAZARD_REMOVAL_MOVES/
# _ONHIT_SETUP_MOVES. Verified against the real local pokemon-showdown/
# data/moves.ts checkout (poke-env's own GenData strips the condition/
# onTryMove fields this needs): all six carry a real
# `condition: { onInvulnerability: ... }` block; every other charge move
# (Solar Beam, Sky Attack, Skull Bash, Freeze Shock, Ice Burn, Meteor Beam,
# Electro Shot, Geomancy, Razor Wind) carries no `condition` key at all -
# see module docstring for the full verification, including tracing
# Phantom Force/Shadow Force's literal `onInvulnerability: false` through
# sim/battle.ts's runEvent to confirm it means "always invulnerable, no
# exceptions" rather than "not invulnerable."
_SEMI_INVULNERABLE_CHARGE_MOVES = {
    "fly", "dig", "dive", "bounce", "phantomforce", "shadowforce",
}

# --- Phase 3: weather/terrain-conditional tables -----------------------------
# See module docstring for the real pokemon-showdown/data/moves.ts and
# data/abilities.ts reads each of these was verified against, not memory.

# Weather Ball's real effective type/power under each weather - Normal/50BP
# otherwise (the move's own static dex entry, used unmodified).
_WEATHER_BALL_TYPE_BY_WEATHER: Dict[str, PokemonType] = {
    "sunnyday": PokemonType.FIRE,
    "raindance": PokemonType.WATER,
    "sandstorm": PokemonType.ROCK,
    "snow": PokemonType.ICE,
}

# The only two charge moves whose real onTryMove skips the charge turn in
# sun (verified: every other desolateland/sunnyday-conditional moves.ts hit
# is an unrelated weather-scaled healing move, see module docstring).
_SUN_CHARGE_SKIP_MOVES = {"solarbeam", "solarblade"}

# Real per-move weather-conditional accuracy override (the dex's own
# `accuracy` field is a flat 70 for all three - see module docstring).
_WEATHER_ACCURACY_OVERRIDES: Dict[str, Dict[str, float]] = {
    "thunder": {"raindance": 1.0, "sunnyday": 0.5},
    "hurricane": {"raindance": 1.0, "sunnyday": 0.5},
    "blizzard": {"snow": 1.0},
}

# Electric/Grassy/Psychic Terrain boost their own type's moves for a
# grounded attacker - Misty Terrain deliberately excluded (see module
# docstring: it doesn't boost Fairy moves in the real game).
_TERRAIN_BOOST_TYPE: Dict[str, PokemonType] = {
    "electricterrain": PokemonType.ELECTRIC,
    "grassyterrain": PokemonType.GRASS,
    "psychicterrain": PokemonType.PSYCHIC,
}
_TERRAIN_POWER_MULTIPLIER = 5325 / 4096  # real chainModify fraction, ~1.3x

# Electric Terrain blocks sleep only; Misty Terrain blocks any status -
# real, differently-scoped mechanics, kept as separate tables/booleans
# rather than collapsed into one (see module docstring).
_SLEEP_BLOCKING_TERRAINS = {"electricterrain", "mistyterrain"}
_STATUS_BLOCKING_TERRAINS = {"mistyterrain"}

# Abilities that double Speed under a specific weather/terrain - exact
# poke-env-normalized ids and exact matching condition verified directly
# against abilities.ts's onModifySpe handlers (see module docstring). None
# of the five require groundedness.
_WEATHER_SPEED_ABILITIES: Dict[str, str] = {
    "swiftswim": "raindance",
    "chlorophyll": "sunnyday",
    "sandrush": "sandstorm",
    "slushrush": "snow",
}
_TERRAIN_SPEED_ABILITIES: Dict[str, str] = {"surgesurfer": "electricterrain"}

# --- per-move-slot features (MoveView) --------------------------------------
#
# MAX_MOVES matches poke-env's/Metamon's own real cap (a Pokemon can't know
# more than 4 moves) - same "real per-entity ceiling, loud failure if
# exceeded" convention as MAX_BENCH/_pad_bench.
MAX_MOVES = 4

# Movedex `target` values verified (2026-08-26, real distribution survey of
# every gen-9 move) to actually hit the opposing active Pokemon - "normal"
# (632), "allAdjacentFoes" (62), "adjacentFoe" (53), "any" (24),
# "allAdjacent" (20), "randomNormal" (6), "scripted" (4 - Counter/Mirror
# Coat/Metal Burst/Comeuppance, real retaliation moves, confirmed
# opponent-directed by inspection). Everything else ("self" - Swords Dance;
# "foeSide"/"allySide" - hazards/screens, side-wide not mon-specific; "all";
# ally-only targets, doubles-only and irrelevant in singles) never resolves
# against a specific opposing Pokemon, so a type-effectiveness number for
# those would be meaningless, not just uninteresting - see this phase's own
# Edge Cases note. MoveView.targets_opponent records this per move so
# encode() knows when the effectiveness number it computes is real signal
# vs. a padded 0.0.
_OPPONENT_DIRECTED_TARGETS = {
    "normal", "any", "adjacentFoe", "allAdjacentFoes", "allAdjacent",
    "randomNormal", "scripted",
}

_MOVE_CATEGORIES = ["Physical", "Special", "Status"]  # real _MOVES_DEX string values, used as-is
_SECONDARY_KINDS = ["status", "boost_drop", "flinch"]
_PRIORITY_SCALE = 7.0  # real gen-9 movedex priority range is exactly -7..+5

# Verified against the real replay vocabulary: the 20 most common held
# items. An "other known item" bucket below covers everything outside this
# list rather than silently treating a rare item as blank/no-item. Original
# 2026-07-31 pass sampled ~640 states across 800 replays and put
# blackglasses 20th; review re-checked against a larger 400-file sample and
# found blacksludge (338 occurrences) is actually more common than
# blackglasses (214) - swapped. Low-severity either way (a mis-ranked vocab
# item just falls into the "other known" bucket instead of a dedicated
# slot, nothing corrupts), but worth keeping accurate since re-verifying is
# cheap.
_ITEM_VOCAB = [
    "leftovers", "heavydutyboots", "rockyhelmet", "boosterenergy", "lifeorb",
    "choiceband", "choicescarf", "choicespecs", "assaultvest", "airballoon",
    "focussash", "wellspringmask", "toxicorb", "loadeddice", "lightclay",
    "heatrock", "weaknesspolicy", "damprock", "eviolite", "blacksludge",
]
_UNKNOWN_ITEM_TOKENS = {None, "", "noitem", "unknownitem", "unknown_item"}

# Per-move-slot vector layout (see _move_slot_vector): type one-hot,
# category one-hot, secondary-kind one-hot, plus 25 scalars (known, stab,
# base_power, accuracy, priority, targets_opponent, type_effectiveness,
# secondary_chance, self_boost_chance, self_boost_magnitude, fixed_damage,
# multi_hit, is_contact, is_sound, is_punch, is_bite, is_pulse, is_bullet,
# is_wind, is_protect_counter, bypasses_protect, recoil_fraction,
# drain_fraction, is_self_ko, needs_charge_turn - Phase 2 added the 4
# scalars ending at is_self_ko, Phase 3 adds needs_charge_turn at the end
# (see module docstring for both).
_MOVE_VEC_LEN = len(_ALL_TYPES) + len(_MOVE_CATEGORIES) + len(_SECONDARY_KINDS) + 25

_POKEMON_VEC_LEN = (
    1  # known
    + 1  # hp_fraction
    + 1  # fainted
    + len(_STATUSES)
    + len(_ALL_TYPES)
    + len(_BOOST_NAMES)
    + len(_STAT_NAMES)
    + len(_ITEM_VOCAB) + 1  # item one-hot + "other known item" bucket
    + 5  # has_recovery, has_hazard_setup, has_hazard_removal, has_setup_boost, has_pivot
    + 1  # has_priority
    + 1  # max_base_power (normalized)
    + len(_ALL_TYPES)  # move type coverage (distinct from the mon's own types above)
    + MAX_MOVES * _MOVE_VEC_LEN  # per-move-slot block (MoveView) - see module docstring
    + 4  # Phase 4: toxic_counter, has_leech_seed, has_substitute, is_confused - see module docstring
    + 3  # Phase 2: preparing, semi_invulnerable, must_recharge - see module docstring
    + 1  # protect_counter (normalized) - stays LAST, see module docstring
)
VECTOR_LEN = (
    _POKEMON_VEC_LEN * (1 + MAX_BENCH + 1)  # my active, my bench, opponent active
    + 1  # opponent fraction remaining
    + 2 * len(_HAZARD_TOKENS)  # hazards, both sides (9 tokens as of Phase 4 - includes safeguard)
    + len(_WEATHER_NAMES)
    + len(_TERRAIN_NAMES)
    + 4  # Phase 4: my/opp_spikes_layers, my/opp_toxic_spikes_layers
    + 2  # Phase 4: my/opp_used_tera
    + 6  # Phase 3: my/opp_active_speed_doubled, terrain_sleep_immune, terrain_status_immune
    + 1  # active-vs-active type matchup score
    + 2  # my_active_hazard_immune, opp_active_hazard_immune
)


@dataclass
class MoveSummary:
    """Hand-engineered summary of a Pokemon's known moveset - see the
    _MOVES_DEX comment block above for why these specific features and not
    raw move identity. Both adapters build this via _move_summary_features
    from a plain list of move-id strings, so the derivation logic itself
    isn't duplicated per adapter.
    """
    has_recovery: bool = False
    has_hazard_setup: bool = False
    has_hazard_removal: bool = False
    has_setup_boost: bool = False
    has_pivot: bool = False
    has_priority: bool = False
    max_base_power: int = 0
    move_types: frozenset = frozenset()


def _move_summary_features(move_ids: Sequence[str]) -> MoveSummary:
    summary = MoveSummary()
    max_bp = 0
    move_types = set()
    has_recovery = has_hazard_setup = has_hazard_removal = False
    has_setup_boost = has_pivot = has_priority = False
    for move_id in move_ids:
        entry = _MOVES_DEX.get(move_id)
        if entry is None:  # unrecognized/typo-guarded - skip rather than crash
            continue
        flags = entry.get("flags", {})
        if flags.get("heal"):
            has_recovery = True
        if entry.get("sideCondition") in _HAZARD_SETUP_CONDITIONS:
            has_hazard_setup = True
        if move_id in _HAZARD_REMOVAL_MOVES:
            has_hazard_removal = True
        if move_id in _ONHIT_SETUP_MOVES:
            has_setup_boost = True
        elif entry.get("target") == "self" and any(
            v > 0 for v in entry.get("boosts", {}).values()
        ):
            has_setup_boost = True
        if flags.get("selfSwitch") or entry.get("selfSwitch"):
            has_pivot = True
        if entry.get("priority", 0) > 0:
            has_priority = True
        max_bp = max(max_bp, entry.get("basePower", 0) or 0)
        move_type = entry.get("type")
        if move_type:
            move_types.add(PokemonType.from_name(move_type))
    return MoveSummary(
        has_recovery=has_recovery,
        has_hazard_setup=has_hazard_setup,
        has_hazard_removal=has_hazard_removal,
        has_setup_boost=has_setup_boost,
        has_pivot=has_pivot,
        has_priority=has_priority,
        max_base_power=max_bp,
        move_types=frozenset(move_types),
    )


@dataclass
class MoveView:
    """Static, context-free per-move-slot data - everything derivable from
    the move's own _MOVES_DEX entry alone, same verify-against-real-data
    sourcing as _move_summary_features. Battle-context-dependent numbers
    (type effectiveness against the CURRENT opponent, STAB) are deliberately
    NOT here - see module docstring for why encode() computes those once
    both mons' state is available, rather than here at construction time.

    self_boost_chance/self_boost_magnitude cover only the Draco Meteor/Close
    Combat-style unconditional self-stat-change (movedex `self.boosts`,
    always applied on a successful hit) - NOT a pure self-targeted status
    move's top-level `boosts` field (Swords Dance - already covered by the
    existing MoveSummary.has_setup_boost) and NOT a chance-based nested
    `secondary.self.boosts` (Steel Wing/Ancient Power/Flame Charge - rarer,
    deliberately out of scope, same "not attempted, revisit if it matters"
    standard as this module's other named simplifications).

    Phase 2 additions - bypasses_protect, recoil_fraction, drain_fraction,
    is_self_ko - are also movedex-static (see module docstring for the real
    dex fields each reads and how each was verified). The charge/semi-
    invulnerable/recharge signals from the same phase are deliberately NOT
    here - they're per-Pokemon RUNTIME state (which move a mon is currently
    mid-charge on, if any), not a property of a move's dex entry, so they
    live on PokemonView instead (see PokemonView.preparing/
    semi_invulnerable/must_recharge).
    """
    move_id: str
    known: bool = True
    type: Optional[PokemonType] = None
    category: Optional[str] = None
    base_power: int = 0
    accuracy: float = 1.0
    priority: int = 0
    targets_opponent: bool = False
    secondary_chance: float = 0.0
    secondary_kind: Optional[str] = None  # "status" | "boost_drop" | "flinch" | None
    self_boost_chance: float = 0.0
    self_boost_magnitude: float = 0.0  # signed, mean boost delta across affected stats
    fixed_damage: bool = False
    multi_hit: bool = False
    is_contact: bool = False
    is_sound: bool = False
    is_punch: bool = False
    is_bite: bool = False
    is_pulse: bool = False
    is_bullet: bool = False
    is_wind: bool = False
    is_protect_counter: bool = False
    bypasses_protect: bool = False
    recoil_fraction: float = 0.0
    drain_fraction: float = 0.0
    is_self_ko: bool = False
    # Phase 3: movedex-static (flags.charge) - whether this move requires a
    # charge turn at all. Combined with the CONTEXT-dependent current
    # weather in _move_slot_vector to produce needs_charge_turn (see module
    # docstring) - kept separate from that scalar since is_charge_move
    # itself never changes, only whether the charge is currently skipped.
    is_charge_move: bool = False

    @staticmethod
    def unknown() -> "MoveView":
        return MoveView(move_id="", known=False)


def _secondary_effect(entry: dict) -> Tuple[float, Optional[str]]:
    secondary = entry.get("secondary")
    if not secondary:
        return 0.0, None
    chance = secondary.get("chance", 100) / 100.0
    if secondary.get("status"):
        return chance, "status"
    if secondary.get("volatileStatus") == "flinch":
        return chance, "flinch"
    if secondary.get("boosts"):
        return chance, "boost_drop"
    # A real secondary effect exists (chance is still meaningful) but its
    # kind isn't one of the three this module buckets - e.g. a chance-based
    # SELF-only nested effect (Steel Wing's secondary.self.boosts), or a
    # rarer volatileStatus. Not modeled further, see MoveView's docstring.
    return chance, None


def _self_boost(entry: dict) -> Tuple[float, float]:
    self_effect = entry.get("self")
    if not self_effect or not self_effect.get("boosts"):
        return 0.0, 0.0
    values = list(self_effect["boosts"].values())
    return 1.0, (sum(values) / len(values)) / 6.0  # unconditional on hit -> chance 1.0


def _fraction(pair: Optional[Sequence[int]]) -> float:
    # Real dex shape verified against flareblitz.recoil == [33, 100] and
    # gigadrain.drain == [1, 2] (see module docstring) - [numerator,
    # denominator], None when the move has no such effect.
    if not pair:
        return 0.0
    numerator, denominator = pair
    return numerator / denominator


def _move_view(move_id: str) -> Optional[MoveView]:
    entry = _MOVES_DEX.get(move_id)
    if entry is None:  # unrecognized/typo-guarded - skip, matches _move_summary_features
        return None
    flags = entry.get("flags", {})
    secondary_chance, secondary_kind = _secondary_effect(entry)
    self_boost_chance, self_boost_magnitude = _self_boost(entry)
    move_type = entry.get("type")
    accuracy = entry.get("accuracy")
    return MoveView(
        move_id=move_id,
        known=True,
        type=PokemonType.from_name(move_type) if move_type else None,
        category=entry.get("category"),
        base_power=entry.get("basePower", 0) or 0,
        accuracy=1.0 if accuracy is True else (accuracy or 0) / 100.0,
        priority=entry.get("priority", 0) or 0,
        targets_opponent=entry.get("target") in _OPPONENT_DIRECTED_TARGETS,
        secondary_chance=secondary_chance,
        secondary_kind=secondary_kind,
        self_boost_chance=self_boost_chance,
        self_boost_magnitude=self_boost_magnitude,
        fixed_damage=bool(entry.get("damage")),
        multi_hit=bool(entry.get("multihit")),
        is_contact=bool(flags.get("contact")),
        is_sound=bool(flags.get("sound")),
        is_punch=bool(flags.get("punch")),
        is_bite=bool(flags.get("bite")),
        is_pulse=bool(flags.get("pulse")),
        is_bullet=bool(flags.get("bullet")),
        is_wind=bool(flags.get("wind")),
        is_protect_counter=move_id in _PROTECT_COUNTER_MOVES,
        bypasses_protect=not bool(flags.get("protect")),
        recoil_fraction=_fraction(entry.get("recoil")),
        drain_fraction=_fraction(entry.get("drain")),
        is_self_ko=bool(entry.get("selfdestruct")),
        is_charge_move=bool(flags.get("charge")),
    )


def _move_views(move_ids: Sequence[str]) -> Tuple[MoveView, ...]:
    """Builds up to MAX_MOVES known-move slots, sorted by move id (not
    reveal order) for turn-to-turn identity stability against a
    partially-revealed opponent - same "why sort bench by species name"
    reasoning already established in this module (see docstring) - then
    pads with MoveView.unknown() the same "unknown/zero" way
    PokemonView.unknown() pads a whole missing Pokemon.
    """
    views = sorted(
        (v for v in (_move_view(mid) for mid in move_ids) if v is not None),
        key=lambda v: v.move_id,
    )
    assert len(views) <= MAX_MOVES, (
        f"{len(views)} known moves exceeds MAX_MOVES={MAX_MOVES} - a real "
        "Pokemon can't know more than 4 moves; silently truncating would "
        "drop a real move rather than surface a bug"
    )
    return tuple(views) + tuple(MoveView.unknown() for _ in range(MAX_MOVES - len(views)))


@dataclass
class PokemonView:
    known: bool
    hp_fraction: float
    fainted: bool
    status: Optional[Status]
    types: Tuple[PokemonType, ...]
    boosts: Dict[str, int]
    base_stats: Dict[str, int]
    # Not itself an encoded dimension (see module docstring) - consumed only by
    # _type_multiplier to correct the matchup-score dimension for known
    # type-immunity abilities. None means "no ability" or "not yet revealed";
    # those are indistinguishable from the outside, same as in a real battle.
    ability: Optional[str] = None
    # None means no item held OR not yet revealed - same "indistinguishable
    # from the outside" convention as ability above (see _UNKNOWN_ITEM_TOKENS).
    item: Optional[str] = None
    moves: MoveSummary = field(default_factory=MoveSummary)
    # Up to MAX_MOVES per-move-slot views (see MoveView) - additive
    # alongside `moves` above, not a replacement: the aggregate MoveSummary
    # is still a real, distinct switch-safety signal. Sorted by move id,
    # padded with MoveView.unknown() - see _move_views.
    move_slots: Tuple[MoveView, ...] = field(
        default_factory=lambda: tuple(MoveView.unknown() for _ in range(MAX_MOVES))
    )
    # Phase 2 runtime state (see module docstring for full verification):
    # whether this Pokemon is mid-charge on a two-turn move
    # (Pokemon.preparing), and if so whether that specific move also grants
    # semi-invulnerability (_SEMI_INVULNERABLE_CHARGE_MOVES) versus merely
    # skipping this turn's action (Solar Beam et al). Live-adapter-only -
    # verified no equivalent exists in Metamon's replay schema - so both
    # default False on the replay side and for any off-field (bench/
    # unknown) Pokemon, same "known-false-is-correct, not just a
    # placeholder" status protect_counter already has for the bench case.
    preparing: bool = False
    semi_invulnerable: bool = False
    # Whether this Pokemon must recharge this turn after a recharge move
    # (Hyper Beam, Giga Impact, ...) - Pokemon.must_recharge, direct read,
    # no move-list cross-reference needed (unlike semi_invulnerable above).
    # Same live-adapter-only status as preparing/semi_invulnerable.
    must_recharge: bool = False
    # Phase 4 (2026-08-27): real badly-poisoned severity - see module
    # docstring for the live-exact/replay-gap accounting. 0 is correct for
    # a non-toxic'd Pokemon, an off-field Pokemon, AND (always) the replay
    # side - no equivalent field exists in Metamon's schema.
    toxic_counter: int = 0
    # Phase 4: Leech Seed/Substitute/Confusion - both adapters compute these
    # EXACTLY, no reconstruction needed (see module docstring for why these
    # three don't need the hazards/terrain-style "narrow live to match
    # replay" precedent). False/False/False is correct for an off-field
    # (bench/fainted/unknown) Pokemon on both adapters - poke-env itself
    # clears these on switch-out, matching real Showdown rules.
    has_leech_seed: bool = False
    has_substitute: bool = False
    is_confused: bool = False
    # How many consecutive turns this Pokemon has just used a protect-
    # counter move (Protect, Endure, ...) - see module docstring for why
    # this is a real feature, not a nice-to-have, and how each adapter
    # computes it (exact via poke-env's own tracking on the live side,
    # a verified approximation reconstructed from replay history on the
    # other). 0 for an off-field (bench/unknown) Pokemon, which is always
    # correct, not just a placeholder - poke-env itself resets the real
    # counter to 0 on switch-out. Kept as the LAST field (both here and in
    # _encode_pokemon's concatenation - see module docstring) so its
    # already-anchored vec[-1] test needs no changes as this phase's new
    # fields are added.
    protect_counter: int = 0

    @staticmethod
    def unknown() -> "PokemonView":
        return PokemonView(
            known=False,
            hp_fraction=0.0,
            fainted=False,
            status=None,
            types=(),
            boosts={name: 0 for name in _BOOST_NAMES},
            base_stats={name: 0 for name in _STAT_NAMES},
            ability=None,
            item=None,
            moves=MoveSummary(),
            move_slots=tuple(MoveView.unknown() for _ in range(MAX_MOVES)),
            preparing=False,
            semi_invulnerable=False,
            must_recharge=False,
            toxic_counter=0,
            has_leech_seed=False,
            has_substitute=False,
            is_confused=False,
            protect_counter=0,
        )


@dataclass
class BattleView:
    my_active: PokemonView
    my_bench: Sequence[PokemonView]
    opp_active: PokemonView
    opp_remaining_fraction: float
    my_hazards: set
    opp_hazards: set
    weather: Optional[str]
    terrain: Optional[str]
    # Phase 4 (2026-08-27): real hazard STACK count (Spikes up to 3, Toxic
    # Spikes up to 2) - additive alongside my_hazards/opp_hazards above, not
    # a replacement (see module docstring). Live-exact; always 0 on the
    # replay adapter (documented gap - the real replay schema structurally
    # can't carry this, see module docstring).
    my_spikes_layers: int = 0
    my_toxic_spikes_layers: int = 0
    opp_spikes_layers: int = 0
    opp_toxic_spikes_layers: int = 0
    # Phase 4: whether this side has used its one-time Tera resource at all
    # this battle (whole team, not just the current active mon - see module
    # docstring). Live-exact both sides. Replay: my_used_tera is exact
    # (derived from can_tera); opp_used_tera is always False (documented
    # live-adapter-only gap - no opponent_can_tera field exists).
    my_used_tera: bool = False
    opp_used_tera: bool = False


def _pad_bench(bench: list) -> list:
    assert len(bench) <= MAX_BENCH, (
        f"{len(bench)} bench slots exceeds MAX_BENCH={MAX_BENCH} - silently "
        "truncating would drop a real Pokemon rather than surface a bug"
    )
    return bench + [PokemonView.unknown() for _ in range(MAX_BENCH - len(bench))]


# --- poke-env (live battle) adapter -----------------------------------------


def _poke_env_item(mon: Pokemon) -> Optional[str]:
    return None if mon.item in _UNKNOWN_ITEM_TOKENS else mon.item


def _poke_env_semi_invulnerable(mon: Pokemon) -> bool:
    preparing_move = mon.preparing_move
    return preparing_move is not None and preparing_move.id in _SEMI_INVULNERABLE_CHARGE_MOVES


def _poke_env_pokemon_view(mon: Optional[Pokemon]) -> PokemonView:
    if mon is None:
        return PokemonView.unknown()
    move_ids = [m.id for m in mon.moves.values()]
    return PokemonView(
        known=True,
        hp_fraction=mon.current_hp_fraction,
        fainted=mon.fainted,
        status=mon.status if mon.status != Status.FNT else None,
        types=tuple(t for t in mon.types if t is not None),
        boosts=dict(mon.boosts),
        base_stats=dict(mon.base_stats),
        ability=mon.ability,  # None if not yet revealed, already normalized (to_id_str)
        item=_poke_env_item(mon),
        moves=_move_summary_features(move_ids),
        move_slots=_move_views(move_ids),
        # mon.preparing is poke-env's own public property
        # (bool(preparing_target) or bool(preparing_move)) - used directly
        # rather than re-deriving the same OR from the two private-backed
        # properties (see module docstring).
        preparing=mon.preparing,
        semi_invulnerable=_poke_env_semi_invulnerable(mon),
        must_recharge=mon.must_recharge,
        # Phase 4: exact live reads (see module docstring) - status_counter
        # is only meaningful while actually badly poisoned; Effect
        # membership is checked independently per effect, not narrowed to
        # "at most one" (see module docstring for why, unlike hazards).
        toxic_counter=mon.status_counter if mon.status == Status.TOX else 0,
        has_leech_seed=Effect.LEECH_SEED in mon.effects,
        has_substitute=Effect.SUBSTITUTE in mon.effects,
        is_confused=Effect.CONFUSION in mon.effects,
        protect_counter=mon.protect_counter,
    )


def _poke_env_hazards(side_conditions: dict) -> set:
    """The single most-recently-set hazard/screen, matching the real replay
    data's single-valued field (see module docstring) rather than every
    condition simultaneously active - a deliberate fidelity trade for
    train/inference parity.

    poke-env's own STACKABLE_CONDITIONS (side_condition.py) is the
    authoritative list of which conditions it tracks as a stack *count*
    (Spikes, Toxic Spikes) rather than the *turn number* it stores for
    everything else (verified against abstract_battle.py's _side_start) -
    used directly here rather than a hand-maintained guess at the split, so
    this stays correct if poke-env's own classification ever changes.
    True recency across a stack-counted and a turn-tracked condition can't
    be compared directly, so turn-tracked conditions are ranked by turn
    when any are active; a stackable condition is only reported when none
    are, tie-broken by _HAZARD_SIDE_CONDITIONS' fixed order. An
    approximation, not exact reconstruction - documented rather than
    silently assumed.
    """
    turn_tracked = {
        token: side_conditions[condition]
        for token, condition in _HAZARD_SIDE_CONDITIONS.items()
        if condition in side_conditions and condition not in STACKABLE_CONDITIONS
    }
    if turn_tracked:
        return {max(turn_tracked, key=turn_tracked.get)}

    for token, condition in _HAZARD_SIDE_CONDITIONS.items():
        if condition in side_conditions and condition in STACKABLE_CONDITIONS:
            return {token}
    return set()


def _poke_env_weather(weather: dict) -> Optional[str]:
    for w in weather:
        name = _WEATHER_FROM_POKE_ENV.get(w)
        if name is not None:
            return name
    return None


def _poke_env_terrain(fields: dict) -> Optional[str]:
    """Real bug caught by review: the previous version returned the first
    terrain-type entry found in `fields`, ignoring non-terrain field effects
    entirely (Trick Room, Gravity) - so if Trick Room was set *more
    recently* than an active terrain, this still reported the terrain,
    while the replay adapter's single-valued battle_field would correctly
    show "trickroom" and map to no terrain. Matches that now: rank the
    single most-recently-set field effect by turn (battle.fields stores a
    real turn number for every entry - verified in abstract_battle.py,
    `self._fields[field] = self.turn` unconditionally, no stackable-count
    exception like side_conditions has), and only report it as a terrain if
    it actually is one.
    """
    if not fields:
        return None
    most_recent = max(fields, key=fields.get)
    return _TERRAIN_FROM_POKE_ENV.get(most_recent)


def battle_view_from_poke_env(battle: AbstractBattle) -> BattleView:
    if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
        # Real gap caught by review: at team preview, poke-env's
        # battle.team already has all 6 mons (populated from the
        # teampreview request - verified in poke-env's battle.py) but
        # battle.active_pokemon is still None (nothing has switched in
        # yet). Without this guard, "bench = everything except the active
        # one" keeps all 6, and _pad_bench's overflow assert fires with a
        # confusing "6 bench slots exceeds MAX_BENCH" message that has
        # nothing to do with the actual problem (no active Pokemon yet).
        # This BattleView shape has no representation for "no active
        # Pokemon chosen yet" - callers (e.g. a future team-preview-aware
        # search) need to handle that decision separately, not via this
        # function. Nothing in this codebase calls this during team
        # preview today - search.py already guards on active_pokemon is
        # None before doing anything encoding-related - but milestone E
        # wiring a model into a Player could plausibly do so.
        raise ValueError(
            "battle_view_from_poke_env requires both active Pokemon to be "
            "chosen (not team preview) - no active_pokemon means there's no "
            "well-defined 'bench' to encode yet"
        )
    bench_mons = sorted(
        (mon for mon in battle.team.values() if mon is not battle.active_pokemon),
        key=lambda mon: mon.base_species,
    )
    bench = [_poke_env_pokemon_view(mon) for mon in bench_mons]
    opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
    return BattleView(
        my_active=_poke_env_pokemon_view(battle.active_pokemon),
        my_bench=_pad_bench(bench),
        opp_active=_poke_env_pokemon_view(battle.opponent_active_pokemon),
        opp_remaining_fraction=(6 - opp_fainted) / 6,
        my_hazards=_poke_env_hazards(battle.side_conditions),
        opp_hazards=_poke_env_hazards(battle.opponent_side_conditions),
        weather=_poke_env_weather(battle.weather),
        terrain=_poke_env_terrain(battle.fields),
        # Phase 4: exact live reads (see module docstring). STACKABLE_CONDITIONS
        # (Spikes, Toxic Spikes) store the real layer count directly.
        my_spikes_layers=battle.side_conditions.get(SideCondition.SPIKES, 0),
        my_toxic_spikes_layers=battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0),
        opp_spikes_layers=battle.opponent_side_conditions.get(SideCondition.SPIKES, 0),
        opp_toxic_spikes_layers=battle.opponent_side_conditions.get(SideCondition.TOXIC_SPIKES, 0),
        # Whole team, not just the active mon - a mon that terastallized
        # then fainted/switched out still counts (see module docstring).
        my_used_tera=any(mon.is_terastallized for mon in battle.team.values()),
        opp_used_tera=any(mon.is_terastallized for mon in battle.opponent_team.values()),
    )


# --- Metamon replay-state adapter -------------------------------------------


def _species_key(mon: dict) -> str:
    # base_species, not name: a real bug caught against actual data - Metamon
    # renames a Pokemon's "name" field on in-battle form changes (observed:
    # Terapagos -> "terapagosterastal" on Tera, Minior -> "miniormeteor" on
    # its shield-break trigger), so name-based identity treated one physical
    # teammate as two, overflowing a real replay's bench past MAX_BENCH.
    # base_species is stable across these (confirmed against the same data).
    return mon["base_species"]


def _replay_types(mon: dict) -> Tuple[PokemonType, ...]:
    return tuple(PokemonType.from_name(t) for t in mon["types"].split() if t != "notype")


def _replay_ability(mon: dict) -> Optional[str]:
    return None if mon["ability"] == _UNKNOWN_ABILITY_TOKEN else mon["ability"]


def _replay_item(mon: dict) -> Optional[str]:
    return None if mon["item"] in _UNKNOWN_ITEM_TOKENS else mon["item"]


def _replay_effect_flags(mon: dict) -> Tuple[bool, bool, bool]:
    """Phase 4: mon["effect"] is single-valued (verified against real replay
    data - see module docstring), so at most one of these three is ever
    true here already, unlike the live adapter's independent Effect checks.
    """
    effect = mon.get("effect")
    return effect == "leechseed", effect == "substitute", effect == "confusion"


def _replay_move_ids(mon: dict) -> list:
    # mon["moves"] entries are already Showdown id-form ("stealthrock", not
    # "Stealth Rock") - verified against a real downloaded replay - so each
    # name indexes _MOVES_DEX directly, same as the live adapter's move.id.
    # Shared by both _move_summary_features (MoveSummary) and _move_views
    # (MoveView per-slot) so the id list itself isn't computed twice.
    return [m["name"] for m in mon["moves"]]


def _replay_pokemon_view(mon: Optional[dict], protect_counter: int = 0) -> PokemonView:
    """protect_counter defaults to 0 - correct for bench slots (an off-field
    Pokemon's real streak is 0, poke-env resets it on switch-out too - see
    PokemonView's docstring) and for single-state callers that can't
    reconstruct it at all (battle_view_from_replay_state). Only
    battle_views_from_replay, which has the whole turn sequence to work
    with, ever passes a real nonzero value in.
    """
    if mon is None:
        return PokemonView.unknown()
    status_token = mon["status"]
    status = None
    if status_token not in ("nostatus", "fnt"):
        status = Status[status_token.upper()]
    move_ids = _replay_move_ids(mon)
    has_leech_seed, has_substitute, is_confused = _replay_effect_flags(mon)
    return PokemonView(
        known=True,
        hp_fraction=mon["hp_pct"],
        fainted=status_token == "fnt",
        status=status,
        types=_replay_types(mon),
        boosts={name: mon[f"{name}_boost"] for name in _BOOST_NAMES},
        base_stats={name: mon[f"base_{name}"] for name in _STAT_NAMES},
        ability=_replay_ability(mon),
        item=_replay_item(mon),
        moves=_move_summary_features(move_ids),
        move_slots=_move_views(move_ids),
        # Phase 4: exact on this adapter (see module docstring) -
        # toxic_counter stays 0 (documented live-adapter-only gap, no
        # equivalent field exists in the replay schema).
        has_leech_seed=has_leech_seed,
        has_substitute=has_substitute,
        is_confused=is_confused,
        protect_counter=protect_counter,
    )


def _replay_pokemon_view_fainted(mon: dict) -> PokemonView:
    """A previously-seen teammate no longer active or in available_switches -
    i.e. it fainted since it was last seen. Real per-state replay data
    doesn't record fainted teammates in available_switches at all (verified:
    0 of 3247 real switch-list entries are fainted; Showdown just stops
    offering them), unlike poke-env's live battle.team, which keeps them.
    Reconstructed here from the last snapshot this replay had of the mon,
    with hp/status/boosts overwritten to reflect having fainted - its exact
    HP right before fainting isn't recoverable from this data, only that
    it's now 0, and boosts don't survive a faint anyway. item/moves are kept
    from that last snapshot (unlike hp/status/boosts, they don't change on
    fainting, and a fainted mon can't be switched to regardless).
    """
    move_ids = _replay_move_ids(mon)
    return PokemonView(
        known=True,
        hp_fraction=0.0,
        fainted=True,
        status=None,
        types=_replay_types(mon),
        boosts={name: 0 for name in _BOOST_NAMES},
        base_stats={name: mon[f"base_{name}"] for name in _STAT_NAMES},
        ability=_replay_ability(mon),
        item=_replay_item(mon),
        moves=_move_summary_features(move_ids),
        move_slots=_move_views(move_ids),
        # Phase 4: explicitly False, NOT read from mon["effect"] - that
        # field is a stale snapshot from when this mon was last seen alive,
        # and Leech Seed/Substitute/Confusion don't survive a faint any more
        # than boosts do (see module docstring).
        has_leech_seed=False,
        has_substitute=False,
        is_confused=False,
    )


def _replay_protect_streaks(states: list) -> Tuple[list, list]:
    """Per-state (my, opponent) protect-counter-streak reconstruction - see
    module docstring for the exact simplification versus poke-env's real
    protect_counter (increments regardless of success/failure; resets on a
    non-protect-counter move OR a change of active species, not attempting
    to detect a failed protect specifically).

    Needs the whole turn sequence, same reason fainted-teammate
    reconstruction does: state[i]'s streak depends on what was chosen and
    who was active in state[i-1], not on state[i] alone.

    A first version of this function treated state[i]'s player_prev_move/
    opponent_prev_move as "the move used on the i-1 -> i transition" -
    wrong, and a real bug (found by independent review, 2026-08-01):
    Metamon's prev_move fields are "the last move this Pokemon has ever
    used", carried forward UNCHANGED across states where that side didn't
    actually act (e.g. an extra decision state generated for the
    opponent's forced switch after a KO, where nothing happened on this
    side at all) - not reset each turn. Verified directly: across a real
    sample, whenever a side's prev_move dict is byte-identical to the
    previous state's (name AND current_pp AND every other field), no new
    move actually happened; whenever the same move name is genuinely
    reused, current_pp always differs. The original version had no way to
    tell these apart from name alone, so a stale carried-over "protect"
    silently kept incrementing an already-resolved streak - measured against
    real per-move PP as ground truth: 59.4% of reconstructed streak-2 values
    and 41.7% of streak-3 values were wrong as a direct result.

    Fixed by carrying the PREVIOUS streak forward unchanged (rather than
    incrementing OR resetting) whenever this side's prev_move dict is
    identical to last state's - "nothing happened for this side, so
    whatever was true a moment ago is still true now" - a self-contained
    signal that needs no extra data (like the replay's separate per-state
    action labels) threaded through this function.
    """
    my_streaks: list = []
    opp_streaks: list = []
    my_streak = opp_streak = 0
    my_prev_species: Optional[str] = None
    opp_prev_species: Optional[str] = None
    my_prev_move: Optional[dict] = None
    opp_prev_move: Optional[dict] = None

    for state in states:
        my_species = _species_key(state["player_active_pokemon"])
        opp_species = _species_key(state["opponent_active_pokemon"])
        my_move = state["player_prev_move"]
        opp_move = state["opponent_prev_move"]

        if my_species != my_prev_species:
            my_streak = 0
        elif my_move != my_prev_move:
            my_streak = my_streak + 1 if my_move["name"] in _PROTECT_COUNTER_MOVES else 0
        # else: no real action for this side this transition - my_streak
        # (already computed for the previous state) carries forward as-is.

        if opp_species != opp_prev_species:
            opp_streak = 0
        elif opp_move != opp_prev_move:
            opp_streak = opp_streak + 1 if opp_move["name"] in _PROTECT_COUNTER_MOVES else 0

        my_streaks.append(my_streak)
        opp_streaks.append(opp_streak)
        my_prev_species, opp_prev_species = my_species, opp_species
        my_prev_move, opp_prev_move = my_move, opp_move

    return my_streaks, opp_streaks


def _replay_hazards(conditions: str) -> set:
    return set(conditions.split()) & set(_HAZARD_TOKENS)


def _replay_weather(token: str) -> Optional[str]:
    return token if token in _WEATHER_NAMES else None


def _replay_terrain(token: str) -> Optional[str]:
    return token if token in _TERRAIN_NAMES else None


def battle_view_from_replay_state(state: dict) -> BattleView:
    """Single-state mapper - kept for ad-hoc/debugging use on one turn in
    isolation. Cannot reconstruct fainted teammates (see module docstring):
    a fainted mon is just absent from `available_switches` with no way to
    tell "fainted" from "never on this team" without the rest of the replay.
    Use battle_views_from_replay for real work (e.g. dataset building).
    """
    bench_mons = sorted(state["available_switches"], key=_species_key)
    bench = [_replay_pokemon_view(mon) for mon in bench_mons]
    return BattleView(
        my_active=_replay_pokemon_view(state["player_active_pokemon"]),
        my_bench=_pad_bench(bench),
        opp_active=_replay_pokemon_view(state["opponent_active_pokemon"]),
        opp_remaining_fraction=state["opponents_remaining"] / 6,
        my_hazards=_replay_hazards(state["player_conditions"]),
        opp_hazards=_replay_hazards(state["opponent_conditions"]),
        weather=_replay_weather(state["weather"]),
        terrain=_replay_terrain(state["battle_field"]),
        # Phase 4: my_used_tera is exact from a single state (can_tera is
        # verified monotonic - see module docstring); opp_used_tera stays
        # the documented-False default (live-adapter-only, no
        # opponent_can_tera field exists). Hazard-stack-layer fields also
        # stay their documented-0 defaults - no equivalent in this schema.
        my_used_tera=not state["can_tera"],
    )


def battle_views_from_replay(states: list) -> list:
    """Maps a whole replay's state sequence to one BattleView per state,
    reconstructing fainted teammates that a single isolated state can't
    (see module docstring and battle_view_from_replay_state). Needs the full
    sequence: "this teammate fainted" can only be inferred by noticing it's
    no longer active or switchable after having been seen earlier in the
    same replay.

    Unverified edge case, not seen in the downloaded sample and not chased
    down: if Metamon's replay parser records Illusion Zoroark under its
    disguised species rather than its true one, the roster here would see
    two "names" for one physical teammate, which could push a reconstructed
    bench past 5 slots and trip _pad_bench's assert. Left as a loud failure
    rather than silently handled, consistent with why that assert exists.
    """
    my_protect_streaks, opp_protect_streaks = _replay_protect_streaks(states)

    seen: Dict[str, dict] = {}
    views = []
    for state, my_streak, opp_streak in zip(states, my_protect_streaks, opp_protect_streaks):
        active = state["player_active_pokemon"]
        switches = state["available_switches"]
        seen[_species_key(active)] = active
        for mon in switches:
            seen[_species_key(mon)] = mon

        alive_names = {_species_key(active)} | {_species_key(m) for m in switches}
        bench_entries = {_species_key(m): (m, False) for m in switches}
        for name, mon in seen.items():
            if name not in alive_names:
                bench_entries[name] = (mon, True)

        bench = [
            _replay_pokemon_view_fainted(mon) if fainted else _replay_pokemon_view(mon)
            for name, (mon, fainted) in sorted(bench_entries.items())
        ]

        views.append(BattleView(
            my_active=_replay_pokemon_view(active, protect_counter=my_streak),
            my_bench=_pad_bench(bench),
            opp_active=_replay_pokemon_view(
                state["opponent_active_pokemon"], protect_counter=opp_streak
            ),
            opp_remaining_fraction=state["opponents_remaining"] / 6,
            my_hazards=_replay_hazards(state["player_conditions"]),
            opp_hazards=_replay_hazards(state["opponent_conditions"]),
            weather=_replay_weather(state["weather"]),
            terrain=_replay_terrain(state["battle_field"]),
            # Phase 4: same exact/documented-gap accounting as
            # battle_view_from_replay_state (see module docstring) - a
            # single state is sufficient for my_used_tera, no history needed.
            my_used_tera=not state["can_tera"],
        ))
    return views


# --- encoding ----------------------------------------------------------------


def _one_hot_status(status: Optional[Status]) -> np.ndarray:
    vec = np.zeros(len(_STATUSES), dtype=np.float32)
    if status in _STATUSES:
        vec[_STATUSES.index(status)] = 1.0
    return vec


def _multi_hot_types(types: Tuple[PokemonType, ...]) -> np.ndarray:
    vec = np.zeros(len(_ALL_TYPES), dtype=np.float32)
    for t in types:
        vec[_ALL_TYPES.index(t)] = 1.0
    return vec


def _boost_vector(boosts: Dict[str, int]) -> np.ndarray:
    return np.array([boosts[name] / 6.0 for name in _BOOST_NAMES], dtype=np.float32)


def _base_stat_vector(base_stats: Dict[str, int]) -> np.ndarray:
    return np.array(
        [base_stats[name] / _BASE_STAT_SCALE for name in _STAT_NAMES], dtype=np.float32
    )


def _item_vector(item: Optional[str]) -> np.ndarray:
    vec = np.zeros(len(_ITEM_VOCAB) + 1, dtype=np.float32)
    if item is None:
        return vec
    if item in _ITEM_VOCAB:
        vec[_ITEM_VOCAB.index(item)] = 1.0
    else:
        vec[-1] = 1.0  # known item, outside the curated vocab
    return vec


def _move_summary_vector(moves: MoveSummary) -> np.ndarray:
    flags = np.array(
        [
            1.0 if moves.has_recovery else 0.0,
            1.0 if moves.has_hazard_setup else 0.0,
            1.0 if moves.has_hazard_removal else 0.0,
            1.0 if moves.has_setup_boost else 0.0,
            1.0 if moves.has_pivot else 0.0,
            1.0 if moves.has_priority else 0.0,
        ],
        dtype=np.float32,
    )
    max_bp = np.array([moves.max_base_power / _MAX_BASE_POWER_SCALE], dtype=np.float32)
    coverage = np.zeros(len(_ALL_TYPES), dtype=np.float32)
    for t in moves.move_types:
        coverage[_ALL_TYPES.index(t)] = 1.0
    return np.concatenate([flags, max_bp, coverage])


def _move_slot_vector(
    move: MoveView,
    user_types: Tuple[PokemonType, ...],
    defender_types: Tuple[PokemonType, ...],
    defender_ability: Optional[str],
    defender_item: Optional[str],
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    user_grounded: bool = True,
) -> np.ndarray:
    """One move slot's full feature block: MoveView's static fields plus the
    battle-context-dependent numbers (type effectiveness against
    `defender_types`/`defender_ability`/`defender_item`, STAB against
    `user_types`, and - Phase 3 - the current `weather`/`terrain`'s effect on
    this specific move) - see module docstring for why these live here and
    not on MoveView itself. Effectiveness/STAB/base_power/accuracy default
    to 0.0 for an unknown slot, a not-opponent-directed move (Stealth Rock
    et al - see _OPPONENT_DIRECTED_TARGETS), or a move with no resolvable
    type - the accompanying `known`/`targets_opponent` flags tell a model
    whether that 0.0 means "computed and neutral-immune" or "not
    applicable," same "don't let an information gap silently read as a real
    signal" convention _is_hazard_immune's own docstring states explicitly.

    `weather`/`terrain`/`user_grounded` default to "no weather/terrain,
    grounded" so every pre-Phase-3 caller (including this module's own
    tests) keeps working unchanged - a neutral default, not a real battle
    fact assumption.
    """
    type_vec = np.zeros(len(_ALL_TYPES), dtype=np.float32)
    stab = 0.0
    effectiveness = 0.0
    accuracy = move.accuracy
    needs_charge_turn = 0.0
    if move.known:
        effective_type = move.type
        effective_base_power = move.base_power

        # Weather Ball: real type/power become weather-dependent (see
        # module docstring) - override BEFORE the type one-hot/STAB/
        # effectiveness/base_power scalars below are computed from it.
        if move.move_id == "weatherball" and weather in _WEATHER_BALL_TYPE_BY_WEATHER:
            effective_type = _WEATHER_BALL_TYPE_BY_WEATHER[weather]
            effective_base_power *= 2

        # Terrain power boost: applies to the (possibly weather-ball-
        # overridden) effective type, for a grounded user - the two stack
        # coherently the same way they would in a real damage calculation.
        if (
            terrain in _TERRAIN_BOOST_TYPE
            and effective_type == _TERRAIN_BOOST_TYPE[terrain]
            and user_grounded
        ):
            effective_base_power *= _TERRAIN_POWER_MULTIPLIER

        # Thunder/Hurricane/Blizzard: real per-move weather-conditional
        # accuracy override (see module docstring) - MoveView.accuracy
        # itself stays the static dex value.
        overrides = _WEATHER_ACCURACY_OVERRIDES.get(move.move_id)
        if overrides is not None and weather in overrides:
            accuracy = overrides[weather]

        # Solar Beam/Solar Blade skip their charge turn in sun; every other
        # charge move always needs one (see module docstring).
        if move.is_charge_move:
            needs_charge_turn = (
                0.0 if (move.move_id in _SUN_CHARGE_SKIP_MOVES and weather == "sunnyday") else 1.0
            )

        if effective_type is not None:
            type_vec[_ALL_TYPES.index(effective_type)] = 1.0
            stab = 1.0 if effective_type in user_types else 0.0
            if move.targets_opponent and defender_types:
                effectiveness = _type_multiplier(
                    effective_type, defender_types, defender_ability, defender_item
                )
    category_vec = _one_hot(move.category, _MOVE_CATEGORIES)
    secondary_kind_vec = _one_hot(move.secondary_kind, _SECONDARY_KINDS)
    priority_scaled = max(min(move.priority, _PRIORITY_SCALE), -_PRIORITY_SCALE) / _PRIORITY_SCALE
    scalars = np.array(
        [
            1.0 if move.known else 0.0,
            stab,
            effective_base_power / _MAX_BASE_POWER_SCALE if move.known else 0.0,
            accuracy if move.known else 0.0,
            priority_scaled if move.known else 0.0,
            1.0 if move.targets_opponent else 0.0,
            effectiveness,
            move.secondary_chance,
            move.self_boost_chance,
            move.self_boost_magnitude,
            1.0 if move.fixed_damage else 0.0,
            1.0 if move.multi_hit else 0.0,
            1.0 if move.is_contact else 0.0,
            1.0 if move.is_sound else 0.0,
            1.0 if move.is_punch else 0.0,
            1.0 if move.is_bite else 0.0,
            1.0 if move.is_pulse else 0.0,
            1.0 if move.is_bullet else 0.0,
            1.0 if move.is_wind else 0.0,
            1.0 if move.is_protect_counter else 0.0,
            1.0 if move.bypasses_protect else 0.0,
            move.recoil_fraction,
            move.drain_fraction,
            1.0 if move.is_self_ko else 0.0,
            needs_charge_turn,
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([type_vec, category_vec, secondary_kind_vec, scalars])
    assert vec.shape == (_MOVE_VEC_LEN,)
    return vec


def _move_slots_vector(
    move_slots: Tuple[MoveView, ...],
    user_types: Tuple[PokemonType, ...],
    defender_types: Tuple[PokemonType, ...],
    defender_ability: Optional[str],
    defender_item: Optional[str],
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    user_grounded: bool = True,
) -> np.ndarray:
    return np.concatenate(
        [
            _move_slot_vector(
                m, user_types, defender_types, defender_ability, defender_item,
                weather=weather, terrain=terrain, user_grounded=user_grounded,
            )
            for m in move_slots
        ]
    )


def _encode_pokemon(
    view: PokemonView,
    defender: PokemonView,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
) -> np.ndarray:
    """defender is the Pokemon `view`'s own moves are scored against for
    per-move type effectiveness/STAB (see module docstring: this needs both
    mons' state, unlike the rest of a single Pokemon's block) - always the
    opponent's current active Pokemon for a my-side view, always my current
    active Pokemon for the opponent's active view (see encode()'s call
    sites) - the real defending target a switched-in move would actually
    face right now, not a hypothetical future matchup.

    weather/terrain (Phase 3) are the battle's own current values, threaded
    down to every one of `view`'s move slots - including bench slots, same
    "hypothetical if this mon were on the field" projection the existing
    per-move effectiveness/STAB already applies to a bench Pokemon's moves
    (see module docstring). Both default to None so pre-Phase-3 callers
    (including this module's own already-anchored tests) are unaffected.
    """
    user_grounded = _is_grounded(view)
    return np.concatenate(
        [
            np.array([1.0 if view.known else 0.0], dtype=np.float32),
            np.array([view.hp_fraction], dtype=np.float32),
            np.array([1.0 if view.fainted else 0.0], dtype=np.float32),
            _one_hot_status(view.status),
            _multi_hot_types(view.types),
            _boost_vector(view.boosts),
            _base_stat_vector(view.base_stats),
            _item_vector(view.item),
            _move_summary_vector(view.moves),
            _move_slots_vector(
                view.move_slots, view.types, defender.types, defender.ability, defender.item,
                weather=weather, terrain=terrain, user_grounded=user_grounded,
            ),
            # Phase 4: toxic_counter, has_leech_seed, has_substitute,
            # is_confused - placed BEFORE the Phase 2 runtime-state block
            # below (preparing is the earliest already-anchored tail scalar
            # in this concatenation, vec[-4] - see module docstring) so
            # vec[-4]/-3/-2/-1 stay unchanged.
            np.array(
                [
                    min(view.toxic_counter, _TOXIC_COUNTER_SCALE) / _TOXIC_COUNTER_SCALE,
                    1.0 if view.has_leech_seed else 0.0,
                    1.0 if view.has_substitute else 0.0,
                    1.0 if view.is_confused else 0.0,
                ],
                dtype=np.float32,
            ),
            # Phase 2 runtime state (preparing, semi_invulnerable,
            # must_recharge) - placed BEFORE protect_counter so
            # protect_counter stays the last field of this block (see
            # PokemonView's own docstring for why that ordering matters).
            np.array(
                [
                    1.0 if view.preparing else 0.0,
                    1.0 if view.semi_invulnerable else 0.0,
                    1.0 if view.must_recharge else 0.0,
                ],
                dtype=np.float32,
            ),
            np.array(
                [min(view.protect_counter, _PROTECT_COUNTER_SCALE) / _PROTECT_COUNTER_SCALE],
                dtype=np.float32,
            ),
        ]
    )


def _hazard_vector(hazards: set) -> np.ndarray:
    return np.array(
        [1.0 if token in hazards else 0.0 for token in _HAZARD_TOKENS], dtype=np.float32
    )


def _one_hot(value: Optional[str], vocab: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        vec[list(vocab).index(value)] = 1.0
    return vec


def _type_multiplier(
    attacking: PokemonType,
    defending_types: Tuple[PokemonType, ...],
    defending_ability: Optional[str] = None,
    defending_item: Optional[str] = None,
) -> float:
    """Type-chart multiplier, corrected for the defender's ability when it's
    known and grants a hard type immunity (see _TYPE_IMMUNITY_ABILITIES) or is
    Wonder Guard (blocks anything that isn't already super-effective - a
    different rule shape, since it depends on the raw multiplier itself
    rather than a fixed type, so it's handled as its own case rather than
    folded into the table), and for two item-based exceptions verified
    against Showdown's own current sim source (see module docstring for the
    exact code read and reasoning, not just the summary here):

    - Air Balloon: a Ground-type attack against a holder is 0.0, regardless
      of typing/ability - approximated as "currently holding" (no
      turn-scoped "already popped this turn" tracking - PokemonView has
      none - same documented-simplification convention as
      _is_hazard_immune's own exclusions).
    - Ring Target: cancels a defender's TYPE-CHART-based immunity (0x from
      typing alone) but NOT an ability-granted one (Levitate, Water Absorb,
      Wonder Guard, ...) - verified in Showdown's real
      Pokemon.runImmunity/isGrounded: Ring Target's negateImmunity bypasses
      only the hasType('Flying') check, never the separate Levitate/Air
      Balloon branches in the same function. Implemented by computing each
      of the (up to two) defending types' own multiplier SEPARATELY and
      overriding only a 0.0 result to 1.0 before multiplying them together -
      not via one combined damage_multiplier(d1, d2) call, which can't
      attribute a 0x result back to a single contributing type. This can
      turn an immunity into neutral, never into a new weakness - the other
      type's real resistance/weakness still applies on top.
    """
    if defending_ability == "wonderguard":
        d1 = defending_types[0]
        d2 = defending_types[1] if len(defending_types) > 1 else None
        raw = attacking.damage_multiplier(d1, d2, type_chart=_TYPE_CHART)
        return raw if raw > 1 else 0.0

    if defending_item == "airballoon" and attacking == PokemonType.GROUND:
        return 0.0

    immune_type = _TYPE_IMMUNITY_ABILITIES.get(defending_ability or "")
    if immune_type is not None and attacking == immune_type:
        return 0.0

    d1 = defending_types[0]
    d2 = defending_types[1] if len(defending_types) > 1 else None
    m1 = attacking.damage_multiplier(d1, type_chart=_TYPE_CHART)
    m2 = attacking.damage_multiplier(d2, type_chart=_TYPE_CHART) if d2 is not None else 1.0
    if defending_item == "ringtarget":
        if m1 == 0.0:
            m1 = 1.0
        if m2 == 0.0:
            m2 = 1.0
    return m1 * m2


def _is_grounded(mon: PokemonView) -> bool:
    """Whether mon is grounded - the real Showdown predicate that governs
    both entry-hazard immunity (_is_hazard_immune below) and this phase's
    terrain power-boost/status-immunity checks (see module docstring for
    the factoring rationale: this used to be duplicated inline inside
    _is_hazard_immune, this phase's own Edge Cases note asked for a shared
    helper once a third call site appeared).

    Verified against the real Pokemon.isGrounded in
    pokemon-showdown/sim/pokemon.ts: Flying-type, Levitate, and Air Balloon
    (while held) all block groundedness. Iron Ball/Gravity/Magnet
    Rise/Ingrain/Smack Down (force or restore groundedness the other
    direction) remain deliberately unmodeled - same named-simplification
    convention as _is_hazard_immune's own accounting; PokemonView tracks
    none of that volatile/field state.

    Defaults to True (grounded) when types are unknown/empty - the
    direction that keeps an information gap from ever granting an unearned
    immunity on either downstream feature (_is_hazard_immune's own
    already-anchored "unknown Pokemon must not read as hazard-immune" test
    depends on this).
    """
    if not mon.types:
        return True
    if PokemonType.FLYING in mon.types:
        return False
    if mon.ability == "levitate":
        return False
    if mon.item == "airballoon":
        return False
    return True


def _ability_speed_doubled(
    mon: PokemonView, weather: Optional[str], terrain: Optional[str]
) -> bool:
    """Whether mon's known ability doubles its Speed under the battle's
    current weather/terrain (Swift Swim/Chlorophyll/Sand Rush/Slush Rush +
    weather, Surge Surfer + Electric Terrain) - see module docstring for
    the exact abilities.ts onModifySpe verification. None of these five
    require groundedness, unlike the terrain checks below.
    """
    ability = mon.ability or ""
    if weather is not None and _WEATHER_SPEED_ABILITIES.get(ability) == weather:
        return True
    return terrain is not None and _TERRAIN_SPEED_ABILITIES.get(ability) == terrain


def _terrain_sleep_immune(mon: PokemonView, terrain: Optional[str]) -> bool:
    """Electric Terrain and Misty Terrain both block sleep for a grounded
    Pokemon (see module docstring for the real onSetStatus verification).
    """
    return terrain in _SLEEP_BLOCKING_TERRAINS and _is_grounded(mon)


def _terrain_status_immune(mon: PokemonView, terrain: Optional[str]) -> bool:
    """Misty Terrain blocks ANY status (not just sleep) for a grounded
    Pokemon - a strictly broader real mechanic than _terrain_sleep_immune,
    kept as its own boolean rather than derived, since a caller may care
    about either scope independently (see module docstring).
    """
    return terrain in _STATUS_BLOCKING_TERRAINS and _is_grounded(mon)


def _is_hazard_immune(mon: PokemonView) -> bool:
    """Whether mon is immune to ground-based entry hazards (Spikes, Toxic
    Spikes, Sticky Web - and, via Heavy-Duty Boots specifically, Stealth
    Rock too) - a real Showdown mechanic that's a completely separate rule
    from the type chart / _type_multiplier above (Spikes isn't a Ground-type
    move that gets multiplied against the defender's types, it's a hard
    immunity check), and was invisible to the encoder entirely before this -
    found via real replay inspection of a trained PPO policy (see CLAUDE.md's
    Phase 3 status): it spammed Spikes for 32 consecutive turns against a
    Flying-type opponent Spikes could never affect, with nothing in the
    vector able to tell it why every one of those turns was wasted.

    Originally named _is_grounded and Flying/Levitate-only - an independent
    review (2026-08-09) measured on real replay data (300 replays, 20,610
    active-mon states) that Heavy-Duty Boots is the #2 most common immunity
    source (5.19% of states, behind Flying's 17.0%, ahead of Levitate's
    2.46%) and the *only* one of the three that also blocks Stealth Rock -
    the original version confidently mislabeled every boots holder as
    hazard-vulnerable, sitting right next to the raw item feature
    (_ITEM_VOCAB already includes "heavydutyboots") without using it. Fixed
    and renamed to describe what the feature actually claims to model.

    Rarer immunity/anti-immunity interactions deliberately NOT modeled, same
    named-simplification convention as _TYPE_IMMUNITY_ABILITIES' own
    exclusions (Thick Fat, Soundproof, ...) and Air Balloon here (temporary,
    lost on any hit): Iron Ball and Gravity (both *ground* an otherwise-
    Flying/Levitate mon - the inverse direction, would need to be read as a
    de-immunizer, not modeled), Magnet Rise/Ingrain/Smack Down/Roost (turn-
    scoped volatile statuses PokemonView doesn't track at all). All measured
    rarer than Heavy-Duty Boots on the same sample - revisit if evidence
    says otherwise, same "not attempted, revisit if it matters" standard
    already used elsewhere in this module.

    Defaults to False (not immune) when types are unknown/empty (active
    Pokemon's types should always be known in practice on both adapters -
    this only guards the same team-preview-style edge case
    _active_matchup_score itself guards below) so an information gap never
    silently reads as a false immunity signal.

    Phase 3 (2026-08-26): the inline Flying/Levitate check here was
    factored out into the shared _is_grounded helper above (a third real
    call site - this phase's own terrain power-boost/status-immunity
    checks - appeared, matching this function's own docstring note above
    about when a shared helper becomes worth it). _is_grounded additionally
    checks Air Balloon, which this function's Flying/Levitate-only version
    never did - a real, incidental correctness fix (Air Balloon genuinely
    blocks groundedness in real Showdown, verified against sim/pokemon.ts's
    isGrounded - see _is_grounded's own docstring), not a deliberate
    hazard-behavior change made for its own sake. Verified this refactor
    changes no existing test's result: heavydutyboots stays its own
    additional-immunity check layered on top (Boots doesn't affect
    groundedness itself, it's a separate hazard-specific exception), and
    _is_grounded's own "unknown types -> True (grounded)" default means
    `not _is_grounded(mon)` is False for an unknown Pokemon, same as this
    function's prior direct `return False` - the "unknown must not read as
    hazard-immune" contract is unchanged.
    """
    if mon.item == "heavydutyboots":
        return True
    return not _is_grounded(mon)


def _active_matchup_score(view: BattleView) -> float:
    """Mirrors evaluation.type_matchup_score's semantics (offense - defense,
    each the best multiplier available across a dual typing), but built from
    bare PokemonType tuples rather than full poke-env Pokemon objects, since
    a replay-derived PokemonView isn't a real Pokemon instance. Each side's
    ability (if known) is folded in via _type_multiplier.
    """
    my_types, opp_types = view.my_active.types, view.opp_active.types
    if not my_types or not opp_types:
        return 0.0
    offense = max(
        _type_multiplier(t, opp_types, view.opp_active.ability, view.opp_active.item)
        for t in my_types
    )
    defense = max(
        _type_multiplier(t, my_types, view.my_active.ability, view.my_active.item)
        for t in opp_types
    )
    return offense - defense


def encode(view: BattleView) -> np.ndarray:
    vec = np.concatenate(
        [
            _encode_pokemon(view.my_active, defender=view.opp_active,
                             weather=view.weather, terrain=view.terrain),
            *[_encode_pokemon(p, defender=view.opp_active,
                               weather=view.weather, terrain=view.terrain)
              for p in view.my_bench],
            _encode_pokemon(view.opp_active, defender=view.my_active,
                             weather=view.weather, terrain=view.terrain),
            np.array([view.opp_remaining_fraction], dtype=np.float32),
            _hazard_vector(view.my_hazards),
            _hazard_vector(view.opp_hazards),
            _one_hot(view.weather, _WEATHER_NAMES),
            _one_hot(view.terrain, _TERRAIN_NAMES),
            # Phase 4: global (not per-Pokemon) hazard-stack-layer counts and
            # used_tera booleans - inserted BEFORE the Phase 3 6-scalar block
            # below, not just before _active_matchup_score in isolation -
            # _ability_speed_doubled's pair is the earliest already-anchored
            # tail scalar in this concatenation (vec[-9] - see module
            # docstring), so this keeps vec[-9] through vec[-1] unchanged.
            np.array(
                [
                    min(view.my_spikes_layers, _SPIKES_MAX_LAYERS) / _SPIKES_MAX_LAYERS,
                    min(view.my_toxic_spikes_layers, _TOXIC_SPIKES_MAX_LAYERS) / _TOXIC_SPIKES_MAX_LAYERS,
                    min(view.opp_spikes_layers, _SPIKES_MAX_LAYERS) / _SPIKES_MAX_LAYERS,
                    min(view.opp_toxic_spikes_layers, _TOXIC_SPIKES_MAX_LAYERS) / _TOXIC_SPIKES_MAX_LAYERS,
                    1.0 if view.my_used_tera else 0.0,
                    1.0 if view.opp_used_tera else 0.0,
                ],
                dtype=np.float32,
            ),
            # Phase 3: global (not per-Pokemon) side-level booleans, mirroring
            # my_active_hazard_immune/opp_active_hazard_immune's own tail
            # placement - inserted BEFORE _active_matchup_score, not after
            # the hazard-immunity pair, so those two already-anchored tests
            # (vec[-3]/-2/-1) need no changes (see module docstring).
            np.array(
                [
                    1.0 if _ability_speed_doubled(view.my_active, view.weather, view.terrain) else 0.0,
                    1.0 if _ability_speed_doubled(view.opp_active, view.weather, view.terrain) else 0.0,
                    1.0 if _terrain_sleep_immune(view.my_active, view.terrain) else 0.0,
                    1.0 if _terrain_sleep_immune(view.opp_active, view.terrain) else 0.0,
                    1.0 if _terrain_status_immune(view.my_active, view.terrain) else 0.0,
                    1.0 if _terrain_status_immune(view.opp_active, view.terrain) else 0.0,
                ],
                dtype=np.float32,
            ),
            np.array([_active_matchup_score(view)], dtype=np.float32),
            np.array(
                [
                    1.0 if _is_hazard_immune(view.my_active) else 0.0,
                    1.0 if _is_hazard_immune(view.opp_active) else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
    )
    assert vec.shape == (VECTOR_LEN,)
    return vec
