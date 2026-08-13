// M4: direct C++ port of battle_engine/evaluation.py's evaluate() - the
// default leaf evaluator for M6's search (see the Scope decision in
// plans/precious-crafting-bachman.md for why the hand-crafted eval, not a
// learned model, is the default). Every weight/formula here must match the
// Python source exactly - it's cited inline per function, but
// battle_engine/evaluation.py is the ground truth if anything here is
// ambiguous.
#pragma once

#include "be/battle_state.hpp"

namespace be {

// Same values as evaluation.py's module-level constants of the same name -
// keep these in sync if the Python side ever changes.
inline constexpr float kHpFractionWeight = 1.0f;
inline constexpr float kAliveCountWeight = 0.5f;
inline constexpr float kTypeMatchupWeight = 0.5f;
inline constexpr float kStatusWeight = 0.3f;
inline constexpr float kHazardLayerWeight = 0.15f;
inline constexpr float kHazardPresenceWeight = 0.2f;
inline constexpr float kSpeedControlWeight = 0.2f;

// Port of evaluation.py's type_matchup_score(mon, opponent): how favorable
// mon's typing is against opponent's - offense minus defense, each the
// *best* (most damage-dealing) multiplier available across a dual typing.
// offense = max over mon's own 1-2 types t of "how hard opponent takes t" =
//   max_t( kTypeChart[opponent.type1][t] * kTypeChart[opponent.type2][t] ),
//   skipping the type2 factor entirely if opponent is single-typed (treat
//   as if that factor were 1, i.e. Type::None contributes no multiplier).
// defense = the same computation with mon and opponent swapped (how hard
//   mon takes opponent's best type).
// Returns offense - defense; positive means `mon` has the better matchup.
// Note the array index direction: kTypeChart[defender][attacker] (see
// pokedex_table.hpp's own comment on this - it trips people up).
float type_matchup_score(const PokemonSlot& mon, const PokemonSlot& opponent);

// Port of evaluation.py's _hazard_score(side_conditions): Spikes/Toxic
// Spikes contribute kHazardLayerWeight per stacked layer; Stealth Rock/
// Sticky Web each contribute a flat kHazardPresenceWeight if present.
float hazard_score(const SideConditions& conditions);

// Port of evaluation.py's _boosted_speed(mon) + _speed_control_score():
// compute each side's real (stat-stage-boosted, paralysis-halved) Speed via
// - stat = mon.spe_stat if known (>= 0), else estimate from
//   mon.base_stats.spe + mon.level (see PokemonSlot's own doc comment for
//   the exact formula - same one damage.py's estimate_stat() uses)
// - multiplier = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
// - halve the result if mon.status == Status::Par
// Returns +1.0 if my_active's boosted speed is strictly greater, -1.0 if
// opp_active's is strictly greater, 0.0 on an exact tie.
float speed_control_score(const PokemonSlot& my_active, const PokemonSlot& opp_active);

// Score `state` from "my" side's perspective - positive favors my_team /
// my_active. Direct port of evaluation.py's evaluate():
//   score  = kHpFractionWeight   * (sum of my revealed teams' hp_fraction
//                                    minus opp revealed team's hp_fraction)
//   score += kAliveCountWeight   * (count of my non-fainted, revealed slots
//                                    minus opp's)
//   score += kStatusWeight       * (count of opp's status-afflicted revealed
//                                    slots minus mine) - note the sign: MY
//                                    status hurts me, so it's opp-minus-mine
//   score += hazard_score(opp_hazards) - hazard_score(my_hazards)
//   if both sides have an active (non -1) slot:
//     score += kTypeMatchupWeight  * type_matchup_score(my_active, opp_active)
//     score += kSpeedControlWeight * speed_control_score(my_active, opp_active)
// "Status-afflicted" mirrors evaluation.py's `status not in (None, FNT)`
// check - Status::None and Status::Fnt both count as NOT afflicted.
// Unrevealed opponent slots (PokemonSlot::revealed == false) must be
// skipped entirely from every sum above, matching how battle.opponent_team
// being a dict-of-only-revealed-entries behaves in the Python original -
// don't count them as fainted, don't count them as 0 HP, just skip them.
float evaluate(const BattleState& state);

}  // namespace be
