// M5: the approximate C++ forward model - the state-transition function
// M6's MCTS calls once per simulated turn. Tier 1 mechanics only (see the
// tiered mechanics scope in plans/precious-crafting-bachman.md): turn-order
// resolution, SAMPLED (not averaged) crit/damage-roll/accuracy/multi-hit
// reusing battle_engine/damage.py's documented formula and crit table,
// fainting + forced replacement, and switch-in hazard chip damage - the
// exact gap search.py's own docstring names as invisible to the existing
// (Phase 1/2) search.
//
// Status-category moves are a turn-cost-only no-op in Tier 1: their real
// field effects (stat boosts, hazard setup, status infliction, screens,
// ...) are Tier 2, explicitly deferred. A Status move still has a real
// priority bracket and still consumes the turn - both handled by
// resolve_turn - it just deals no damage and changes no other state.
#pragma once

#include "be/action.hpp"
#include "be/battle_state.hpp"
#include "be/movedex_table.hpp"
#include "be/rng.hpp"

namespace be {

// Mirrors damage.py's _CRIT_CHANCE_BY_RATIO exactly (the gen7+ crit table,
// sourced from Showdown's own sim/battle-actions.ts): ratio 0 and 1 both
// map to the base 1/24 rate, 2 -> 1/8, 3 -> 1/2, 4 -> guaranteed. Ratio 6
// (poke-env's "willCrit" marker - see scripts/dump_movedex.py) is also
// guaranteed (1.0) despite not appearing in damage.py's own dict, since
// that Python table only ever sees ratios 0-4 there (its _crit_chance()
// special-cases 6 separately, immediately above the dict lookup) - do the
// same here rather than adding a "6" entry to a table modeled on the
// original 5-entry one.
float crit_chance(int crit_ratio);

// What happened after one simulated turn - the signal M6's MCTS uses to
// decide whether a rollout continues, needs a forced-replacement decision
// injected, or has reached a terminal (won/lost) leaf.
enum class TurnResolution {
  kContinue,     // both sides still have a usable active Pokemon
  kMyFainted,    // my active fainted this turn and needs a replacement switch
  kOppFainted,   // opponent's active fainted this turn and needs a replacement switch
  kBothFainted,  // both fainted the same turn (real Showdown case - e.g. a
                 // priority tie where both attacks land simultaneously)
  kTerminal,     // one side has no non-fainted Pokemon left anywhere on
                 // its team - the battle itself is over
};

// A real ambiguity worth naming explicitly (found while writing this
// milestone's own tests, not a hypothetical): kTerminal is only reliably
// detectable for `state.my_team`, since a live translated battle always
// reveals all 6 of your own slots (battle_state.hpp's own doc comment on
// PokemonSlot::revealed). For `state.opp_team` under Tier 1's
// revealed-only opponent model, "every REVEALED opp slot is fainted" does
// NOT mean the opponent's real team is defeated - they may have
// genuinely-unrevealed Pokemon left that this state simply doesn't know
// about. Treating "unrevealed" as equivalent to "doesn't exist" when
// checking for kTerminal would be a real, silent overreach beyond what
// Tier 1's opponent model is entitled to claim - decide deliberately how
// resolve_turn's own kTerminal check should treat this (e.g. only ever
// return kTerminal for state.my_team being wiped, and use kOppFainted for
// every opponent-side case even when it might, in the real game, actually
// be defeat), not by accident via whatever a same-shaped loop over both
// team arrays happens to do.

// Samples ONE stochastic damage instance for `move` used by `attacker`
// against `defender`, as a fraction of defender's real max HP - the same
// HP-fraction convention damage.py's expected_damage() uses, but with
// roll/crit/accuracy/multi-hit all SAMPLED here instead of folded into an
// expectation (the actual value-add over Phase 1/2's search - see the
// plan's Scope decision on why that's the point of this phase).
//
// Returns 0.0f for:
//   - a miss, sampled against MovedexEntry::accuracy (-1 = "always hits",
//     matching move.accuracy's own -1 sentinel)
//   - a Status-category move (see this header's own module comment - its
//     Tier-1 effect is turn-cost-only, handled by resolve_turn, not here)
//   - a move whose type `defender` is immune to: checked explicitly for
//     fixed-damage moves (matching damage.py's _fixed_damage() handling),
//     and falls out for free on base-power moves since kTypeChart already
//     contains a real 0.0 entry for every immunity
//
// Fixed-damage moves (MovedexEntry::fixed_damage != 0 - Seismic Toss,
// Dragon Rage, ...) skip STAB/crit/roll entirely, matching real Showdown
// mechanics for these moves (same as damage.py's _fixed_damage()), but are
// still subject to the accuracy check and the type-immunity check above.
//
// Named simplifications, each real and deliberate (see damage.py's own
// docstring for the ones this project already accepts more broadly):
//   - attacker/defender atk/def/spa/spd are ALWAYS estimated from
//     base_stats + level (the same unknown-stat formula damage.py's
//     estimate_stat() falls back to), even for my own team - narrower
//     than the Python original, which knows my own team's real stats
//     exactly. PokemonSlot has no known-exact field for these four stats
//     (unlike spe_stat, which eval.hpp's speed_control_score() already
//     needed) - a real, C++-forward-model-specific scope cut, not present
//     in damage.py itself. Revisit only if a benchmark shows this
//     specifically bottlenecks search quality.
//   - STAB is always the standard 1.5x - Adaptability's 2x isn't modeled
//     (no ability field on PokemonSlot; abilities generally are Tier 2).
//   - multi-hit sampling uses MovedexEntry::min_hits/max_hits: if equal,
//     that many hits, each independently re-rolling crit/damage-roll (but
//     NOT a separate accuracy check per hit - one whole-move accuracy
//     check up front, same simplification named in
//     scripts/dump_movedex.py's docstring for Triple Kick/Axel/Population
//     Bomb); if 2-5 (the only real gen-9 variable range), sample a hit
//     count via the standard distribution (35% 2, 35% 3, 15% 4, 15% 5).
float sample_damage_fraction(const PokemonSlot& attacker, const PokemonSlot& defender,
                              const MovedexEntry& move, Rng& rng);

// Applies switch-in hazard chip damage to `incoming` as it switches into a
// side with `hazards` active - the exact gap search.py's own docstring
// names as invisible to the Phase 1/2 search. Only ever changes
// incoming.hp_fraction (and incoming.fainted if hazards alone bring it to
// 0) - never status or boost_spe, since Sticky Web (stat drop) and Toxic
// Spikes (poison) are move-EFFECT-shaped, Tier 2, deferred; this function
// only handles the two hazards with a DIRECT damage component:
//   - Stealth Rock: 1/8 of incoming's max HP, scaled by incoming's own
//     typing's effectiveness against Rock (a quadruple-weak switch-in
//     takes 1/2 max HP; a Rock-resistant one takes less; a Rock-immune
//     type - none exist - would take none).
//   - Spikes: flat damage unaffected by typing EXCEPT that Ground-immune
//     types (via kTypeChart, i.e. Flying-type or an equivalent immunity)
//     take none entirely - 1/8 max HP at 1 layer, 1/6 at 2, 1/4 at 3.
//     (Levitate's grounding-immunity nuance isn't modeled - no ability
//     field on PokemonSlot, same abilities-are-Tier-2 scope cut as
//     sample_damage_fraction's STAB simplification above.)
// Sticky Web/Toxic Spikes presence in `hazards` is simply ignored by this
// function - it is not a bug that they cause no change here.
void apply_switch_in_hazards(PokemonSlot& incoming, const SideConditions& hazards);

// The state-transition function M6's MCTS calls once per simulated turn.
// Resolves BOTH sides' chosen actions simultaneously - turn order by
// priority bracket (MovedexEntry::priority; a switch action always goes
// before any move, real Showdown priority effectively +6), then Speed
// (PokemonSlot::spe_stat / boost_spe, same formula eval.hpp's
// speed_control_score() already implements - reuse it, don't
// re-derive), with a true 50/50 coin flip on an exact speed tie - and
// mutates `state` in place: HP fractions drop from sample_damage_fraction,
// a fainted mon's `fainted` flag is set, and a mon that switches in THIS
// turn (via a switch action, not a later forced replacement) takes hazard
// chip damage via apply_switch_in_hazards().
//
// Preconditions this function does NOT itself re-check (a caller bug there
// is a caller bug, not resolve_turn's responsibility - legality lives in
// exactly one place, action.hpp's legal_actions(), not duplicated here):
// `state` must already satisfy is_valid(), and my_action/opp_action must
// already be legal per legal_actions() for their respective sides.
//
// Does NOT perform a forced-replacement switch itself when a mon faints
// mid-turn - it only reports which side(s) need one via the returned
// TurnResolution. A caller (M6) issues that as a SEPARATE, single-agent
// decision (its own legal_actions() call restricted to switch actions,
// then whatever mechanism M6 uses to apply just that switch + hazard chip
// damage) - mirrors real Showdown's own two-phase "attack, then replace"
// turn structure, and keeps this function's contract to "resolve exactly
// the two actions passed in," not an open-ended chain of decisions. (M6's
// DecisionNode already treats forced-switch as its own tagged variant for
// exactly this reason - see the plan's M6 section.)
TurnResolution resolve_turn(BattleState& state, ActionId my_action, ActionId opp_action, Rng& rng);

}  // namespace be
