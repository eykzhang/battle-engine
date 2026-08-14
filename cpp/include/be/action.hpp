// M5: the fixed, state-independent action scheme decided once at team
// preview (see "Fixed, state-independent action scheme" in
// plans/precious-crafting-bachman.md) - the C++ engine's own scheme,
// distinct from both battle_engine/action_space.py's poke-env translation
// and battle_engine/dataset.py's 13-way Metamon scheme. Neither of those
// is actually fixed for a whole battle (bench membership shifts as
// different mons faint); this one is, by construction:
//
//   ActionId 0-5: switch to team-preview slot i - BattleState::my_team[i]
//     / opp_team[i], FIXED for the whole battle regardless of who's
//     currently active or which slots have since fainted. Legality (not
//     renumbering) is what rules out "switch to the already-active slot"
//     / "switch to a fainted slot" / "switch to an unrevealed slot."
//   ActionId 6-9: move slot 0-3 for whichever Pokemon is CURRENTLY active
//     on that side (not a fixed mon - "the active one," whoever that is
//     when the action is taken).
//
// Tera-variant actions are deferred (see the plan's Scope decision).
// Translation to/from a submittable poke-env BattleOrder happens ONLY at
// the root, once per real turn actually played (battle_engine/
// mcts_player.py, M7) - internal search nodes (M6) never touch poke-env's
// action numbering at all.
#pragma once

#include <vector>

#include "be/battle_state.hpp"

namespace be {

using ActionId = int8_t;

inline constexpr int kNumSwitchActions = 6;  // ActionId 0-5
inline constexpr int kMoveActionOffset = 6;  // ActionId 6-9 = offset + move slot 0-3
inline constexpr int kNumMoveActions = 4;

// Which side a legality query is for - matches BattleState's own
// my_*/opp_* field naming (not "Player1/Player2", to stay consistent with
// the rest of this header set).
enum class Side { Me, Opp };

// Legal ActionIds for `side` in `state`, using the fixed scheme above.
//
// Switch legality (ActionId 0-5): the target slot must be revealed, not
// fainted, and not already the active slot on that side.
//
// Move legality (ActionId 6-9): `side`'s active Pokemon must exist
// (my_active_slot / opp_active_slot != -1) for any move action to be
// well-defined at all - mirrors action_space.py's own
// _require_active_pokemon pattern for the identical reason (no active mon
// means no well-defined moveset to index into). A move slot i is legal
// iff active.moves[i] is non-empty (a known move id) - see
// PokemonSlot::moves's own doc comment.
//
// Tier 1's opponent-modeling limitation, applied twice over (a real,
// named limitation - see the plan's Scope decision, not a bug): for
// Side::Opp, BOTH switch targets and move slots are restricted to
// already-revealed information (opp_team[i].revealed for switches,
// opp_active.moves[i] non-empty for moves). An opponent's genuinely
// unrevealed bench Pokemon or unrevealed move can never appear as a legal
// action during search, even though it exists in the real game - this
// systematically understates the opponent's real option set, most
// visibly early in a battle before much has been revealed.
//
// A side with no legal actions at all (e.g. every non-active team member
// fainted/unrevealed AND the active mon's every known move slot is
// somehow empty) returns an empty vector - callers (M6's MCTS) must
// handle this rather than assume at least one action always exists;
// real Showdown's own equivalent ("Struggle") is Tier 2 scope, not
// modeled here.
std::vector<ActionId> legal_actions(const BattleState& state, Side side);

}  // namespace be
