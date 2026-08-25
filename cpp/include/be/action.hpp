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

// M6b: Metamon's 13-way action scheme (battle_engine/dataset.py's
// ACTION_SPACE_SIZE - 0-3 move, 4-8 switch, 9-12 move+terastallize) mapped
// onto THIS scheme's fixed ActionId, ported from
// battle_engine/action_space.py's _switch_action_to_poke_env /
// _poke_env_switch_to_metamon species-sort logic (not re-derived - see this
// project's own "never invent a magic number" convention). Move labels need
// no translation function at all: Metamon move slot i (label i, i in 0-3)
// and ActionId's move slot i (kMoveActionOffset + i) both index the SAME
// active mon's SAME move slot directly - see action_id_to_metamon_label's
// own move-branch for the one place this identity is actually used.
//
// Both directions operate on state.my_team/my_active_slot ONLY, never
// opp_team - see battle_state.hpp's species_sorted_bench_slots() doc
// comment (which these both build on) for why: the OPPONENT's own mapping
// is obtained by calling these same functions on a mirror()-ed BattleState,
// not by adding a Side parameter here.

// Metamon switch label (dataset.py's _SWITCH_ACTIONS, 4-8) -> the ActionId
// (a team-preview slot 0-5) of the Pokemon at that species-sorted bench
// position. Returns kNoAction-shaped -1 if that position has no real
// target - fewer than 5 real non-active team members on this side (a
// bench<5 team; every real Showdown team has exactly 6, so this only
// reachably fires against a hand-built test fixture, never a live battle -
// still handled here, not asserted away, matching this codebase's
// defensive-fallback convention for an untested/degenerate shape rather
// than an out-of-bounds read). `metamon_label` outside [4, 8] is a caller
// bug - not checked here, same "callers own their inputs" convention as
// legal_actions()/select_ucb1_action elsewhere in this codebase.
ActionId metamon_switch_label_to_action_id(const BattleState& state, int metamon_label);

// The inverse direction, generalized to any of state's own ActionIds
// (moves included, unlike metamon_switch_label_to_action_id's switch-only
// scope - see this comment block's own top note on why moves need no
// species-sort at all): move ActionId 6-9 maps 1:1 onto Metamon label 0-3;
// switch ActionId 0-5 maps onto whichever Metamon switch label (4-8) that
// team-preview slot occupies in the species-sorted bench view. Returns -1
// if `action` doesn't correspond to any real bench position - shouldn't
// happen for a legal switch ActionId (legal_actions()'s switch-legality
// contract - revealed, not fainted, not active - is exactly
// species_sorted_bench_slots()'s own inclusion set, by construction) -
// verified by this milestone's own Catch2 tests rather than assumed.
// `action` outside [0, 9] is a caller bug, same convention as above.
int action_id_to_metamon_label(const BattleState& state, ActionId action);

}  // namespace be
