#include "be/action.hpp"

namespace be {

std::vector<ActionId> legal_actions(const BattleState& state, Side side) {
  // TODO(you): implement per action.hpp's doc comment on legal_actions().
  // Suggested shape:
  //   1. Pick this side's team array, active_slot, and hazards - i.e.
  //      `const auto& team = (side == Side::Me) ? state.my_team : state.opp_team;`
  //      and similarly for the active slot index.
  //   2. Switch actions (ActionId 0-5): for each i in [0, 6), push i if
  //      team[i].revealed && !team[i].fainted && i != active_slot.
  //   3. Move actions (ActionId 6-9): if active_slot == -1, there's no
  //      active Pokemon - skip move actions entirely (don't crash, just
  //      contribute nothing). Otherwise, for each i in [0, kNumMoveActions),
  //      push kMoveActionOffset + i if team[active_slot].moves[i] is
  //      non-empty.
  //   4. Side::Opp gets NO special-casing beyond what's already true of
  //      the data: opp_team[i].revealed and opp_active.moves[i] are
  //      already false/empty for anything not yet seen, so the same two
  //      loops above naturally enforce Tier 1's revealed-only restriction
  //      without a separate branch.
  //
  // cpp/tests/test_action.cpp has fixtures for a fully-revealed side, a
  // partially-revealed opponent, a no-active-Pokemon side, and the
  // "can't switch to self/fainted/unrevealed" cases. tests/
  // test_native_legality.py cross-checks this function against a real
  // poke-env battle's own available_moves/available_switches once the
  // pybind11 binding is built (see bindings/module.cpp).
  std::vector<ActionId> ret;

  const auto& team = (side == Side::Me) ? state.my_team : state.opp_team;
  const auto& active_slot_index = (side == Side::Me) ? state.my_active_slot : state.opp_active_slot;
  for (int i = 0; i < 6; i++) {
    if (i != active_slot_index && team[i].revealed && !team[i].fainted) ret.push_back(i);
  }
  if (active_slot_index > -1) {
    for (int i = 0; i < kNumMoveActions; i++) {
      if (!team[active_slot_index].moves[i].empty()) ret.push_back(i + kMoveActionOffset);
    }
  }
  return ret;
}

}  // namespace be
