#include "be/action.hpp"

namespace be {

std::vector<ActionId> legal_actions(const BattleState& state, Side side) {
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
