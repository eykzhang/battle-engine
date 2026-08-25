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

namespace {
constexpr int kMetamonSwitchOffset = 4;  // dataset.py's _SWITCH_ACTIONS starts here
}  // namespace

ActionId metamon_switch_label_to_action_id(const BattleState& state, int metamon_label) {
  int bench_slot = metamon_label - kMetamonSwitchOffset;
  std::array<int, kMaxBench> slots = species_sorted_bench_slots(state);
  if (bench_slot < 0 || bench_slot >= kMaxBench || slots[static_cast<size_t>(bench_slot)] < 0) {
    return static_cast<ActionId>(-1);  // matches mcts.hpp's kNoAction value; not #included here to
                                        // keep action.hpp/action.cpp a lower-level module than mcts.hpp
  }
  return static_cast<ActionId>(slots[static_cast<size_t>(bench_slot)]);
}

int action_id_to_metamon_label(const BattleState& state, ActionId action) {
  if (action >= kMoveActionOffset) {
    return static_cast<int>(action) - kMoveActionOffset;  // 1:1, see this header's own doc comment
  }
  std::array<int, kMaxBench> slots = species_sorted_bench_slots(state);
  for (int i = 0; i < kMaxBench; ++i) {
    if (slots[static_cast<size_t>(i)] == static_cast<int>(action)) return kMetamonSwitchOffset + i;
  }
  return -1;
}

}  // namespace be
