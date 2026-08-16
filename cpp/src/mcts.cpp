#include "be/mcts.hpp"

#include <cstdint>

#include "be/eval.hpp"
#include "be/forward_model.hpp"

#include <limits>

namespace be {

namespace {

// "How many simulations have passed through this node," from whichever
// table you ask about. Equal for my_stats/opp_stats at a kDecision node
// (both increment together every visit); 0 for whichever pair is left
// unpopulated at a kForcedSwitch node, which select_ucb1_action already
// handles on its own (returns kNoAction for an empty actions list).
int total_visits(const std::vector<VisitStats>& stats) {
  int total = 0;
  for (const VisitStats& s : stats) total += s.visits;
  return total;
}

// Position of `action` within `actions`, or -1 if absent (e.g. `action`
// is kNoAction, or this side's pair is the unpopulated one at a
// kForcedSwitch node) - used during backup to recover the VisitStats
// index to update.
int find_index(const std::vector<ActionId>& actions, ActionId action) {
  for (int i = 0; i < (int)actions.size(); i++) {
    if (actions[i] == action) return i;
  }
  return -1;
}

std::vector<ActionId> legal_switch_actions(const BattleState& state, Side side) {
  std::vector<ActionId> switches;
  for (ActionId a : legal_actions(state, side)) {
    if (a < kNumSwitchActions) switches.push_back(a);
  }
  return switches;
}

}  // namespace

uint32_t pack_action_pair(ActionId my_action, ActionId opp_action) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(my_action)) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(opp_action));
}

float default_eval(const BattleState& state) { return evaluate(state); }

ActionId select_ucb1_action(const std::vector<ActionId>& actions, const std::vector<VisitStats>& stats,
                             int parent_visits, float exploration_constant) {
  if (actions.empty()) return kNoAction;
  float score = -std::numeric_limits<float>::infinity();
  int best = 0;
  for (int i = 0; i < (int)actions.size(); i++) {
    if (stats[i].visits == 0) return actions[i];
    float cur = stats[i].value_sum / stats[i].visits + exploration_constant * std::sqrt(std::log(parent_visits) / stats[i].visits);
    if (cur > score) { score = cur; best = i; }
  }
  return actions[best];
}

SearchResult search(const BattleState& root, const EvalFn& leaf_eval, int n_simulations, uint64_t seed) {
  Rng rng(seed);
  SearchNode root_node;
  root_node.my_actions = legal_actions(root, Side::Me);
  root_node.my_stats.resize(root_node.my_actions.size());
  root_node.opp_actions = legal_actions(root, Side::Opp);
  root_node.opp_stats.resize(root_node.opp_actions.size());

  for (int sim = 0; sim < n_simulations; sim++) {
    BattleState state = root;
    SearchNode* cur = &root_node;
    std::vector<std::tuple<SearchNode*, ActionId, ActionId>> path;
    float leaf_value = 0.0f;

    while (true) {
      if (cur->kind == NodeKind::kDecision) {
        ActionId my_pick = select_ucb1_action(cur->my_actions, cur->my_stats, total_visits(cur->my_stats),
                                               kDefaultExplorationConstant);
        ActionId opp_pick = select_ucb1_action(cur->opp_actions, cur->opp_stats, total_visits(cur->opp_stats),
                                                kDefaultExplorationConstant);
        uint32_t picks = pack_action_pair(my_pick, opp_pick);
        path.push_back({cur, my_pick, opp_pick});

        TurnResolution result = resolve_turn(state, my_pick, opp_pick, rng);

        auto it = cur->children.find(picks);
        if (it != cur->children.end()) {
          cur = it->second.get();
          continue;
        }

        auto new_node = std::make_unique<SearchNode>();
        switch (result) {
          case TurnResolution::kContinue: {
            new_node->kind = NodeKind::kDecision;
            new_node->my_actions = legal_actions(state, Side::Me);
            new_node->my_stats.resize(new_node->my_actions.size());
            new_node->opp_actions = legal_actions(state, Side::Opp);
            new_node->opp_stats.resize(new_node->opp_actions.size());
            break;
          }
          case TurnResolution::kMyFainted: {
            new_node->kind = NodeKind::kForcedSwitch;
            new_node->forced_switch_side = Side::Me;
            new_node->my_actions = legal_switch_actions(state, Side::Me);
            new_node->my_stats.resize(new_node->my_actions.size());
            break;
          }
          case TurnResolution::kOppFainted: {
            new_node->kind = NodeKind::kForcedSwitch;
            new_node->forced_switch_side = Side::Opp;
            new_node->opp_actions = legal_switch_actions(state, Side::Opp);
            new_node->opp_stats.resize(new_node->opp_actions.size());
            break;
          }
          case TurnResolution::kBothFainted: {
            // First of two chained kForcedSwitch nodes (SearchNode's own
            // doc comment) - my side's, arbitrarily; the opponent's own
            // forced-switch node becomes this node's child once expanded.
            new_node->kind = NodeKind::kForcedSwitch;
            new_node->forced_switch_side = Side::Me;
            new_node->my_actions = legal_switch_actions(state, Side::Me);
            new_node->my_stats.resize(new_node->my_actions.size());
            break;
          }
          case TurnResolution::kTerminal: {
            break;
          }
        }

        leaf_value = leaf_eval(state);
        if (result != TurnResolution::kTerminal) {
          cur->children[picks] = std::move(new_node);
        }
        break;

      } else {
        // kForcedSwitch is deliberately never routed through
        // resolve_turn(): that function requires a real, legal action
        // from both sides, and the non-acting side here only has
        // kNoAction to offer - is_switch_action() reads any value below
        // kNumSwitchActions as a switch, so passing kNoAction (-1)
        // through would misread it as "switch to slot -1." Apply the one
        // chosen switch directly instead.
        Side side = cur->forced_switch_side;
        const std::vector<ActionId>& actions = (side == Side::Me) ? cur->my_actions : cur->opp_actions;
        const std::vector<VisitStats>& stats = (side == Side::Me) ? cur->my_stats : cur->opp_stats;
        ActionId chosen = select_ucb1_action(actions, stats, total_visits(stats), kDefaultExplorationConstant);

        uint32_t picks =
            (side == Side::Me) ? pack_action_pair(chosen, kNoAction) : pack_action_pair(kNoAction, chosen);
        if (side == Side::Me) {
          path.push_back({cur, chosen, kNoAction});
        } else {
          path.push_back({cur, kNoAction, chosen});
        }

        if (side == Side::Me) {
          state.my_active_slot = chosen;
          apply_switch_in_hazards(state.my_team[chosen], state.my_hazards);
        } else {
          state.opp_active_slot = chosen;
          apply_switch_in_hazards(state.opp_team[chosen], state.opp_hazards);
        }

        auto it = cur->children.find(picks);
        if (it != cur->children.end()) {
          cur = it->second.get();
          continue;
        }

        // A second forced switch is still owed (the kBothFainted chain)
        // iff the OTHER side's active mon is still marked fainted here -
        // this branch never touches anything but `side`'s own slot, so
        // that can only be true if the original turn was kBothFainted.
        Side other = (side == Side::Me) ? Side::Opp : Side::Me;
        bool other_still_fainted = (other == Side::Me) ? state.my_team[state.my_active_slot].fainted
                                                         : state.opp_team[state.opp_active_slot].fainted;

        auto new_node = std::make_unique<SearchNode>();
        if (other_still_fainted) {
          new_node->kind = NodeKind::kForcedSwitch;
          new_node->forced_switch_side = other;
          if (other == Side::Me) {
            new_node->my_actions = legal_switch_actions(state, Side::Me);
            new_node->my_stats.resize(new_node->my_actions.size());
          } else {
            new_node->opp_actions = legal_switch_actions(state, Side::Opp);
            new_node->opp_stats.resize(new_node->opp_actions.size());
          }
        } else {
          new_node->kind = NodeKind::kDecision;
          new_node->my_actions = legal_actions(state, Side::Me);
          new_node->my_stats.resize(new_node->my_actions.size());
          new_node->opp_actions = legal_actions(state, Side::Opp);
          new_node->opp_stats.resize(new_node->opp_actions.size());
        }

        leaf_value = leaf_eval(state);
        cur->children[picks] = std::move(new_node);
        break;
      }
    }

    for (auto rit = path.rbegin(); rit != path.rend(); ++rit) {
      auto& [node, my_action, opp_action] = *rit;
      int my_idx = find_index(node->my_actions, my_action);
      if (my_idx != -1) {
        node->my_stats[my_idx].visits += 1;
        node->my_stats[my_idx].value_sum += leaf_value;
      }
      int opp_idx = find_index(node->opp_actions, opp_action);
      if (opp_idx != -1) {
        node->opp_stats[opp_idx].visits += 1;
        node->opp_stats[opp_idx].value_sum += -leaf_value;
      }
    }
  }

  SearchResult final_result;
  final_result.best_action = kNoAction;
  int best_visits = -1;
  for (int idx = 0; idx < (int)root_node.my_actions.size(); idx++) {
    int visits = root_node.my_stats[idx].visits;
    final_result.root_visit_distribution.push_back({root_node.my_actions[idx], visits});
    if (visits > best_visits) {
      best_visits = visits;
      final_result.best_action = root_node.my_actions[idx];
    }
  }
  return final_result;
}

}  // namespace be
