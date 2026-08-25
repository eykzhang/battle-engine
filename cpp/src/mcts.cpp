#include "be/mcts.hpp"

#include <algorithm>
#include <cmath>
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

// ---------------------------------------------------------------------------
// M6b: PUCT search - see mcts.hpp's own doc comments (SearchNode::my_priors/
// opp_priors, select_puct_action, puct_priors_from_actor_logits,
// search_puct) for the full design rationale. Everything below is the
// implementation those comments describe.
// ---------------------------------------------------------------------------

ActionId select_puct_action(const std::vector<ActionId>& actions, const std::vector<VisitStats>& stats,
                             const std::vector<float>& priors, int parent_visits, float c_puct) {
  if (actions.empty()) return kNoAction;
  float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));
  float best_score = -std::numeric_limits<float>::infinity();
  int best = 0;
  for (int i = 0; i < (int)actions.size(); i++) {
    float q = (stats[i].visits == 0) ? 0.0f : stats[i].value_sum / stats[i].visits;
    float score = q + c_puct * priors[i] * sqrt_parent / (1.0f + static_cast<float>(stats[i].visits));
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return actions[best];
}

std::vector<float> puct_priors_from_actor_logits(const BattleState& state, const std::vector<ActionId>& actions,
                                                   const std::vector<float>& actor_logits) {
  std::vector<float> gathered(actions.size());
  for (size_t i = 0; i < actions.size(); ++i) {
    int label = action_id_to_metamon_label(state, actions[i]);
    // A legal ActionId always has a real Metamon label (see
    // action_id_to_metamon_label's own doc comment - legal_actions()'s
    // switch-legality contract is exactly species_sorted_bench_slots()'s
    // own inclusion set, by construction) - a negative label here would be
    // a caller bug (an illegal ActionId slipped through), not handled
    // specially, same "callers own their inputs" convention as elsewhere.
    gathered[i] = actor_logits[static_cast<size_t>(label)];
  }
  if (gathered.empty()) return gathered;

  // Numerically-stable softmax over exactly this gathered subset -
  // mathematically identical to sb3_contrib's real masked-softmax
  // behavior, see this function's own doc comment in mcts.hpp for the
  // verified equivalence.
  float max_logit = *std::max_element(gathered.begin(), gathered.end());
  float sum = 0.0f;
  std::vector<float> probs(gathered.size());
  for (size_t i = 0; i < gathered.size(); ++i) {
    probs[i] = std::exp(gathered[i] - max_logit);
    sum += probs[i];
  }
  for (float& p : probs) p /= sum;
  return probs;
}

namespace {

// See search_puct()'s own doc comment (mcts.hpp) for the full "why" - a
// kDecision-shaped state only has TESTED actor/critic semantics when both
// sides have a real, alive active Pokemon. encode_native()'s own throw only
// covers active_slot == -1; a fainted-but-index-valid active (confirmed via
// forward_model.cpp: resolve_turn/apply_action never resets active_slot to
// -1 on a faint) does NOT throw but is still untested territory for a model
// PPO never trained on such an observation for, so this predicate checks
// both conditions explicitly rather than relying on encode_native()'s own
// (narrower) guard.
bool has_encodable_actives(const BattleState& state) {
  if (state.my_active_slot < 0 || state.opp_active_slot < 0) return false;
  if (state.my_team[static_cast<size_t>(state.my_active_slot)].fainted) return false;
  if (state.opp_team[static_cast<size_t>(state.opp_active_slot)].fainted) return false;
  return true;
}

// The two backed-up values for one leaf state - both use the "-v" sign
// convention for the opponent-table backup, for two INDEPENDENTLY justified
// reasons kept separate rather than merged into one unexplained branch:
// default_eval's antisymmetry is PROVEN (verified directly against
// eval.cpp's formula, see EvalFn's own doc comment above); the critic's is
// MEASURED (see search_puct()'s own doc comment in mcts.hpp for the actual
// mirrored-pair experiment and result) - a coincidence of measurement for
// THIS trained checkpoint, not a guaranteed property of PPO critics in
// general, so the two cases are not the same claim even though the formula
// they land on is identical.
struct LeafScore {
  float my_value = 0.0f;
  float opp_value = 0.0f;
};

LeafScore score_leaf_puct(const BattleState& state, const PolicyWeights& weights) {
  if (has_encodable_actives(state)) {
    float v = weights.critic.forward(encode_native(state))[0];
    return {v, -v};
  }
  float v = default_eval(state);
  return {v, -v};
}

// Result of populating a freshly-created kDecision node's actions/stats/
// priors - `encodable` mirrors has_encodable_actives(state) at the point
// this node was created; `critic_value` (meaningful only when encodable) is
// the critic's forward pass on the SAME encode_native(state) already
// computed for the "my"-side actor prior, avoiding a second, redundant
// encode_native() call for what would otherwise be a second forward-pass
// boundary immediately afterward (score_leaf_puct's own encode_native call
// is skipped in this path for exactly this reason - see search_puct()'s own
// use of this struct).
struct DecisionNodeResult {
  bool encodable = false;
  float critic_value = 0.0f;
};

// Populates `node`'s my_actions/my_stats/opp_actions/opp_stats (always, via
// legal_actions() - unchanged from search()'s own population) and, only
// when has_encodable_actives(state), my_priors/opp_priors (via the actor)
// plus the critic's leaf value for `state` - all bundled in one call site
// since both "kContinue produced a new kDecision node" and "a forced-switch
// chain resolved into a new kDecision node" need the exact same population
// logic (this function exists specifically so those two call sites in
// search_puct() don't duplicate it, a real divergence risk given the
// mirror()+two-actor-forward-passes involved).
DecisionNodeResult populate_decision_node(SearchNode& node, const BattleState& state, const PolicyWeights& weights) {
  node.my_actions = legal_actions(state, Side::Me);
  node.my_stats.resize(node.my_actions.size());
  node.opp_actions = legal_actions(state, Side::Opp);
  node.opp_stats.resize(node.opp_actions.size());

  if (!has_encodable_actives(state)) return {};

  std::vector<float> my_encoded = encode_native(state);
  std::vector<float> my_logits = weights.actor.forward(my_encoded);
  node.my_priors = puct_priors_from_actor_logits(state, node.my_actions, my_logits);
  float critic_value = weights.critic.forward(my_encoded)[0];

  BattleState mirrored = mirror(state);
  std::vector<float> opp_logits = weights.actor.forward(encode_native(mirrored));
  node.opp_priors = puct_priors_from_actor_logits(mirrored, node.opp_actions, opp_logits);

  return {true, critic_value};
}

}  // namespace

SearchResult search_puct(const BattleState& root, const PolicyWeights& weights, int n_simulations, uint64_t seed) {
  Rng rng(seed);
  SearchNode root_node;
  populate_decision_node(root_node, root, weights);

  for (int sim = 0; sim < n_simulations; sim++) {
    BattleState state = root;
    SearchNode* cur = &root_node;
    std::vector<std::tuple<SearchNode*, ActionId, ActionId>> path;
    float leaf_my_value = 0.0f;
    float leaf_opp_value = 0.0f;

    while (true) {
      if (cur->kind == NodeKind::kDecision) {
        ActionId my_pick = cur->my_priors.empty()
                                ? select_ucb1_action(cur->my_actions, cur->my_stats, total_visits(cur->my_stats),
                                                      kDefaultExplorationConstant)
                                : select_puct_action(cur->my_actions, cur->my_stats, cur->my_priors,
                                                      total_visits(cur->my_stats), kDefaultPuctConstant);
        ActionId opp_pick = cur->opp_priors.empty()
                                 ? select_ucb1_action(cur->opp_actions, cur->opp_stats, total_visits(cur->opp_stats),
                                                       kDefaultExplorationConstant)
                                 : select_puct_action(cur->opp_actions, cur->opp_stats, cur->opp_priors,
                                                       total_visits(cur->opp_stats), kDefaultPuctConstant);

        // Same empty-action-list handling as search() (see that function's
        // own EMPTY-ACTION-LIST NOTE) - a genuinely real, tested scenario,
        // not hypothetical.
        if (my_pick == kNoAction || opp_pick == kNoAction) {
          LeafScore score = score_leaf_puct(state, weights);
          leaf_my_value = score.my_value;
          leaf_opp_value = score.opp_value;
          break;
        }

        uint32_t picks = pack_action_pair(my_pick, opp_pick);
        path.push_back({cur, my_pick, opp_pick});

        TurnResolution result = resolve_turn(state, my_pick, opp_pick, rng);

        auto it = cur->children.find(picks);
        if (it != cur->children.end()) {
          cur = it->second.get();
          continue;
        }

        auto new_node = std::make_unique<SearchNode>();
        DecisionNodeResult decision_result;
        switch (result) {
          case TurnResolution::kContinue: {
            new_node->kind = NodeKind::kDecision;
            decision_result = populate_decision_node(*new_node, state, weights);
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

        // M6b DEPARTURE from search(): a newly-created kForcedSwitch node
        // does NOT stop this simulation - expansion continues past it (see
        // search_puct()'s own doc comment in mcts.hpp, point 2) until a
        // real kDecision state is reached for the critic's leaf value.
        bool created_forced_switch =
            (result != TurnResolution::kTerminal) && (new_node->kind == NodeKind::kForcedSwitch);
        if (result != TurnResolution::kTerminal) {
          cur->children[picks] = std::move(new_node);
        }
        if (created_forced_switch) {
          cur = cur->children[picks].get();
          continue;
        }

        if (result == TurnResolution::kContinue && decision_result.encodable) {
          leaf_my_value = decision_result.critic_value;
        } else {
          leaf_my_value = default_eval(state);
        }
        leaf_opp_value = -leaf_my_value;
        break;

      } else {
        // kForcedSwitch: plain UCB1-only selection, no prior (see
        // SearchNode's own my_priors/opp_priors doc comment) - unchanged
        // selection mechanism from search(), only the post-selection
        // control flow differs (continues past a freshly-created
        // kForcedSwitch child instead of stopping).
        Side side = cur->forced_switch_side;
        const std::vector<ActionId>& actions = (side == Side::Me) ? cur->my_actions : cur->opp_actions;
        const std::vector<VisitStats>& stats = (side == Side::Me) ? cur->my_stats : cur->opp_stats;
        ActionId chosen = select_ucb1_action(actions, stats, total_visits(stats), kDefaultExplorationConstant);

        if (chosen == kNoAction) {
          LeafScore score = score_leaf_puct(state, weights);
          leaf_my_value = score.my_value;
          leaf_opp_value = score.opp_value;
          break;
        }

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

        Side other = (side == Side::Me) ? Side::Opp : Side::Me;
        bool other_still_fainted = (other == Side::Me) ? state.my_team[state.my_active_slot].fainted
                                                         : state.opp_team[state.opp_active_slot].fainted;

        auto new_node = std::make_unique<SearchNode>();
        DecisionNodeResult decision_result;
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
          decision_result = populate_decision_node(*new_node, state, weights);
        }

        bool created_forced_switch = (new_node->kind == NodeKind::kForcedSwitch);
        cur->children[picks] = std::move(new_node);
        if (created_forced_switch) {
          cur = cur->children[picks].get();
          continue;
        }

        if (decision_result.encodable) {
          leaf_my_value = decision_result.critic_value;
        } else {
          leaf_my_value = default_eval(state);
        }
        leaf_opp_value = -leaf_my_value;
        break;
      }
    }

    for (auto rit = path.rbegin(); rit != path.rend(); ++rit) {
      auto& [node, my_action, opp_action] = *rit;
      int my_idx = find_index(node->my_actions, my_action);
      if (my_idx != -1) {
        node->my_stats[my_idx].visits += 1;
        node->my_stats[my_idx].value_sum += leaf_my_value;
      }
      int opp_idx = find_index(node->opp_actions, opp_action);
      if (opp_idx != -1) {
        node->opp_stats[opp_idx].visits += 1;
        node->opp_stats[opp_idx].value_sum += leaf_opp_value;
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

        // M7 integration finding, more severe than the kForcedSwitch one
        // below (this one is real from TURN 1 of essentially every real
        // battle, not a deep-tree edge case): either side's action list
        // can be genuinely empty at a plain kDecision node - most commonly
        // the OPPONENT, whose active has 0 revealed moves yet and (early
        // game) no revealed bench to switch to, but "my" side can hit this
        // too if a freshly-switched-in mon's moveset isn't known and no
        // switch target remains. resolve_turn() requires a REAL legal
        // action from both sides (kForcedSwitch's own doc comment already
        // establishes this - is_switch_action()'s "< kNumSwitchActions"
        // check reads kNoAction's -1 as "switch to slot -1", an
        // out-of-bounds be::BattleState team-array index a few frames
        // down inside apply_action/apply_switch_in_hazards - an ASan-
        // caught stack-use-after-scope/UB crash during M7 bring-up, not
        // hypothetical). Treat it as a terminal leaf for this simulation
        // instead - evaluate `state` as-is and stop descending, same
        // "caller owns kNoAction" convention select_ucb1_action's own doc
        // comment states, and the same shape as the kForcedSwitch fix
        // below. Don't push to `path`: neither side made a real choice
        // here, so there's no action to attribute credit to in cur's own
        // stats.
        if (my_pick == kNoAction || opp_pick == kNoAction) {
          leaf_value = leaf_eval(state);
          break;
        }

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

        // M7 integration finding (not hypothetical - caught by ASan against
        // a realistic early-battle state during M7 bring-up): `actions` can
        // be genuinely empty here. This is NOT "every real Pokemon on
        // side's team is fainted" (a real Showdown battle would just end
        // instead of prompting a forced switch with nothing to switch to)
        // - it's the Tier-1 revealed-only opponent-modeling limitation
        // (legal_actions()'s own doc comment) surfacing at a forced-switch
        // node: overwhelmingly this is the OPPONENT's own case early in a
        // real battle, where only their lead is revealed (my_team is
        // always fully revealed by team preview's end, so `side == Me`
        // hitting this is far rarer but not structurally impossible if
        // every other my_team slot happens to be fainted along this
        // simulated path too). Either way, there is no real ActionId to
        // choose - select_ucb1_action's own doc comment already names this
        // "the caller's problem." Treat it as a terminal leaf for this
        // simulation: evaluate `state` as-is (the fainted mon stays
        // inactive/fainted) and stop descending, same shape as the
        // kDecision branch's own kTerminal handling above. Do NOT push
        // this node's own (non-)decision onto `path` - there is no real
        // action to attribute credit to in cur's own stats, and
        // `chosen == kNoAction` would otherwise be used as an out-of-
        // bounds std::array index two branches below (a genuine
        // stack-use-after-scope/UB crash, not a style nit).
        if (chosen == kNoAction) {
          leaf_value = leaf_eval(state);
          break;
        }

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
