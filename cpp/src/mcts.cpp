#include "be/mcts.hpp"

#include <cstdint>

#include "be/eval.hpp"
#include "be/forward_model.hpp"

namespace be {

uint32_t pack_action_pair(ActionId my_action, ActionId opp_action) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(my_action)) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(opp_action));
}

float default_eval(const BattleState& state) { return evaluate(state); }

ActionId select_ucb1_action(const std::vector<ActionId>& actions, const std::vector<VisitStats>& stats,
                             int parent_visits, float exploration_constant) {
  // TODO(you): implement per mcts.hpp's doc comment on
  // select_ucb1_action(). Suggested shape:
  //   1. actions.empty() -> return kNoAction immediately.
  //   2. Walk actions/stats together (index-aligned, same length per the
  //      header's own contract). For each i:
  //      - stats[i].visits == 0 -> this action wins immediately (untried
  //        actions always beat any visited one) - you can early-return
  //        actions[i] the first time you see this, or track it as
  //        "score = +infinity" (std::numeric_limits<float>::infinity())
  //        in the same running-argmax loop as the visited case, either
  //        is fine as long as an untried action always wins ties against
  //        a visited one.
  //      - otherwise: score = stats[i].value_sum / stats[i].visits
  //                          + exploration_constant * std::sqrt(std::log(parent_visits) / stats[i].visits)
  //   3. Track and return the actions[i] with the highest score seen.
  //
  // cpp/tests/test_mcts.cpp has a synthetic-bandit test with a
  // known-correct answer (one action with real accumulated value but many
  // visits, one barely-visited action - correct behavior depends on
  // parent_visits/exploration_constant, worked out concretely in the test
  // itself) plus an explicit "untried action always wins" case.
  (void)actions;
  (void)stats;
  (void)parent_visits;
  (void)exploration_constant;
  return kNoAction;
}

SearchResult search(const BattleState& root, const EvalFn& leaf_eval, int n_simulations, uint64_t seed) {
  // TODO(you): implement per mcts.hpp's doc comment on search() (and this
  // whole header's top-of-file note on the open-loop design). Suggested
  // shape, one simulation at a time, n_simulations times:
  //   1. SELECTION: starting at the root SearchNode (lazily created on the
  //      first simulation - see step 2 for how a fresh node gets its
  //      actions/stats populated), walk down while every node visited so
  //      far has ALL its actions already tried at least once (no
  //      untried-action case left for select_ucb1_action to hand back).
  //      At a kDecision node: call select_ucb1_action independently for
  //      my_actions/my_stats and opp_actions/opp_stats (that's the
  //      "decoupled" in DUCT - two independent argmaxes, not one joint
  //      one), pack_action_pair the two results, look up (or note as
  //      needing expansion) the matching child. At a kForcedSwitch node:
  //      only one side's table exists - select from it, key the child by
  //      pack_action_pair(chosen, kNoAction).
  //      Track the full path (nodes + chosen action-index-pairs) as you
  //      go - you'll need it for backup in step 4.
  //   2. EXPANSION: once you reach an action pair that has no child yet
  //      (or the very first simulation, where even the root doesn't
  //      exist): actually call resolve_turn() on a FRESH state resampled
  //      from `root` by replaying the whole accumulated path so far
  //      (open-loop - see this header's top-of-file note; there is no
  //      shortcut that avoids replaying from root every simulation).
  //      Inspect the returned TurnResolution to decide the new child
  //      node's shape:
  //        - kContinue: a new kDecision node, my_actions/opp_actions from
  //          legal_actions() for each side on the resulting state.
  //        - kMyFainted / kOppFainted: a new kForcedSwitch node for
  //          whichever side fainted, my_actions restricted to that side's
  //          legal SWITCH actions only (legal_actions() returns both
  //          switch and move actions - filter to action < kNumSwitchActions).
  //        - kBothFainted: see this header's own SearchNode doc comment
  //          for the suggested two-CHAINED-kForcedSwitch-nodes resolution
  //          (order between the two doesn't affect the resulting state).
  //        - kTerminal: a leaf - no child node at all, this is where a
  //          simulation ends this round (see step 3).
  //   3. LEAF EVALUATION: once you hit a genuinely new node (just
  //      expanded) or a kTerminal TurnResolution, call leaf_eval() on the
  //      resulting state to get this simulation's raw value - CAREFUL of
  //      the sign convention documented on EvalFn itself (default_eval is
  //      -v under perspective swap, NOT 1-v - that doc comment explains
  //      why in detail, it's a real correction to the plan's own text).
  //   4. BACKUP: walk the path from step 1 back up to the root. At each
  //      node, increment the visited action's `visits` and add the
  //      value (from THAT node's own side's POV - flip sign each level
  //      per the EvalFn doc comment's convention, since each step up the
  //      tree flips whose "my side" the score should read from) to
  //      `value_sum`, for whichever side(s) actually chose at that node
  //      (both, at a kDecision node - DUCT backs up both sides'
  //      independent tables from the same simulation; one, at a
  //      kForcedSwitch node).
  // Finally: build a SearchResult from the root's own my_actions/my_stats
  // (root_visit_distribution, per this header's own doc comment on which
  // side's table that field reports) and best_action = whichever
  // my_actions[i] has the highest final visits[i] (NOT re-run through
  // select_ucb1_action - argmax by visits is the standard "final answer"
  // rule, UCB1's exploration term is only for guiding search itself).
  //
  // cpp/tests/test_mcts.cpp has: a toy lethal-move BattleState where visits
  // should concentrate on the winning move (no poke-env integration
  // required), and a fixed-seed determinism test (same seed + inputs ->
  // identical root_visit_distribution). Measuring and documenting real
  // ms/turn at whatever n_simulations you settle on remains a manual step
  // for M7, not something the unit tests themselves assert - see the test
  // file's own note.
  (void)root;
  (void)leaf_eval;
  (void)n_simulations;
  (void)seed;
  return SearchResult{kNoAction, {}};
}

}  // namespace be
