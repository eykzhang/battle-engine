#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "be/mcts.hpp"

using namespace be;

// ---------------------------------------------------------------------------
// select_ucb1_action - a synthetic bandit with a known-correct answer,
// independent of any BattleState/resolve_turn machinery (per the plan's own
// M6 "Done" criterion #1). These pin down the UCB1 arithmetic itself before
// it's ever exercised through a real tree walk.
// ---------------------------------------------------------------------------

TEST_CASE("select_ucb1_action: an untried action always wins, regardless of a visited action's average value",
          "[mcts]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/1, /*value_sum=*/1.0},  // action 0: perfect average (1.0), but only 1 visit
      {/*visits=*/0, /*value_sum=*/0.0},  // action 1: never tried
  };

  ActionId picked = select_ucb1_action(actions, stats, /*parent_visits=*/1, /*exploration_constant=*/1.4f);

  REQUIRE(picked == 1);
}

TEST_CASE("select_ucb1_action: with low exploration, the higher-average-value action wins", "[mcts]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/100, /*value_sum=*/60.0},  // action 0: avg 0.6
      {/*visits=*/100, /*value_sum=*/10.0},  // action 1: avg 0.1, same visit count
  };

  ActionId picked = select_ucb1_action(actions, stats, /*parent_visits=*/200, /*exploration_constant=*/0.1f);

  REQUIRE(picked == 0);
}

TEST_CASE("select_ucb1_action: with high exploration, a heavily under-visited action can win despite a lower average",
          "[mcts]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/100, /*value_sum=*/60.0},  // action 0: avg 0.6, well-explored
      {/*visits=*/4, /*value_sum=*/1.0},     // action 1: avg 0.25, barely explored
  };

  // At exploration_constant=0.1, action 0's higher average dominates (see
  // the low-exploration test above). At exploration_constant=5.0, action
  // 1's much larger sqrt(ln(parent_visits)/visits) exploration bonus (far
  // fewer visits) should flip the pick - this is the actual point of
  // UCB1's exploration term, not just an edge case.
  ActionId picked = select_ucb1_action(actions, stats, /*parent_visits=*/104, /*exploration_constant=*/5.0f);

  REQUIRE(picked == 1);
}

TEST_CASE("select_ucb1_action: picks the higher (less negative) score when every candidate is losing", "[mcts]") {
  // value_sum can legitimately be negative - evaluate() (the default
  // leaf_eval) isn't bounded to [0, inf), it goes negative whenever a side
  // is behind. An implementation that seeds its running best-score at 0.0f
  // rather than the first candidate's own score (or -infinity) would never
  // update away from action 0 here, regardless of which action's score is
  // actually higher - this pins that down concretely.
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/100, /*value_sum=*/-500.0f},  // action 0: avg -5.0
      {/*visits=*/100, /*value_sum=*/-100.0f},  // action 1: avg -1.0, clearly better
  };

  ActionId picked = select_ucb1_action(actions, stats, /*parent_visits=*/200, /*exploration_constant=*/0.1f);

  REQUIRE(picked == 1);
}

TEST_CASE("select_ucb1_action: an empty action list returns kNoAction", "[mcts]") {
  std::vector<ActionId> actions;
  std::vector<VisitStats> stats;

  ActionId picked = select_ucb1_action(actions, stats, /*parent_visits=*/0, /*exploration_constant=*/1.4f);

  REQUIRE(picked == kNoAction);
}

// ---------------------------------------------------------------------------
// pack_action_pair - trivial, but real: confirms the packing is injective
// over the range of values SearchNode's children map actually keys by
// (real ActionIds 0-9, plus the kNoAction sentinel for a kForcedSwitch
// node's unused slot).
// ---------------------------------------------------------------------------

TEST_CASE("pack_action_pair: distinct (my, opp) pairs never collide, including kNoAction", "[mcts]") {
  std::vector<ActionId> candidates = {0, 1, 5, 6, 9, kNoAction};
  std::vector<uint32_t> packed;
  for (ActionId my : candidates) {
    for (ActionId opp : candidates) {
      packed.push_back(pack_action_pair(my, opp));
    }
  }
  std::vector<uint32_t> sorted_packed = packed;
  std::sort(sorted_packed.begin(), sorted_packed.end());
  REQUIRE(std::adjacent_find(sorted_packed.begin(), sorted_packed.end()) == sorted_packed.end());
}

// ---------------------------------------------------------------------------
// search() - full tree-walk tests over real (but toy) BattleStates. Per the
// plan's M6 "Done" criteria #2-3: an obviously-lethal-move toy state (no
// poke-env integration required) and fixed-seed determinism.
//
// Criterion #4 (measured ms/turn at the chosen n_simulations, documented) is
// deliberately NOT a Catch2 assertion here - it's a real-world measurement
// to take and record (in CLAUDE.md, same as every other phase's laptop-
// feasibility numbers) once search() actually works and a simulation count
// has been chosen, not something a synthetic toy state can meaningfully
// assert on.
// ---------------------------------------------------------------------------

namespace {

PokemonSlot make_filler_bench_slot() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.hp_fraction = 1.0f;
  slot.level = 100;
  slot.base_stats = {100, 100, 100, 100, 100, 100};
  return slot;
}

// Sorts a visit distribution by ActionId so equality checks (and lookups by
// a specific action) don't depend on unordered_map iteration order.
std::vector<std::pair<ActionId, int>> sorted_by_action(std::vector<std::pair<ActionId, int>> dist) {
  std::sort(dist.begin(), dist.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  return dist;
}

int visits_for(const std::vector<std::pair<ActionId, int>>& dist, ActionId action) {
  for (const auto& [a, v] : dist) {
    if (a == action) return v;
  }
  return 0;
}

}  // namespace

TEST_CASE("search: visits concentrate on an obviously lethal move over a no-op status move", "[mcts]") {
  // My active: full HP, can pick between a guaranteed-lethal fixed-damage
  // hit (Seismic Toss - deals attacker.level damage, bypassing crit/roll/
  // STAB entirely, so its outcome is effectively deterministic here) and a
  // pure no-op (Protect - Tier 1 treats Status moves as a turn-cost-only
  // no-op, see forward_model.hpp's own module comment).
  BattleState state;

  PokemonSlot me;
  me.revealed = true;
  me.hp_fraction = 1.0f;
  me.level = 100;
  me.base_stats = {100, 100, 100, 100, 100, 100};
  me.spe_stat = 100;
  me.moves = {"seismictoss", "protect", "", ""};
  state.my_team[0] = me;
  state.my_active_slot = 0;

  // Opponent: one hit point above 0 - Seismic Toss's ~105 flat damage
  // guarantees a faint; Protect leaves it alive to act back.
  PokemonSlot opp;
  opp.revealed = true;
  opp.hp_fraction = 0.01f;
  opp.level = 100;
  opp.base_stats = {100, 100, 100, 100, 100, 100};
  opp.spe_stat = 100;
  opp.moves = {"tackle", "protect", "", ""};
  state.opp_team[0] = opp;
  state.opp_active_slot = 0;

  for (int i = 1; i < 6; ++i) {
    state.my_team[i] = make_filler_bench_slot();
    state.opp_team[i] = make_filler_bench_slot();
  }

  SearchResult result = search(state, default_eval, /*n_simulations=*/500, /*seed=*/1);

  auto dist = sorted_by_action(result.root_visit_distribution);
  int seismic_toss_visits = visits_for(dist, kMoveActionOffset + 0);
  int protect_visits = visits_for(dist, kMoveActionOffset + 1);

  REQUIRE(seismic_toss_visits > protect_visits);
  REQUIRE(result.best_action == kMoveActionOffset + 0);
}

TEST_CASE("search: same seed and inputs produce an identical root visit distribution (determinism)", "[mcts]") {
  auto make_state = []() {
    BattleState state;

    PokemonSlot me;
    me.revealed = true;
    me.hp_fraction = 0.5f;
    me.level = 100;
    me.base_stats = {100, 100, 100, 100, 100, 100};
    me.spe_stat = 100;
    me.moves = {"tackle", "quickattack", "", ""};
    state.my_team[0] = me;
    state.my_active_slot = 0;

    PokemonSlot opp;
    opp.revealed = true;
    opp.hp_fraction = 0.5f;
    opp.level = 100;
    opp.base_stats = {100, 100, 100, 100, 100, 100};
    opp.spe_stat = 100;
    opp.moves = {"tackle", "quickattack", "", ""};
    state.opp_team[0] = opp;
    state.opp_active_slot = 0;

    for (int i = 1; i < 6; ++i) {
      state.my_team[i] = make_filler_bench_slot();
      state.opp_team[i] = make_filler_bench_slot();
    }
    return state;
  };

  BattleState state_a = make_state();
  BattleState state_b = make_state();

  SearchResult result_a = search(state_a, default_eval, /*n_simulations=*/300, /*seed=*/42);
  SearchResult result_b = search(state_b, default_eval, /*n_simulations=*/300, /*seed=*/42);

  REQUIRE(result_a.best_action == result_b.best_action);
  REQUIRE(sorted_by_action(result_a.root_visit_distribution) == sorted_by_action(result_b.root_visit_distribution));
}
