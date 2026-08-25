#include <algorithm>
#include <cmath>
#include <numeric>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "be/mcts.hpp"

using namespace be;
using Catch::Approx;

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

TEST_CASE(
    "search: a forced-switch node with no revealed replacement (unrevealed bench, the realistic "
    "early-battle case) is treated as a terminal leaf, not a crash",
    "[mcts]") {
  // Regression for an M7 integration finding, not a synthetic worry: every
  // real early-battle state built by battle_state_from_poke_env has only
  // the OPPONENT's lead revealed (my_team is always fully revealed by team
  // preview's end, but opp_team grows one entry at a time as poke-env
  // reveals it) - deliberately reproduced here with ONLY state.my_team[0]/
  // state.opp_team[0] populated (all other slots left default/unrevealed),
  // unlike every other search() test above, which always fills all 6
  // slots via make_filler_bench_slot(). Before the fix, a kForcedSwitch
  // node whose legal_switch_actions() comes back empty (found via
  // select_ucb1_action returning kNoAction) got that -1 used as a raw
  // std::array index a few lines later once the node was revisited on a
  // later simulation - an ASan-caught stack-use-after-scope/UB crash, not
  // a hypothetical. Seismic Toss's near-guaranteed 1-hit KO on the
  // 0.01-HP-fraction opponent (see the lethal-move test above for why its
  // damage is effectively deterministic) drives most simulations straight
  // into exactly that forced-switch node, so this reproduces reliably
  // rather than depending on rare UCB1 exploration.
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

  PokemonSlot opp;
  opp.revealed = true;
  opp.hp_fraction = 0.01f;
  opp.level = 100;
  opp.base_stats = {100, 100, 100, 100, 100, 100};
  opp.spe_stat = 100;
  opp.moves = {"tackle", "protect", "", ""};
  state.opp_team[0] = opp;
  state.opp_active_slot = 0;
  // my_team[1..5] and opp_team[1..5] left default (unrevealed=false, the
  // struct's own default) - the whole point of this fixture.

  SearchResult result = search(state, default_eval, /*n_simulations=*/500, /*seed=*/7);

  // The root itself always has a real action (my active starts alive) -
  // confirms search() completed all 500 simulations rather than bailing
  // out early, and that the fix's "treat as a terminal leaf" path doesn't
  // corrupt the root's own selection.
  REQUIRE(result.best_action != kNoAction);
  int total_visits = 0;
  for (const auto& [action, visits] : result.root_visit_distribution) total_visits += visits;
  REQUIRE(total_visits == 500);
}

TEST_CASE(
    "search: a plain kDecision node where one side has zero legal actions (no known moves, no "
    "switch target) is treated as a terminal leaf, not a crash",
    "[mcts]") {
  // Regression for the OTHER M7 integration finding (more severe than the
  // forced-switch one above: this one is real from turn 1 of essentially
  // every real battle, not a deep-tree edge case). Before the fix,
  // select_ucb1_action's documented kNoAction return for an empty action
  // list got passed straight into resolve_turn(): is_switch_action()'s
  // "< kNumSwitchActions" check misreads kNoAction's -1 as "switch to slot
  // -1", an out-of-bounds be::BattleState team-array index a few frames
  // down inside apply_action/apply_switch_in_hazards - ASan-caught, not
  // hypothetical. Reproduced here at the ROOT itself: the opponent's
  // active has zero known moves (moves = {"", "", "", ""}, the real state
  // of a freshly-revealed lead before it's ever acted) and no revealed
  // bench to switch to - exactly turn 1 of a real battle from this
  // engine's own information state.
  BattleState state;

  PokemonSlot me;
  me.revealed = true;
  me.hp_fraction = 1.0f;
  me.level = 100;
  me.base_stats = {100, 100, 100, 100, 100, 100};
  me.spe_stat = 100;
  me.moves = {"tackle", "protect", "", ""};
  state.my_team[0] = me;
  state.my_active_slot = 0;
  for (int i = 1; i < 6; ++i) state.my_team[i] = make_filler_bench_slot();

  PokemonSlot opp;
  opp.revealed = true;
  opp.hp_fraction = 1.0f;
  opp.level = 100;
  opp.base_stats = {100, 100, 100, 100, 100, 100};
  opp.spe_stat = 100;
  opp.moves = {"", "", "", ""};  // nothing revealed yet - turn 1's real state
  state.opp_team[0] = opp;
  state.opp_active_slot = 0;
  // opp_team[1..5] deliberately left default/unrevealed - no switch target
  // either, so opp_actions is genuinely empty at the root.

  SearchResult result = search(state, default_eval, /*n_simulations=*/300, /*seed=*/11);

  // Since the empty side is at the ROOT itself, every single simulation
  // bails out at the very first node, before any pick is ever pushed onto
  // `path` - so root_node's own my_stats never accumulate a recorded
  // visit (nothing downstream of the root ever runs). That's correct,
  // expected behavior for THIS degenerate "empty side is the root itself"
  // shape specifically (not a general property of the fix - a node
  // reached one or more levels deep would still have its ancestors'
  // visits recorded normally, since those picks are pushed to `path`
  // before the empty node is ever reached). best_action still resolves to
  // a real action (the first candidate, via search()'s own "first action
  // beats an uninitialized best_visits=-1" tie-break) rather than
  // kNoAction - the meaningful assertion here is that this completes at
  // all (300 simulations, no ASan abort), not a specific visit count.
  REQUIRE(result.best_action != kNoAction);
}

// ---------------------------------------------------------------------------
// M6b: select_puct_action - synthetic bandit tests (DW-5.1). PUCT differs
// from UCB1's cold-start behavior deliberately (see select_puct_action's
// own doc comment in mcts.hpp): an unvisited action does NOT automatically
// win - the prior biases early exploration instead. parent_visits=0
// (literally the very first-ever call at a brand-new node) makes every
// exploration term 0 regardless of prior - a real, accepted PUCT
// characteristic (the first pick at a node is arbitrary, same as most PUCT
// implementations), not something these tests special-case around: they
// exercise parent_visits>=1, the state every SUBSEQUENT call at a node is
// actually in.
// ---------------------------------------------------------------------------

TEST_CASE("select_puct_action: among two unvisited actions, the higher-prior one wins", "[mcts][puct]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {{0, 0.0f}, {0, 0.0f}};
  std::vector<float> priors = {0.1f, 0.9f};

  ActionId picked = select_puct_action(actions, stats, priors, /*parent_visits=*/1, /*c_puct=*/1.4f);

  REQUIRE(picked == 1);
}

TEST_CASE("select_puct_action: a strong empirical average can outweigh a weaker prior once BOTH actions are "
          "similarly well-explored",
          "[mcts][puct]") {
  // Both actions have equal visits (so the exploration term's 1/(1+visits)
  // denominator is the same for both, unlike a lightly-visited action -
  // where a large sqrt(parent_visits)/(1+visits) bonus can dominate
  // regardless of prior, exactly the OTHER test below's own point) - with
  // visit counts matched, Q's difference is what decides it.
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/50, /*value_sum=*/45.0f},  // action 0: avg 0.9, low prior
      {/*visits=*/50, /*value_sum=*/0.0f},   // action 1: avg 0.0, high prior
  };
  std::vector<float> priors = {0.1f, 0.9f};

  ActionId picked = select_puct_action(actions, stats, priors, /*parent_visits=*/100, /*c_puct=*/0.5f);

  REQUIRE(picked == 0);
}

TEST_CASE("select_puct_action: a high enough c_puct lets a high-prior, unvisited action beat a well-explored one",
          "[mcts][puct]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/100, /*value_sum=*/60.0f},  // action 0: avg 0.6
      {/*visits=*/0, /*value_sum=*/0.0f},     // action 1: unvisited, high prior
  };
  std::vector<float> priors = {0.1f, 0.9f};

  ActionId picked = select_puct_action(actions, stats, priors, /*parent_visits=*/100, /*c_puct=*/10.0f);

  REQUIRE(picked == 1);
}

TEST_CASE("select_puct_action: an empty action list returns kNoAction", "[mcts][puct]") {
  std::vector<ActionId> actions;
  std::vector<VisitStats> stats;
  std::vector<float> priors;

  ActionId picked = select_puct_action(actions, stats, priors, /*parent_visits=*/0, /*c_puct=*/1.4f);

  REQUIRE(picked == kNoAction);
}

// ---------------------------------------------------------------------------
// M6b: puct_priors_from_actor_logits (DW-5.1) - gather-then-softmax over
// legal ActionIds only, dropping any mass on Metamon labels with no
// ActionId counterpart (tera, 9-12).
// ---------------------------------------------------------------------------

TEST_CASE("puct_priors_from_actor_logits: matches a hand-computed softmax over exactly the legal move labels",
          "[mcts][puct]") {
  BattleState state;
  PokemonSlot active;
  active.revealed = true;
  active.species = "Zapdos";
  active.moves = {"thunderbolt", "drillpeck", "roost", "protect"};
  state.my_team[0] = active;
  state.my_active_slot = 0;

  std::vector<ActionId> actions = {kMoveActionOffset + 0, kMoveActionOffset + 2};  // Metamon labels 0 and 2
  std::vector<float> logits(13, 0.0f);
  logits[0] = 2.0f;
  logits[2] = 1.0f;
  logits[9] = 100.0f;  // a tera label with huge mass - must be dropped, not just diluted

  std::vector<float> priors = puct_priors_from_actor_logits(state, actions, logits);

  REQUIRE(priors.size() == 2);
  float expected_0 = std::exp(2.0f - 2.0f) / (std::exp(2.0f - 2.0f) + std::exp(1.0f - 2.0f));
  float expected_2 = std::exp(1.0f - 2.0f) / (std::exp(2.0f - 2.0f) + std::exp(1.0f - 2.0f));
  REQUIRE(priors[0] == Approx(expected_0));
  REQUIRE(priors[1] == Approx(expected_2));
  REQUIRE(priors[0] + priors[1] == Approx(1.0f));
}

TEST_CASE("puct_priors_from_actor_logits: tera-label mass is dropped, not folded into the remaining distribution",
          "[mcts][puct]") {
  BattleState state;
  PokemonSlot active;
  active.revealed = true;
  active.species = "Zapdos";
  active.moves = {"thunderbolt", "", "", ""};
  state.my_team[0] = active;
  state.my_active_slot = 0;

  std::vector<ActionId> actions = {kMoveActionOffset + 0};  // only move slot 0 legal
  std::vector<float> logits(13, 0.0f);
  logits[0] = -5.0f;    // even a very unfavorable legal logit...
  logits[9] = 1000.0f;  // ...still wins 100% of the mass once labels 9-12 are dropped

  std::vector<float> priors = puct_priors_from_actor_logits(state, actions, logits);

  REQUIRE(priors.size() == 1);
  REQUIRE(priors[0] == Approx(1.0f));
}

// ---------------------------------------------------------------------------
// M6b: search_puct() - DW-5.2 (determinism), DW-5.5 (kForcedSwitch
// continuation), and DW-5.4 (critic sign-backup convention, measured).
// All load the real data/cpp_weights/ppo.bin via BE_TEST_PPO_WEIGHTS_PATH
// (same compile-time-path convention as test_mlp.cpp's own DW-3.3
// benchmark) - these exercise the actor+critic for real, unlike search()'s
// toy fixtures above (which only ever need default_eval).
// ---------------------------------------------------------------------------

namespace {

PokemonSlot make_rich_puct_slot(const std::string& species, Type t1, Type t2 = Type::None) {
  PokemonSlot slot;
  slot.revealed = true;
  slot.hp_fraction = 1.0f;
  slot.level = 100;
  slot.species = species;
  slot.type1 = t1;
  slot.type2 = t2;
  slot.base_stats = {100, 100, 100, 100, 100, 100};
  slot.spe_stat = 100;
  slot.moves = {"tackle", "quickattack", "", ""};
  return slot;
}

// A full, encode_native()-safe 6v6 state (species/types set on every slot,
// distinct species so the bench sort is well-defined).
BattleState make_rich_search_puct_state() {
  BattleState state;
  const char* my_species[6] = {"Zapdos", "Bulbasaur", "Charmander", "Squirtle", "Gengar", "Pikachu"};
  const char* opp_species[6] = {"Moltres", "Ivysaur", "Charmeleon", "Wartortle", "Haunter", "Raichu"};
  for (int i = 0; i < 6; ++i) {
    state.my_team[i] = make_rich_puct_slot(my_species[i], Type::Normal);
    state.opp_team[i] = make_rich_puct_slot(opp_species[i], Type::Normal);
  }
  state.my_active_slot = 0;
  state.opp_active_slot = 0;
  return state;
}

}  // namespace

TEST_CASE("search_puct: same seed and inputs produce an identical root visit distribution (determinism)",
          "[mcts][puct]") {
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);
  BattleState state_a = make_rich_search_puct_state();
  BattleState state_b = make_rich_search_puct_state();

  SearchResult result_a = search_puct(state_a, weights, /*n_simulations=*/40, /*seed=*/42);
  SearchResult result_b = search_puct(state_b, weights, /*n_simulations=*/40, /*seed=*/42);

  REQUIRE(result_a.best_action == result_b.best_action);
  REQUIRE(sorted_by_action(result_a.root_visit_distribution) == sorted_by_action(result_b.root_visit_distribution));
}

TEST_CASE("search_puct: a forced-switch node with no revealed replacement falls back to default_eval as a "
          "terminal leaf, not a crash",
          "[mcts][puct]") {
  // Same shape as search()'s own identically-motivated regression test
  // above, through search_puct() instead - confirms the "chosen ==
  // kNoAction" fallback (mcts.hpp's search_puct() doc comment, point 2's
  // named exception) doesn't crash and doesn't attempt to encode a
  // fainted-active state.
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);

  BattleState state;
  PokemonSlot me = make_rich_puct_slot("Zapdos", Type::Electric, Type::Flying);
  me.moves = {"seismictoss", "protect", "", ""};
  state.my_team[0] = me;
  state.my_active_slot = 0;

  PokemonSlot opp = make_rich_puct_slot("Moltres", Type::Fire, Type::Flying);
  opp.hp_fraction = 0.01f;
  opp.moves = {"tackle", "protect", "", ""};
  state.opp_team[0] = opp;
  state.opp_active_slot = 0;
  // my_team[1..5]/opp_team[1..5] left default/unrevealed - no replacement
  // available on either side, the whole point of this fixture.

  SearchResult result = search_puct(state, weights, /*n_simulations=*/40, /*seed=*/7);

  REQUIRE(result.best_action != kNoAction);
  int total_visits = 0;
  for (const auto& [action, visits] : result.root_visit_distribution) total_visits += visits;
  REQUIRE(total_visits == 40);
}

TEST_CASE("search_puct: continues expansion past a forced-switch node with a real replacement, reaching a real "
          "kDecision state for the critic's leaf value (no crash, no invented encoding)",
          "[mcts][puct]") {
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);

  BattleState state = make_rich_search_puct_state();
  state.my_team[0].moves = {"seismictoss", "protect", "", ""};
  state.opp_team[0].hp_fraction = 0.01f;
  state.opp_team[0].moves = {"tackle", "protect", "", ""};
  // opp_team[1] (Ivysaur) stays a real, revealed replacement - unlike the
  // no-replacement fixture above, this exercises the "apply the switch and
  // keep going" path (SearchNode/search_puct's own doc comments) rather
  // than the "nothing legal, fall back to default_eval" one.

  SearchResult result = search_puct(state, weights, /*n_simulations=*/60, /*seed=*/11);

  int total_visits = 0;
  for (const auto& [action, visits] : result.root_visit_distribution) total_visits += visits;
  REQUIRE(total_visits == 60);
  REQUIRE(result.best_action != kNoAction);
}

TEST_CASE("search_puct: a root with no active Pokemon (team-preview / a forced-switch response) doesn't crash "
          "and falls back to UCB1-only selection at the root",
          "[mcts][puct]") {
  // Real scenario, not synthetic: poke-env's forced-switch request (an
  // active just fainted) presents battle.active_pokemon is None,
  // translating to my_active_slot == -1 (Phase 1's MctsPlayer already
  // tests this same shape at the plain UCB1 level). encode_native() would
  // NOT throw here either way for the OPPONENT's still-valid active, but
  // populate_decision_node()'s has_encodable_actives() gate gets tripped by
  // MY side's -1 - the whole node's priors must stay empty (not just my
  // side's), since encode_native() itself throws on ANY -1 active,
  // regardless of side - see mcts.hpp's own search_puct() doc comment,
  // point 1.
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);

  BattleState state = make_rich_search_puct_state();
  state.my_team[0].fainted = true;
  state.my_team[0].hp_fraction = 0.0f;
  state.my_active_slot = -1;  // forced-switch response - no active chosen yet

  SearchResult result = search_puct(state, weights, /*n_simulations=*/40, /*seed=*/3);

  REQUIRE(result.best_action != kNoAction);
  int total_visits = 0;
  for (const auto& [action, visits] : result.root_visit_distribution) total_visits += visits;
  REQUIRE(total_visits == 40);
}

// ---------------------------------------------------------------------------
// DW-5.4: the critic's sign-backup convention, measured empirically against
// the real trained checkpoint on synthetic mirror(state)/state pairs - NOT
// assumed. See search_puct()'s own doc comment in mcts.hpp for how this
// measurement determined the constant search_puct() actually implements.
// ---------------------------------------------------------------------------

TEST_CASE("DW-5.4: critic sign-backup convention measured on synthetic mirrored-state pairs", "[mcts][puct]") {
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);

  const char* species_pool[10] = {"Zapdos",  "Bulbasaur", "Charmander", "Squirtle",  "Gengar",
                                   "Pikachu", "Moltres",   "Snorlax",    "Dragonite", "Alakazam"};
  Type type_pool[6] = {Type::Electric, Type::Grass, Type::Fire, Type::Water, Type::Ghost, Type::Psychic};

  std::vector<float> sums;
  sums.reserve(30);
  for (int i = 0; i < 30; ++i) {
    BattleState state = make_rich_search_puct_state();
    // Vary HP/status/boosts/species/types across pairs so the measurement
    // isn't just re-checking one single fixed state.
    float hp = 0.1f + 0.03f * static_cast<float>(i);
    state.my_team[0].hp_fraction = std::min(hp, 1.0f);
    state.opp_team[0].hp_fraction = std::min(1.1f - hp, 1.0f);
    state.my_team[0].boost_spe = static_cast<int8_t>((i % 5) - 2);
    state.opp_team[0].boost_atk = static_cast<int8_t>((i % 3) - 1);
    state.my_team[0].species = species_pool[i % 10];
    state.opp_team[0].species = species_pool[(i + 3) % 10];
    state.my_team[0].type1 = type_pool[i % 6];
    state.opp_team[0].type1 = type_pool[(i + 2) % 6];
    if (i % 4 == 0) state.my_team[0].status = Status::Brn;
    if (i % 5 == 0) state.opp_team[0].status = Status::Par;

    float v_s = weights.critic.forward(encode_native(state))[0];
    float v_mirror = weights.critic.forward(encode_native(mirror(state)))[0];
    sums.push_back(v_s + v_mirror);
  }

  float mean = std::accumulate(sums.begin(), sums.end(), 0.0f) / static_cast<float>(sums.size());
  float sq_dev = 0.0f;
  for (float x : sums) sq_dev += (x - mean) * (x - mean);
  float stddev = std::sqrt(sq_dev / static_cast<float>(sums.size()));

  INFO("mean(v(s)+v(mirror(s))) = " << mean << ", stddev = " << stddev << " (over " << sums.size()
                                     << " synthetic pairs)");
  INFO("distance to 0 (-v convention, matches default_eval's own proven antisymmetry): " << std::abs(mean));
  INFO("distance to 1 (1-v convention, correct for a bounded [0,1] win-probability evaluator - the critic is "
       "NOT one): " << std::abs(mean - 1.0f));

  // Documents the measured result rather than gating the build on a
  // specific numeric outcome - see mcts.hpp's search_puct() doc comment
  // for the actual measured value and which convention search_puct()
  // implements as a result (this build measured "-v" as the clearly
  // closer convention, matching default_eval's own proven antisymmetry -
  // recorded there, not duplicated here to avoid this comment going stale
  // against a future retrained checkpoint).
  SUCCEED("see INFO output above for the measured mean/stddev this build recorded");
}

// ---------------------------------------------------------------------------
// DW-5.3: measured ms/turn for search_puct at a candidate n_simulations,
// against the real ppo.bin - hidden by default ([!benchmark] tag, same
// convention as test_mlp.cpp's DW-3.3 benchmark), run explicitly via
// `./be_tests "[puct][!benchmark]"` post-build.
// ---------------------------------------------------------------------------

TEST_CASE("search_puct: microbenchmark ms/turn at n_simulations=200 against the real ppo.bin",
          "[mcts][puct][!benchmark]") {
  PolicyWeights weights = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);
  BattleState state = make_rich_search_puct_state();

  BENCHMARK("search_puct, n_simulations=200") {
    return search_puct(state, weights, /*n_simulations=*/200, /*seed=*/1).best_action;
  };
}
