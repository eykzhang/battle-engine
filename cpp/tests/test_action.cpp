#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "be/action.hpp"

using namespace be;

namespace {

bool contains(const std::vector<ActionId>& actions, ActionId a) {
  return std::find(actions.begin(), actions.end(), a) != actions.end();
}

PokemonSlot make_revealed_slot() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.fainted = false;
  return slot;
}

}  // namespace

TEST_CASE("legal_actions: switch targets exclude self, fainted, and unrevealed slots", "[action]") {
  BattleState state;
  state.my_team[0] = make_revealed_slot();  // active
  state.my_team[1] = make_revealed_slot();  // legal switch target
  state.my_team[2] = make_revealed_slot();
  state.my_team[2].fainted = true;  // illegal: fainted
  // my_team[3..5] default to unrevealed - illegal: unrevealed
  state.my_active_slot = 0;

  auto actions = legal_actions(state, Side::Me);

  REQUIRE(contains(actions, 1));
  REQUIRE_FALSE(contains(actions, 0));  // can't switch to the already-active slot
  REQUIRE_FALSE(contains(actions, 2));  // fainted
  REQUIRE_FALSE(contains(actions, 3));  // unrevealed
  REQUIRE_FALSE(contains(actions, 4));
  REQUIRE_FALSE(contains(actions, 5));
}

TEST_CASE("legal_actions: move actions are exactly the active mon's known (non-empty) move slots", "[action]") {
  BattleState state;
  PokemonSlot active = make_revealed_slot();
  active.moves = {"stealthrock", "", "uturn", ""};
  state.my_team[0] = active;
  state.my_active_slot = 0;

  auto actions = legal_actions(state, Side::Me);

  REQUIRE(contains(actions, kMoveActionOffset + 0));  // stealthrock
  REQUIRE_FALSE(contains(actions, kMoveActionOffset + 1));  // empty slot
  REQUIRE(contains(actions, kMoveActionOffset + 2));  // uturn
  REQUIRE_FALSE(contains(actions, kMoveActionOffset + 3));  // empty slot
}

TEST_CASE("legal_actions: no active Pokemon means no move actions, but switches still work", "[action]") {
  BattleState state;
  state.my_team[0] = make_revealed_slot();
  state.my_team[0].fainted = true;  // just fainted, forced-switch pending
  state.my_team[1] = make_revealed_slot();
  state.my_active_slot = -1;

  auto actions = legal_actions(state, Side::Me);

  REQUIRE(contains(actions, 1));
  for (ActionId a = kMoveActionOffset; a < kMoveActionOffset + kNumMoveActions; ++a) {
    REQUIRE_FALSE(contains(actions, a));
  }
}

TEST_CASE("legal_actions: Side::Opp switch targets are restricted to revealed slots only (Tier 1)", "[action]") {
  BattleState state;
  state.opp_team[0] = make_revealed_slot();  // active
  state.opp_team[1] = make_revealed_slot();  // revealed - legal
  // opp_team[2..5] unrevealed - illegal, even though a real opponent has
  // Pokemon there - this is Tier 1's named opponent-modeling limitation,
  // not a bug (see action.hpp's own doc comment on legal_actions()).
  state.opp_active_slot = 0;

  auto actions = legal_actions(state, Side::Opp);

  REQUIRE(contains(actions, 1));
  REQUIRE_FALSE(contains(actions, 2));
  REQUIRE_FALSE(contains(actions, 3));
  REQUIRE_FALSE(contains(actions, 4));
  REQUIRE_FALSE(contains(actions, 5));
}

TEST_CASE("legal_actions: Side::Opp move actions are restricted to already-revealed moves only (Tier 1)", "[action]") {
  BattleState state;
  PokemonSlot active = make_revealed_slot();
  active.moves = {"flamethrower", "", "", ""};  // only 1 of 4 moves seen so far
  state.opp_team[0] = active;
  state.opp_active_slot = 0;

  auto actions = legal_actions(state, Side::Opp);

  REQUIRE(contains(actions, kMoveActionOffset + 0));
  REQUIRE_FALSE(contains(actions, kMoveActionOffset + 1));
  REQUIRE_FALSE(contains(actions, kMoveActionOffset + 2));
  REQUIRE_FALSE(contains(actions, kMoveActionOffset + 3));
}

TEST_CASE("legal_actions: a side with nothing left to do returns an empty vector", "[action]") {
  BattleState state;  // every slot defaults to unrevealed, no active mon
  state.my_active_slot = -1;

  auto actions = legal_actions(state, Side::Me);

  REQUIRE(actions.empty());
}
