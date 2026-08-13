#include <catch2/catch_test_macros.hpp>

#include "be/battle_state.hpp"

using namespace be;

namespace {

PokemonSlot make_valid_slot() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.fainted = false;
  slot.hp_fraction = 0.75f;
  slot.boost_spe = 2;
  return slot;
}

BattleState make_valid_state() {
  BattleState state;
  state.my_team[0] = make_valid_slot();
  state.opp_team[0] = make_valid_slot();
  state.my_active_slot = 0;
  state.opp_active_slot = 0;
  return state;
}

}  // namespace

TEST_CASE("is_valid accepts a well-formed state", "[battle_state]") {
  REQUIRE(is_valid(make_valid_state()));
}

TEST_CASE("is_valid accepts no active Pokemon (-1) on either side", "[battle_state]") {
  BattleState state = make_valid_state();
  state.my_active_slot = -1;
  state.opp_active_slot = -1;
  REQUIRE(is_valid(state));
}

TEST_CASE("is_valid rejects hp_fraction outside [0, 1]", "[battle_state]") {
  BattleState state = make_valid_state();
  state.my_team[0].hp_fraction = 1.5f;
  REQUIRE_FALSE(is_valid(state));

  state.my_team[0].hp_fraction = -0.1f;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid rejects boost_spe outside [-6, 6]", "[battle_state]") {
  BattleState state = make_valid_state();
  state.my_team[0].boost_spe = 7;
  REQUIRE_FALSE(is_valid(state));

  state.my_team[0].boost_spe = -7;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid rejects a fainted slot with nonzero hp_fraction", "[battle_state]") {
  BattleState state = make_valid_state();
  state.my_team[0].fainted = true;
  state.my_team[0].hp_fraction = 0.3f;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid rejects an active_slot pointing at an unrevealed slot", "[battle_state]") {
  BattleState state = make_valid_state();
  state.opp_team[1] = PokemonSlot{};  // default: revealed == false
  state.opp_active_slot = 1;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid rejects an active_slot pointing at a fainted slot", "[battle_state]") {
  BattleState state = make_valid_state();
  state.opp_team[0].fainted = true;
  state.opp_team[0].hp_fraction = 0.0f;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid rejects an out-of-range active_slot index", "[battle_state]") {
  BattleState state = make_valid_state();
  state.my_active_slot = 6;
  REQUIRE_FALSE(is_valid(state));
}

TEST_CASE("is_valid ignores unrevealed slots' garbage field values", "[battle_state]") {
  // An unrevealed slot is at PokemonSlot's default values - shouldn't fail
  // validation just for being unrevealed, even though a default-constructed
  // slot's hp_fraction (1.0) and boost_spe (0) happen to already be valid.
  // This case exists so is_valid's implementation can't accidentally
  // require every array slot (including never-revealed ones) to pass the
  // same checks as a revealed one.
  BattleState state = make_valid_state();
  state.opp_team[3].revealed = false;
  REQUIRE(is_valid(state));
}
