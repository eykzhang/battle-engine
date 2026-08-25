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

// ---------------------------------------------------------------------------
// M6b: Metamon-mapping functions (DW-5.1) - known team -> known mapping,
// species-sorted-bench identity ported from action_space.py's
// _switch_action_to_poke_env/_poke_env_switch_to_metamon (see this file's
// own module comment at the top of action.hpp for the full contract).
// ---------------------------------------------------------------------------

namespace {

PokemonSlot make_slot(const std::string& species, bool fainted = false) {
  PokemonSlot slot;
  slot.revealed = true;
  slot.fainted = fainted;
  slot.hp_fraction = fainted ? 0.0f : 1.0f;
  slot.species = species;
  return slot;
}

// A full 6-member team, team-preview order deliberately NOT
// alphabetical, so a passing test can't be an accident of the two
// orderings happening to coincide. Active is slot 0 ("Zapdos" - last
// alphabetically), so every other slot is a bench candidate.
// Species-sorted bench order: Bulbasaur(1), Charmander(4), Gengar(3),
// Pikachu(2), Squirtle(5) -> Metamon switch labels 4,5,6,7,8 respectively.
BattleState make_full_team_state() {
  BattleState state;
  state.my_team[0] = make_slot("Zapdos");
  state.my_team[1] = make_slot("Bulbasaur");
  state.my_team[2] = make_slot("Pikachu");
  state.my_team[3] = make_slot("Gengar");
  state.my_team[4] = make_slot("Charmander");
  state.my_team[5] = make_slot("Squirtle");
  state.my_team[0].moves = {"thunderbolt", "drillpeck", "", ""};
  state.my_active_slot = 0;
  return state;
}

}  // namespace

TEST_CASE("metamon_switch_label_to_action_id: known team maps each switch label to its species-sorted slot",
          "[action]") {
  BattleState state = make_full_team_state();

  REQUIRE(metamon_switch_label_to_action_id(state, 4) == 1);  // Bulbasaur
  REQUIRE(metamon_switch_label_to_action_id(state, 5) == 4);  // Charmander
  REQUIRE(metamon_switch_label_to_action_id(state, 6) == 3);  // Gengar
  REQUIRE(metamon_switch_label_to_action_id(state, 7) == 2);  // Pikachu
  REQUIRE(metamon_switch_label_to_action_id(state, 8) == 5);  // Squirtle
}

TEST_CASE("action_id_to_metamon_label: inverse of metamon_switch_label_to_action_id for every switch slot",
          "[action]") {
  BattleState state = make_full_team_state();

  for (int label = 4; label <= 8; ++label) {
    ActionId action = metamon_switch_label_to_action_id(state, label);
    REQUIRE(action != static_cast<ActionId>(-1));
    REQUIRE(action_id_to_metamon_label(state, action) == label);
  }
}

TEST_CASE("action_id_to_metamon_label: move ActionIds map 1:1 onto Metamon move labels, no species-sort involved",
          "[action]") {
  BattleState state = make_full_team_state();

  REQUIRE(action_id_to_metamon_label(state, kMoveActionOffset + 0) == 0);
  REQUIRE(action_id_to_metamon_label(state, kMoveActionOffset + 1) == 1);
  REQUIRE(action_id_to_metamon_label(state, kMoveActionOffset + 2) == 2);
  REQUIRE(action_id_to_metamon_label(state, kMoveActionOffset + 3) == 3);
}

TEST_CASE("Metamon mapping: a fainted teammate still occupies a real species-sorted bench position",
          "[action]") {
  // action_space.py's own species-sort has no fainted filter (a switch
  // TARGET would be illegal, but the position/mapping itself is still
  // well-defined) - mirrored here, not re-derived. Gengar (slot 3) fainted.
  BattleState state = make_full_team_state();
  state.my_team[3].fainted = true;
  state.my_team[3].hp_fraction = 0.0f;

  REQUIRE(metamon_switch_label_to_action_id(state, 6) == 3);  // Gengar, still mapped despite fainted
  REQUIRE(action_id_to_metamon_label(state, 3) == 6);
}

TEST_CASE("Metamon mapping: bench<5 (fewer than 6 real team members) - unfilled positions return no mapping",
          "[action]") {
  // Only 3 real Pokemon revealed (1 active + 2 bench) - a synthetic
  // fixture, since every real Showdown team has exactly 6 (see this
  // milestone's own edge-case note: this shape only reachably fires
  // against a hand-built test, never a live battle).
  BattleState state;
  state.my_team[0] = make_slot("Zapdos");
  state.my_team[1] = make_slot("Bulbasaur");
  state.my_team[2] = make_slot("Pikachu");
  state.my_active_slot = 0;
  // my_team[3..5] left default (unrevealed).

  REQUIRE(metamon_switch_label_to_action_id(state, 4) == 1);  // Bulbasaur - real
  REQUIRE(metamon_switch_label_to_action_id(state, 5) == 2);  // Pikachu - real
  REQUIRE(metamon_switch_label_to_action_id(state, 6) == static_cast<ActionId>(-1));  // no 3rd bench member
  REQUIRE(metamon_switch_label_to_action_id(state, 7) == static_cast<ActionId>(-1));
  REQUIRE(metamon_switch_label_to_action_id(state, 8) == static_cast<ActionId>(-1));

  // The reverse direction still resolves cleanly for the two real slots.
  REQUIRE(action_id_to_metamon_label(state, 1) == 4);
  REQUIRE(action_id_to_metamon_label(state, 2) == 5);
}
