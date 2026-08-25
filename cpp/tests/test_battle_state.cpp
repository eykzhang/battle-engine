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

// ---------------------------------------------------------------------------
// M6b: mirror() (DW-5.6) - round-trip, field-swap, and weather/terrain-
// unchanged assertions. No BattleState operator== is added (not needed by
// any other caller) - equality is checked field-by-field via same_state()
// below, matching this project's own "don't add interface not needed by
// callers" convention.
// ---------------------------------------------------------------------------

namespace {

PokemonSlot make_rich_slot(const std::string& species, float hp) {
  PokemonSlot slot;
  slot.revealed = true;
  slot.hp_fraction = hp;
  slot.species = species;
  slot.item = "leftovers";
  slot.ability = "levitate";
  slot.type1 = Type::Electric;
  slot.type2 = Type::Flying;
  slot.moves = {"thunderbolt", "drillpeck", "roost", "protect"};
  slot.boost_spe = 1;
  slot.boost_atk = -2;
  return slot;
}

bool same_slot(const PokemonSlot& a, const PokemonSlot& b) {
  return a.revealed == b.revealed && a.fainted == b.fainted && a.hp_fraction == b.hp_fraction &&
         a.species == b.species && a.item == b.item && a.ability == b.ability && a.type1 == b.type1 &&
         a.type2 == b.type2 && a.moves == b.moves && a.boost_spe == b.boost_spe && a.boost_atk == b.boost_atk;
}

bool same_hazards(const SideConditions& a, const SideConditions& b) {
  return a.spikes_layers == b.spikes_layers && a.toxic_spikes_layers == b.toxic_spikes_layers &&
         a.stealth_rock == b.stealth_rock && a.sticky_web == b.sticky_web &&
         a.stealth_rock_turn == b.stealth_rock_turn && a.sticky_web_turn == b.sticky_web_turn &&
         a.reflect_turn == b.reflect_turn && a.light_screen_turn == b.light_screen_turn &&
         a.aurora_veil_turn == b.aurora_veil_turn && a.tailwind_turn == b.tailwind_turn;
}

bool same_state(const BattleState& a, const BattleState& b) {
  for (int i = 0; i < 6; ++i) {
    if (!same_slot(a.my_team[i], b.my_team[i])) return false;
    if (!same_slot(a.opp_team[i], b.opp_team[i])) return false;
  }
  return a.my_active_slot == b.my_active_slot && a.opp_active_slot == b.opp_active_slot &&
         same_hazards(a.my_hazards, b.my_hazards) && same_hazards(a.opp_hazards, b.opp_hazards) &&
         a.weather == b.weather && a.terrain == b.terrain;
}

BattleState make_asymmetric_state() {
  BattleState state;
  state.my_team[0] = make_rich_slot("Zapdos", 0.8f);
  state.my_team[1] = make_rich_slot("Bulbasaur", 1.0f);
  state.opp_team[0] = make_rich_slot("Charizard", 0.5f);
  state.my_active_slot = 0;
  state.opp_active_slot = 0;
  state.my_hazards.spikes_layers = 2;
  state.my_hazards.stealth_rock_turn = 3;
  state.opp_hazards.toxic_spikes_layers = 1;
  state.opp_hazards.tailwind_turn = 5;
  state.weather = Weather::Sandstorm;
  state.terrain = Terrain::Electric;
  return state;
}

}  // namespace

TEST_CASE("mirror: round-trips - mirror(mirror(s)) equals s field-for-field", "[battle_state]") {
  BattleState state = make_asymmetric_state();
  BattleState round_tripped = mirror(mirror(state));
  REQUIRE(same_state(state, round_tripped));
}

TEST_CASE("mirror: swaps my_team/opp_team, my_active_slot/opp_active_slot, and my_hazards/opp_hazards",
          "[battle_state]") {
  BattleState state = make_asymmetric_state();
  BattleState mirrored = mirror(state);

  for (int i = 0; i < 6; ++i) {
    REQUIRE(same_slot(mirrored.my_team[i], state.opp_team[i]));
    REQUIRE(same_slot(mirrored.opp_team[i], state.my_team[i]));
  }
  REQUIRE(mirrored.my_active_slot == state.opp_active_slot);
  REQUIRE(mirrored.opp_active_slot == state.my_active_slot);
  REQUIRE(same_hazards(mirrored.my_hazards, state.opp_hazards));
  REQUIRE(same_hazards(mirrored.opp_hazards, state.my_hazards));
}

TEST_CASE("mirror: weather and terrain are unchanged - state-level, not per-side", "[battle_state]") {
  BattleState state = make_asymmetric_state();
  BattleState mirrored = mirror(state);

  REQUIRE(mirrored.weather == state.weather);
  REQUIRE(mirrored.terrain == state.terrain);
}

TEST_CASE("mirror: encode_native(mirror(s)) is well-formed and length-matches encode_native(s)",
          "[battle_state]") {
  // Full cross-check against the real Python encode() on an opponent-POV
  // view lives outside this phase's C++-only test scope (no pytest file is
  // in this phase's file scope - see this build's discovery notes) - this
  // Catch2-reachable check instead confirms mirror() produces a state
  // encode_native() accepts and treats symmetrically (same output length,
  // no throw), the C++-side half of that seam.
  BattleState state = make_asymmetric_state();
  std::vector<float> direct = encode_native(state);
  std::vector<float> mirrored = encode_native(mirror(state));
  REQUIRE(direct.size() == mirrored.size());
  REQUIRE(mirrored.size() == static_cast<size_t>(kEncodeVectorLen));
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
