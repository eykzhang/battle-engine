#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "be/eval.hpp"

using namespace be;
using Catch::Approx;

// Expected multipliers below are pulled directly from
// poke_env.data.GenData.from_gen(9).type_chart (verified via a real
// interactive check before writing these, not from memory of the games) -
// see eval.hpp's comment on kTypeChart's [defender][attacker] index order.

TEST_CASE("type_matchup_score: single types, favorable matchup", "[eval]") {
  PokemonSlot water;
  water.type1 = Type::Water;
  PokemonSlot fire;
  fire.type1 = Type::Fire;

  // offense: Fire (opponent) takes from Water = 2.0
  // defense: Water (mon) takes from Fire = 0.5
  REQUIRE(type_matchup_score(water, fire) == Approx(1.5f));
}

TEST_CASE("type_matchup_score: single types, unfavorable matchup", "[eval]") {
  PokemonSlot grass;
  grass.type1 = Type::Grass;
  PokemonSlot fire;
  fire.type1 = Type::Fire;

  // offense: Fire (opponent) takes from Grass = 0.5
  // defense: Grass (mon) takes from Fire = 2.0
  REQUIRE(type_matchup_score(grass, fire) == Approx(-1.5f));
}

TEST_CASE("type_matchup_score: dual-typed mon takes the best offensive type", "[eval]") {
  PokemonSlot mon;  // synthetic Fire/Water dual type - not a real species,
  mon.type1 = Type::Fire;  // just exercising the "max over my types" logic
  mon.type2 = Type::Water;
  PokemonSlot opponent;
  opponent.type1 = Type::Grass;

  // offense: max(Grass takes from Fire = 2.0, Grass takes from Water = 0.5) = 2.0
  // defense: mon (Fire/Water) takes from Grass = 0.5 * 2.0 = 1.0 (product of both types)
  REQUIRE(type_matchup_score(mon, opponent) == Approx(1.0f));
}

TEST_CASE("type_matchup_score: dual-typed opponent takes the best defensive type", "[eval]") {
  PokemonSlot mon;
  mon.type1 = Type::Grass;
  PokemonSlot opponent;  // synthetic Fire/Water dual type
  opponent.type1 = Type::Fire;
  opponent.type2 = Type::Water;

  // offense: opponent (Fire/Water) takes from Grass = 0.5 * 2.0 = 1.0
  // defense: max(Grass takes from Fire = 2.0, Grass takes from Water = 0.5) = 2.0
  REQUIRE(type_matchup_score(mon, opponent) == Approx(-1.0f));
}

TEST_CASE("type_matchup_score: mutual immunity nets to zero", "[eval]") {
  PokemonSlot ghost;
  ghost.type1 = Type::Ghost;
  PokemonSlot normal;
  normal.type1 = Type::Normal;

  // Ghost and Normal are mutually immune to each other in gen 9.
  REQUIRE(type_matchup_score(ghost, normal) == Approx(0.0f));
}

TEST_CASE("hazard_score: no hazards", "[eval]") {
  REQUIRE(hazard_score(SideConditions{}) == Approx(0.0f));
}

TEST_CASE("hazard_score: stacked layer hazards scale linearly", "[eval]") {
  SideConditions conditions;
  conditions.spikes_layers = 3;
  conditions.toxic_spikes_layers = 2;
  // 3*0.15 (spikes) + 2*0.15 (toxic spikes) = 0.75
  REQUIRE(hazard_score(conditions) == Approx(0.75f));
}

TEST_CASE("hazard_score: presence-only hazards are flat regardless of stacking", "[eval]") {
  SideConditions conditions;
  conditions.stealth_rock = true;
  conditions.sticky_web = true;
  REQUIRE(hazard_score(conditions) == Approx(0.4f));  // 0.2 + 0.2
}

TEST_CASE("hazard_score: all four hazards combined", "[eval]") {
  SideConditions conditions;
  conditions.spikes_layers = 2;
  conditions.toxic_spikes_layers = 1;
  conditions.stealth_rock = true;
  conditions.sticky_web = true;
  // 2*0.15 + 1*0.15 + 0.2 + 0.2 = 0.85
  REQUIRE(hazard_score(conditions) == Approx(0.85f));
}

TEST_CASE("speed_control_score: known stats, mine faster", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 150;
  PokemonSlot opp;
  opp.spe_stat = 100;
  REQUIRE(speed_control_score(me, opp) == Approx(1.0f));
}

TEST_CASE("speed_control_score: known stats, opponent faster", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 100;
  PokemonSlot opp;
  opp.spe_stat = 150;
  REQUIRE(speed_control_score(me, opp) == Approx(-1.0f));
}

TEST_CASE("speed_control_score: exact tie is zero", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 100;
  PokemonSlot opp;
  opp.spe_stat = 100;
  REQUIRE(speed_control_score(me, opp) == Approx(0.0f));
}

TEST_CASE("speed_control_score: positive boost stage multiplies correctly", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 100;
  me.boost_spe = 2;  // multiplier (2+2)/2 = 2.0 -> effective 200
  PokemonSlot opp;
  opp.spe_stat = 150;  // effective 150, no boost
  REQUIRE(speed_control_score(me, opp) == Approx(1.0f));
}

TEST_CASE("speed_control_score: negative boost stage multiplies correctly", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 100;
  me.boost_spe = -2;  // multiplier 2/(2-(-2)) = 0.5 -> effective 50
  PokemonSlot opp;
  opp.spe_stat = 100;  // effective 100, no boost
  REQUIRE(speed_control_score(me, opp) == Approx(-1.0f));
}

TEST_CASE("speed_control_score: paralysis halves effective speed", "[eval]") {
  PokemonSlot me;
  me.spe_stat = 200;
  me.status = Status::Par;  // effective 100
  PokemonSlot opp;
  opp.spe_stat = 150;  // effective 150, faster despite lower raw stat
  REQUIRE(speed_control_score(me, opp) == Approx(-1.0f));
}

TEST_CASE("speed_control_score: unknown stat is estimated from base_stats + level, truncating not rounding", "[eval]") {
  // estimate formula: int(int(2*base + 31) * level / 100) + 5
  // base=100, level=50: int(231) * 50 / 100 = 115.5 -> truncates to 115, + 5 = 120 (exact)
  // A naive round()-based implementation would give 116 + 5 = 121 instead -
  // this test's opp stat is deliberately set to exactly 120 so truncation
  // gives a tie (0.0) while rounding would wrongly give +1.0.
  PokemonSlot me;
  me.spe_stat = -1;  // unknown, must estimate
  me.base_stats.spe = 100;
  me.level = 50;
  PokemonSlot opp;
  opp.spe_stat = 120;
  REQUIRE(speed_control_score(me, opp) == Approx(0.0f));
}

TEST_CASE("evaluate: symmetric empty-ish state nets to zero", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 1.0f;
  // No active Pokemon on either side (-1) - type-matchup/speed terms should
  // be skipped entirely, not treated as a 0-0 tie contribution.
  REQUIRE(evaluate(state) == Approx(0.0f));
}

TEST_CASE("evaluate: HP fraction advantage scores positively", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 0.5f;
  // kHpFractionWeight * (1.0 - 0.5) = 0.5
  REQUIRE(evaluate(state) == Approx(0.5f));
}

TEST_CASE("evaluate: unrevealed opponent slots are skipped, not counted as fainted", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 1.0f;
  // opp_team[1..5] default to revealed == false - must not be summed as if
  // they were 0 revealed HP or fainted (that would wrongly favor "me" by
  // 5 alive-count/HP-fraction's worth of phantom advantage).
  REQUIRE(evaluate(state) == Approx(0.0f));
}

TEST_CASE("evaluate: my own status affliction is a penalty, not a bonus", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.my_team[0].status = Status::Par;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 1.0f;
  // kStatusWeight * (opp_afflicted_count(0) - my_afflicted_count(1)) = 0.3 * (0 - 1) = -0.3
  REQUIRE(evaluate(state) == Approx(-0.3f));
}

TEST_CASE("evaluate: fainted mon counts against alive_count, not status", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 0.0f;
  state.opp_team[0].fainted = true;
  state.opp_team[0].status = Status::Fnt;
  // HP: 1.0 * (1.0 - 0.0) = 1.0
  // alive: 0.5 * (1 - 0) = 0.5
  // status: Status::Fnt must NOT count as "afflicted" (matches
  // evaluation.py's `status not in (None, FNT)`) - opp afflicted count is 0.
  REQUIRE(evaluate(state) == Approx(1.5f));
}

TEST_CASE("evaluate: hazards favor the side with fewer hazards on their side", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 1.0f;
  state.my_hazards.stealth_rock = true;  // hurts me
  // hazard_score(opp_hazards=0) - hazard_score(my_hazards=0.2) = -0.2
  REQUIRE(evaluate(state) == Approx(-0.2f));
}

TEST_CASE("evaluate: type-matchup and speed terms only apply when both sides have an active mon", "[eval]") {
  BattleState state;
  state.my_team[0].revealed = true;
  state.my_team[0].hp_fraction = 1.0f;
  state.my_team[0].type1 = Type::Water;
  state.my_team[0].spe_stat = 150;
  state.opp_team[0].revealed = true;
  state.opp_team[0].hp_fraction = 1.0f;
  state.opp_team[0].type1 = Type::Fire;
  state.opp_team[0].spe_stat = 100;
  state.my_active_slot = 0;
  state.opp_active_slot = 0;
  // type: 0.5 * 1.5 = 0.75; speed: 0.2 * 1.0 = 0.2; total on top of HP-tie's 0.0
  REQUIRE(evaluate(state) == Approx(0.95f));
}
