#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "be/forward_model.hpp"

using namespace be;
using Catch::Approx;

// ---------------------------------------------------------------------------
// crit_chance
// ---------------------------------------------------------------------------

TEST_CASE("crit_chance: matches damage.py's _CRIT_CHANCE_BY_RATIO exactly", "[forward_model]") {
  REQUIRE(crit_chance(0) == Approx(1.0f / 24.0f));
  REQUIRE(crit_chance(1) == Approx(1.0f / 24.0f));
  REQUIRE(crit_chance(2) == Approx(1.0f / 8.0f));
  REQUIRE(crit_chance(3) == Approx(1.0f / 2.0f));
  REQUIRE(crit_chance(4) == Approx(1.0f));
  REQUIRE(crit_chance(6) == Approx(1.0f));  // poke-env's "willCrit" marker
}

TEST_CASE("crit_chance: an unrecognized ratio falls back to the base rate", "[forward_model]") {
  // Mirrors damage.py's dict.get(move.crit_ratio, 1 / 24) fallback - no
  // real gen-9 move actually has crit_ratio 5, but the fallback should
  // still behave sanely rather than crash or return something wild.
  REQUIRE(crit_chance(5) == Approx(1.0f / 24.0f));
}

// ---------------------------------------------------------------------------
// sample_damage_fraction
// ---------------------------------------------------------------------------

namespace {

PokemonSlot make_attacker() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.type1 = Type::Fire;  // deliberately not the move's type - no STAB
  slot.level = 100;
  slot.base_stats = {100, 100, 100, 100, 100, 100};
  return slot;
}

PokemonSlot make_defender() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.type1 = Type::Normal;  // neutral against the test move's Water type
  slot.level = 100;
  slot.base_stats = {100, 100, 100, 100, 100, 100};
  return slot;
}

MovedexEntry make_move(int base_power, Type type, MoveCategory category, int accuracy) {
  MovedexEntry move{};
  move.base_power = base_power;
  move.type = type;
  move.category = category;
  move.priority = 0;
  move.crit_ratio = 0;
  move.accuracy = accuracy;
  move.fixed_damage = 0;
  move.min_hits = 1;
  move.max_hits = 1;
  return move;
}

}  // namespace

TEST_CASE("sample_damage_fraction: a Status move always deals zero damage", "[forward_model]") {
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  MovedexEntry move = make_move(/*base_power=*/0, Type::Normal, MoveCategory::Status, /*accuracy=*/100);
  Rng rng(1);
  REQUIRE(sample_damage_fraction(attacker, defender, move, rng) == Approx(0.0f));
}

TEST_CASE("sample_damage_fraction: type immunity deals zero damage regardless of base power", "[forward_model]") {
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  defender.type1 = Type::Ghost;  // immune to Normal-type moves
  MovedexEntry move = make_move(/*base_power=*/100, Type::Normal, MoveCategory::Physical, /*accuracy=*/100);
  Rng rng(1);
  REQUIRE(sample_damage_fraction(attacker, defender, move, rng) == Approx(0.0f));
}

TEST_CASE("sample_damage_fraction: a guaranteed miss (0% accuracy) deals zero damage", "[forward_model]") {
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  MovedexEntry move = make_move(/*base_power=*/80, Type::Water, MoveCategory::Special, /*accuracy=*/0);
  Rng rng(1);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(sample_damage_fraction(attacker, defender, move, rng) == Approx(0.0f));
  }
}

TEST_CASE("sample_damage_fraction: fixed-damage moves ignore base_power/STAB/crit/roll", "[forward_model]") {
  PokemonSlot attacker = make_attacker();
  attacker.level = 50;
  PokemonSlot defender = make_defender();
  // Real fixed-damage moves are Physical/Special, never Status (e.g.
  // Seismic Toss is Physical) - category matters here because
  // sample_damage_fraction's Status short-circuit must NOT swallow this
  // case.
  MovedexEntry move = make_move(/*base_power=*/0, Type::Fighting, MoveCategory::Physical, /*accuracy=*/100);
  move.fixed_damage = -1;  // 'level'-based (Seismic Toss / Night Shade)
  Rng rng(1);
  // damage = attacker.level (50), as a fraction of defender's estimated
  // max HP: floor(floor(2*100+31)*100/100) + 100 + 10 = 341 (base_stats.hp
  // = 100, level = 100 - see make_defender()).
  REQUIRE(sample_damage_fraction(attacker, defender, move, rng) == Approx(50.0f / 341.0f).margin(0.001f));
}

TEST_CASE("sample_damage_fraction: fixed-damage moves still respect type immunity", "[forward_model]") {
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  defender.type1 = Type::Ghost;  // immune to Normal
  MovedexEntry move = make_move(/*base_power=*/0, Type::Normal, MoveCategory::Special, /*accuracy=*/100);
  move.fixed_damage = 40;  // Dragon Rage-style flat damage, but wrong type on purpose
  Rng rng(1);
  REQUIRE(sample_damage_fraction(attacker, defender, move, rng) == Approx(0.0f));
}

TEST_CASE("sample_damage_fraction: same seed produces bit-identical results (determinism)", "[forward_model]") {
  // Cheap smoke test per the plan's M5 "Corrected test design" note - not
  // a check against an independently-computed number (implementation
  // order of the accuracy/crit/roll draws is the implementer's choice),
  // just "the same seed always reproduces the same outcome."
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  MovedexEntry move = make_move(/*base_power=*/80, Type::Water, MoveCategory::Special, /*accuracy=*/100);

  Rng rng_a(42);
  Rng rng_b(42);
  float a = sample_damage_fraction(attacker, defender, move, rng_a);
  float b = sample_damage_fraction(attacker, defender, move, rng_b);
  REQUIRE(a == b);
}

TEST_CASE("sample_damage_fraction: mean over many samples matches damage.py's expected_damage formula", "[forward_model]") {
  // A real statistical correctness check (per the plan's M5 note, this is
  // the (b) option, not just a smoke test). All stats/levels are equal and
  // untyped-relative-to-the-move (no STAB, no resistance/weakness) so the
  // expected value has a clean hand-derivation:
  //   estimate_stat(atk) == estimate_stat(def) == floor(floor(2*100+31)*100/100)+5 == 236
  //   estimate_stat(hp)  == floor(floor(2*100+31)*100/100)+100+10 == 341
  //   base = floor(floor(42 * 80 * 236 / 236) / 50) + 2 = floor(3360/50)+2 = 69
  //   expected_hit_damage = base * 1.0(stab) * 1.0(type) * 0.925(avg roll)
  //                         * (1 + 0.5/24)(expected crit mult, ratio 0)
  //   expected_fraction = expected_hit_damage / 341
  PokemonSlot attacker = make_attacker();
  PokemonSlot defender = make_defender();
  MovedexEntry move = make_move(/*base_power=*/80, Type::Water, MoveCategory::Special, /*accuracy=*/100);

  const double base = 69.0;
  const double expected_crit_mult = (23.0 / 24.0) * 1.0 + (1.0 / 24.0) * 1.5;
  const double expected_fraction = (base * 0.925 * expected_crit_mult) / 341.0;

  constexpr int kSamples = 20000;
  double total = 0.0;
  Rng rng(7);
  for (int i = 0; i < kSamples; ++i) {
    total += sample_damage_fraction(attacker, defender, move, rng);
  }
  double mean = total / kSamples;

  REQUIRE(mean == Approx(expected_fraction).margin(0.003));
}

// ---------------------------------------------------------------------------
// apply_switch_in_hazards
// ---------------------------------------------------------------------------

TEST_CASE("apply_switch_in_hazards: Stealth Rock scales with the incoming mon's typing", "[forward_model]") {
  PokemonSlot incoming;
  incoming.revealed = true;
  incoming.hp_fraction = 1.0f;
  incoming.type1 = Type::Fire;  // 2x weak to Rock
  SideConditions hazards;
  hazards.stealth_rock = true;

  apply_switch_in_hazards(incoming, hazards);

  // 1/8 max HP * 2x weakness = 1/4 max HP.
  REQUIRE(incoming.hp_fraction == Approx(0.75f));
  REQUIRE_FALSE(incoming.fainted);
}

TEST_CASE("apply_switch_in_hazards: Spikes scale with layer count, ignoring typing", "[forward_model]") {
  PokemonSlot incoming;
  incoming.revealed = true;
  incoming.hp_fraction = 1.0f;
  incoming.type1 = Type::Normal;
  SideConditions hazards;
  hazards.spikes_layers = 2;

  apply_switch_in_hazards(incoming, hazards);

  REQUIRE(incoming.hp_fraction == Approx(1.0f - 1.0f / 6.0f));
}

TEST_CASE("apply_switch_in_hazards: a Ground-immune type takes no Spikes damage", "[forward_model]") {
  PokemonSlot incoming;
  incoming.revealed = true;
  incoming.hp_fraction = 1.0f;
  incoming.type1 = Type::Flying;
  SideConditions hazards;
  hazards.spikes_layers = 3;

  apply_switch_in_hazards(incoming, hazards);

  REQUIRE(incoming.hp_fraction == Approx(1.0f));
}

TEST_CASE("apply_switch_in_hazards: combined hazard damage can faint the incoming mon", "[forward_model]") {
  PokemonSlot incoming;
  incoming.revealed = true;
  incoming.hp_fraction = 0.4f;
  incoming.type1 = Type::Fire;  // 2x weak to Rock, not Ground-immune
  SideConditions hazards;
  hazards.stealth_rock = true;
  hazards.spikes_layers = 3;

  apply_switch_in_hazards(incoming, hazards);

  // SR: 1/4 max HP. Spikes at 3 layers: 1/4 max HP. Combined 1/2 > the
  // starting 0.4 hp_fraction - must clamp to exactly 0, not go negative.
  REQUIRE(incoming.hp_fraction == Approx(0.0f));
  REQUIRE(incoming.fainted);
}

// ---------------------------------------------------------------------------
// resolve_turn
// ---------------------------------------------------------------------------

namespace {

// A minimal but fully "legal" 1v1 state: both sides have exactly one
// revealed, active, non-fainted Pokemon with one real, known move each,
// PLUS 5 full-HP filler bench slots per side. Real Showdown move ids (not
// synthetic ones) are used deliberately - the implementation is expected
// to resolve them via lookup_movedex(), same as a real translated battle
// would. Move slot 0 is Tackle (priority 0), slot 1 is Quick Attack
// (priority 1) - callers pick which slot to submit as the action to
// control priority in a given test.
//
// The filler bench slots exist so these tests exercise ONLY the turn-order/
// damage mechanics they're named for, not resolve_turn's separate (and
// genuinely ambiguous under Tier 1 - see forward_model.hpp's own note on
// TurnResolution::kTerminal) "has this side's whole team been wiped out"
// question - a fixture with just one populated slot per side would
// conflate "the active mon fainted" with "the team is empty," which isn't
// the same thing for a real 6-Pokemon side.
PokemonSlot make_filler_bench_slot() {
  PokemonSlot slot;
  slot.revealed = true;
  slot.hp_fraction = 1.0f;
  slot.level = 100;
  slot.base_stats = {100, 100, 100, 100, 100, 100};
  return slot;
}

BattleState make_lethal_duel(int my_spe, int opp_spe) {
  BattleState state;

  PokemonSlot me;
  me.revealed = true;
  me.hp_fraction = 0.01f;  // any nonzero damage guarantees a faint
  me.level = 100;
  me.base_stats = {100, 100, 100, 100, 100, 100};
  me.spe_stat = my_spe;
  me.moves = {"tackle", "quickattack", "", ""};
  state.my_team[0] = me;
  state.my_active_slot = 0;

  PokemonSlot opp;
  opp.revealed = true;
  opp.hp_fraction = 0.01f;
  opp.level = 100;
  opp.base_stats = {100, 100, 100, 100, 100, 100};
  opp.spe_stat = opp_spe;
  opp.moves = {"tackle", "quickattack", "", ""};
  state.opp_team[0] = opp;
  state.opp_active_slot = 0;

  for (int i = 1; i < 6; ++i) {
    state.my_team[i] = make_filler_bench_slot();
    state.opp_team[i] = make_filler_bench_slot();
  }

  return state;
}

}  // namespace

TEST_CASE("resolve_turn: a higher-priority move acts first, pre-empting a slower faint", "[forward_model]") {
  // Both moves are guaranteed (100% accuracy, nonzero base power) KOs
  // against the 0.01 hp_fraction targets - Quick Attack (priority 1) must
  // resolve before Tackle (priority 0) regardless of equal speed, so the
  // opponent should faint WITHOUT ever landing its own retaliation.
  BattleState state = make_lethal_duel(/*my_spe=*/100, /*opp_spe=*/100);
  Rng rng(1);

  TurnResolution result = resolve_turn(state, kMoveActionOffset + 1 /*quickattack*/,
                                        kMoveActionOffset + 0 /*tackle*/, rng);

  REQUIRE(result == TurnResolution::kOppFainted);
  REQUIRE(state.opp_team[0].fainted);
  REQUIRE_FALSE(state.my_team[0].fainted);
  REQUIRE(state.my_team[0].hp_fraction == Approx(0.01f));  // untouched - opp never acted
}

TEST_CASE("resolve_turn: equal priority falls back to comparing Speed", "[forward_model]") {
  BattleState state = make_lethal_duel(/*my_spe=*/200, /*opp_spe=*/100);
  Rng rng(1);

  TurnResolution result = resolve_turn(state, kMoveActionOffset + 0 /*tackle*/,
                                        kMoveActionOffset + 0 /*tackle*/, rng);

  REQUIRE(result == TurnResolution::kOppFainted);
  REQUIRE(state.opp_team[0].fainted);
  REQUIRE_FALSE(state.my_team[0].fainted);
}

TEST_CASE("resolve_turn: an exact speed AND priority tie is genuinely sampled both ways", "[forward_model]") {
  bool saw_my_faint = false;
  bool saw_opp_faint = false;
  for (uint64_t seed = 0; seed < 200; ++seed) {
    BattleState state = make_lethal_duel(/*my_spe=*/100, /*opp_spe=*/100);
    Rng rng(seed);
    TurnResolution result = resolve_turn(state, kMoveActionOffset + 0, kMoveActionOffset + 0, rng);
    if (result == TurnResolution::kMyFainted) saw_my_faint = true;
    if (result == TurnResolution::kOppFainted) saw_opp_faint = true;
  }
  REQUIRE(saw_my_faint);
  REQUIRE(saw_opp_faint);
}

TEST_CASE("resolve_turn: same seed and inputs produce an identical resulting state (determinism)", "[forward_model]") {
  BattleState state_a = make_lethal_duel(/*my_spe=*/100, /*opp_spe=*/90);
  BattleState state_b = make_lethal_duel(/*my_spe=*/100, /*opp_spe=*/90);
  Rng rng_a(123);
  Rng rng_b(123);

  TurnResolution result_a = resolve_turn(state_a, kMoveActionOffset + 0, kMoveActionOffset + 0, rng_a);
  TurnResolution result_b = resolve_turn(state_b, kMoveActionOffset + 0, kMoveActionOffset + 0, rng_b);

  REQUIRE(result_a == result_b);
  REQUIRE(state_a.my_team[0].hp_fraction == state_b.my_team[0].hp_fraction);
  REQUIRE(state_a.opp_team[0].hp_fraction == state_b.opp_team[0].hp_fraction);
  REQUIRE(state_a.my_team[0].fainted == state_b.my_team[0].fainted);
  REQUIRE(state_a.opp_team[0].fainted == state_b.opp_team[0].fainted);
}

TEST_CASE("resolve_turn: a switch always resolves before either side's move", "[forward_model]") {
  // My switch-in should take its action before the opponent's Tackle can
  // land on the mon that was active at the start of the turn.
  BattleState state = make_lethal_duel(/*my_spe=*/1, /*opp_spe=*/200);  // opponent much faster
  PokemonSlot bench;
  bench.revealed = true;
  bench.hp_fraction = 1.0f;
  bench.level = 100;
  bench.base_stats = {100, 100, 100, 100, 100, 100};
  bench.spe_stat = 1;
  bench.moves = {"tackle", "", "", ""};
  state.my_team[1] = bench;
  Rng rng(1);

  TurnResolution result = resolve_turn(state, /*my_action=*/1 /*switch to slot 1*/,
                                        kMoveActionOffset + 0 /*opp tackle*/, rng);

  // The opponent's Tackle should hit whichever Pokemon is active AFTER the
  // switch (my_team[1], full HP) - not the original my_team[0] (0.01 HP,
  // which never got attacked at all since it switched out first).
  REQUIRE(state.my_active_slot == 1);
  REQUIRE(state.my_team[0].hp_fraction == Approx(0.01f));  // untouched
  REQUIRE(result != TurnResolution::kMyFainted);  // one real hit can't KO a full-HP mon here
}
