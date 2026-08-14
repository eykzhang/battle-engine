#include "be/forward_model.hpp"

#include "be/eval.hpp"
#include "be/pokedex_table.hpp"

namespace be {
  namespace {
    float single_type_multiplier(Type defender_type, Type attacker_type) {
      if (defender_type == Type::None) return 1.0f;
      return kTypeChart[static_cast<int>(defender_type)][static_cast<int>(attacker_type)];
    }

    float type_effectiveness(Type defender_type1, Type defender_type2, Type attacker_type) {
      return single_type_multiplier(defender_type1, attacker_type) *
            single_type_multiplier(defender_type2, attacker_type);
    }

    // damage.py's estimate_stat(), unknown-stat branch only - the C++
    // forward model always estimates (see sample_damage_fraction's own
    // "always estimated" doc comment in forward_model.hpp).
    int estimate_offdef_stat(int base_stat, int level) {
      return int(int(2 * base_stat + 31) * level / 100) + 5;
    }

    // Same formula, HP's own +level+10 tail instead of the other five
    // stats' flat +5.
    int estimate_hp_stat(int base_hp, int level) {
      return int(int(2 * base_hp + 31) * level / 100) + level + 10;
    }

    // Standard gen multi-hit distribution (35/35/15/15% for 2/3/4/5 hits) -
    // the only real gen-9 variable range, per scripts/dump_movedex.py's own
    // module docstring. A fixed hit count (min_hits == max_hits) is
    // returned directly, no sampling needed.
    int sample_hit_count(int min_hits, int max_hits, Rng& rng) {
      if (min_hits == max_hits) return min_hits;
      float r = rng.uniform01();
      if (r < 0.35f) return 2;
      if (r < 0.70f) return 3;
      if (r < 0.85f) return 4;
      return 5;
    }
  }

float crit_chance(int crit_ratio) {
  switch (crit_ratio) {
  case 2: return 1.0f / 8.0f;
  case 3: return 1.0f / 2.0f;
  case 4:
  case 6: return 1.0f;
  default: return 1.0f / 24.0f;  // covers 0, 1, and anything unrecognized
}
}

float sample_damage_fraction(const PokemonSlot& attacker, const PokemonSlot& defender,
                              const MovedexEntry& move, Rng& rng) {
  // TODO(you): implement per forward_model.hpp's doc comment on
  // sample_damage_fraction(). Suggested shape, closely mirroring
  // damage.py's expected_damage() but sampling instead of averaging:
  //   1. Status category -> return 0.0f immediately.
  //   2. Sample the accuracy check: if move.accuracy == -1, always hits;
  //      otherwise rng.bernoulli(move.accuracy / 100.0f) must succeed, or
  //      return 0.0f (a miss).
  //   3. Fixed-damage moves (move.fixed_damage != 0): check type immunity
  //      via kTypeChart (return 0.0f if immune), then return
  //      (attacker.level if fixed_damage == -1 else fixed_damage) as a
  //      fraction of defender's estimated max HP - skip STAB/crit/roll
  //      entirely, matching damage.py's _fixed_damage() path.
  //   4. Otherwise (a real base-power move): estimate atk/def (or spa/spd,
  //      by move.category) from base_stats + level for BOTH mons - see
  //      this function's own "always estimated" simplification in
  //      forward_model.hpp. Compute the base damage formula (same
  //      int-truncating arithmetic damage.py's expected_damage() uses),
  //      then multiply by: STAB (1.5x if move.type is one of attacker's
  //      types, else 1.0), type effectiveness (kTypeChart, return 0.0f if
  //      it's exactly 0 - immune), a SAMPLED roll (rng.uniform_int(85,
  //      100) / 100.0f, not the fixed 0.925 average), and a SAMPLED crit
  //      multiplier (rng.bernoulli(crit_chance(move.crit_ratio)) ? 1.5f :
  //      1.0f, not the expected blend).
  //   5. Multi-hit: sample a hit count from move.min_hits/max_hits (equal
  //      -> that many; 2/5 -> rng.bernoulli-based 35/35/15/15% split - see
  //      forward_model.hpp's own doc comment), and sum that many
  //      INDEPENDENTLY re-rolled (roll, crit) instances of step 4's
  //      per-hit damage - one accuracy check total, not per hit.
  //   6. Divide the total by defender's estimated max HP to get the
  //      returned fraction (same convention damage.py's expected_damage()
  //      already uses).
  //
  // cpp/tests/test_forward_model.cpp has both a fixed-seed exact-value
  // test and a mean-of-N-samples-vs-damage.py-parity test (per the plan's
  // "Corrected test design" note on M5) - run `ctest --test-dir cpp/build`
  // after rebuilding to check your implementation against them.
  if (move.category == MoveCategory::Status) return 0.0f;
  if (move.accuracy != -1 && !rng.bernoulli(move.accuracy / 100.0f)) return 0.0f;

  int defender_hp = estimate_hp_stat(defender.base_stats.hp, defender.level);

  if (move.fixed_damage != 0) {
    if (type_effectiveness(defender.type1, defender.type2, move.type) == 0.0f) return 0.0f;
    float fixed = (move.fixed_damage == -1) ? float(attacker.level) : float(move.fixed_damage);
    return fixed / defender_hp;
  }

  float type_eff = type_effectiveness(defender.type1, defender.type2, move.type);
  if (type_eff == 0.0f) return 0.0f;

  bool physical = move.category == MoveCategory::Physical;
  int atk = estimate_offdef_stat(physical ? attacker.base_stats.atk : attacker.base_stats.spa, attacker.level);
  int def = estimate_offdef_stat(physical ? defender.base_stats.def : defender.base_stats.spd, defender.level);
  int base = int(int(int(2 * attacker.level / 5 + 2) * move.base_power * atk / def) / 50) + 2;
  float stab = (move.type == attacker.type1 || move.type == attacker.type2) ? 1.5f : 1.0f;

  int hits = sample_hit_count(move.min_hits, move.max_hits, rng);
  float total = 0.0f;
  for (int i = 0; i < hits; ++i) {
    float roll = rng.uniform_int(85, 100) / 100.0f;
    float crit = rng.bernoulli(crit_chance(move.crit_ratio)) ? 1.5f : 1.0f;
    total += base * stab * type_eff * roll * crit;
  }
  return total / defender_hp;
}

void apply_switch_in_hazards(PokemonSlot& incoming, const SideConditions& hazards) {
  // TODO(you): implement per forward_model.hpp's doc comment on
  // apply_switch_in_hazards(). Suggested shape:
  //   1. Estimate incoming's real max HP from base_stats.hp + level (same
  //      known-vs-estimated pattern used elsewhere - HP is never "known
  //      exactly" in this struct the way spe_stat can be, so always
  //      estimate here).
  //   2. If hazards.stealth_rock: damage = (1/8) * max_hp *
  //      type_effectiveness(incoming's type1/type2 as DEFENDER, Rock as
  //      ATTACKER) - reuse the same kTypeChart[defender][attacker] lookup
  //      eval.hpp's type_matchup_score() uses (product across both of
  //      incoming's types if dual-typed, same "no second-type factor if
  //      Type::None" rule).
  //   3. If hazards.spikes_layers > 0 AND incoming is not Ground-immune
  //      (kTypeChart[incoming's types][Type::Ground] product == 0):
  //      damage = max_hp * (1/8, 1/6, or 1/4 for 1/2/3 layers).
  //   4. Subtract total hazard damage from incoming.hp_fraction (clamped
  //      to >= 0), and set incoming.fainted = true if it reaches 0.
  //
  // cpp/tests/test_forward_model.cpp covers Stealth Rock type-scaling,
  // Spikes layer-scaling, Ground-immunity skipping Spikes, and a combined
  // lethal case.
  (void)incoming;
  (void)hazards;
}

TurnResolution resolve_turn(BattleState& state, ActionId my_action, ActionId opp_action, Rng& rng) {
  // TODO(you): implement per forward_model.hpp's doc comment on
  // resolve_turn(). Suggested shape:
  //   1. Determine turn order: a switch action (< kNumSwitchActions)
  //      always resolves before either side's move (real Showdown
  //      priority +6); between two moves, compare
  //      lookup_movedex(...)->priority first, then speed_control_score()
  //      -style comparison (reuse spe_stat/boost_spe/status the same way
  //      eval.hpp's speed_control_score() does - don't re-derive the
  //      formula), then rng.bernoulli(0.5f) on an exact tie.
  //   2. Resolve the first mover's action:
  //      - switch: state.my_team[my_active_slot] (or opp's) becomes the
  //        new active slot; call apply_switch_in_hazards() on the
  //        incoming slot immediately (this IS a switch-in, unlike a
  //        later forced-replacement switch which is a separate caller
  //        responsibility per this header's own doc comment).
  //      - move: sample_damage_fraction() against the OTHER side's
  //        active mon, subtract from its hp_fraction (clamped >= 0), set
  //        fainted = true at 0. If the defender just fainted, the second
  //        mover's action may need to be skipped or altered (a fainted
  //        active mon can't act, but CAN still be the switch target of
  //        its own side's pending switch action if that's what was
  //        chosen - i.e. only skip the second mover if their action was
  //        a MOVE, not a switch).
  //   3. Resolve the second mover's action the same way (skipping a move
  //      action whose user fainted in step 2, per the note above).
  //   4. Determine the return value from both sides' final fainted state
  //      this turn: kTerminal takes priority if a side has ZERO
  //      non-fainted team members left anywhere (not just the active
  //      one) - check the whole team array, not just active_slot. Else
  //      kBothFainted / kMyFainted / kOppFainted / kContinue based on
  //      just the two actives.
  //
  // cpp/tests/test_forward_model.cpp has a fixed-seed determinism test
  // (same seed + inputs -> identical resulting state) plus turn-order
  // fixtures (priority beats speed, speed breaks a priority tie, a 50/50
  // speed tie is genuinely sampled both ways across many seeds).
  (void)state;
  (void)my_action;
  (void)opp_action;
  (void)rng;
  return TurnResolution::kContinue;
}

}  // namespace be
