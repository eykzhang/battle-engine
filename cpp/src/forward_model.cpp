#include "be/forward_model.hpp"

#include <algorithm>

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

    bool is_switch_action(ActionId action) { return action < kNumSwitchActions; }

    // True iff "my" side's action resolves first this turn: a switch
    // always pre-empts either side's move (real Showdown +6 priority);
    // between two switches the order doesn't interact (neither damages
    // the other), so it's arbitrary; between two moves, compare
    // MovedexEntry::priority, then speed_control_score()'s sign, then a
    // genuine 50/50 coin flip on an exact tie.
    bool my_side_first(const BattleState& state, ActionId my_action, ActionId opp_action, Rng& rng) {
      bool my_switch = is_switch_action(my_action);
      bool opp_switch = is_switch_action(opp_action);
      if (my_switch != opp_switch) return my_switch;
      if (my_switch && opp_switch) return true;

      const PokemonSlot& my_active = state.my_team[state.my_active_slot];
      const PokemonSlot& opp_active = state.opp_team[state.opp_active_slot];
      const MovedexEntry* my_move = lookup_movedex(my_active.moves[my_action - kMoveActionOffset].c_str());
      const MovedexEntry* opp_move = lookup_movedex(opp_active.moves[opp_action - kMoveActionOffset].c_str());
      if (my_move->priority != opp_move->priority) return my_move->priority > opp_move->priority;

      float speed_cmp = speed_control_score(my_active, opp_active);
      if (speed_cmp != 0.0f) return speed_cmp > 0.0f;
      return rng.bernoulli(0.5f);
    }

    // Applies one already-legal action for `side`, mutating `state` in
    // place - either a switch (updates the active slot, then applies
    // switch-in hazard chip damage to the incoming mon) or a move
    // (samples damage against the OTHER side's current active mon).
    void apply_action(BattleState& state, Side side, ActionId action, Rng& rng) {
      bool is_me = (side == Side::Me);
      TeamSlot& active_slot = is_me ? state.my_active_slot : state.opp_active_slot;
      std::array<PokemonSlot, 6>& team = is_me ? state.my_team : state.opp_team;
      SideConditions& hazards = is_me ? state.my_hazards : state.opp_hazards;

      if (is_switch_action(action)) {
        active_slot = action;
        apply_switch_in_hazards(team[active_slot], hazards);
        return;
      }

      PokemonSlot& attacker = team[active_slot];
      std::array<PokemonSlot, 6>& other_team = is_me ? state.opp_team : state.my_team;
      TeamSlot other_active_slot = is_me ? state.opp_active_slot : state.my_active_slot;
      PokemonSlot& defender = other_team[other_active_slot];

      const MovedexEntry* move = lookup_movedex(attacker.moves[action - kMoveActionOffset].c_str());
      float dmg = sample_damage_fraction(attacker, defender, *move, rng);
      defender.hp_fraction = std::max(0.0f, defender.hp_fraction - dmg);
      if (defender.hp_fraction <= 0.0f) defender.fainted = true;
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
  int incoming_hp = estimate_hp_stat(incoming.base_stats.hp, incoming.level);
  float hazard_damage = 0.0f;
  if (hazards.stealth_rock) {
    hazard_damage += 0.125f * single_type_multiplier(incoming.type1, Type::Rock) * single_type_multiplier(incoming.type2, Type::Rock) * incoming_hp;
  }
  if (hazards.spikes_layers > 0 && single_type_multiplier(incoming.type1, Type::Ground) * single_type_multiplier(incoming.type2, Type::Ground) != 0.0f) {
    if (hazards.spikes_layers == 1) hazard_damage += 0.125f * incoming_hp;
    else if (hazards.spikes_layers == 2) hazard_damage += (1.0f/6.0f) * incoming_hp;
    else hazard_damage += 0.25f * incoming_hp;
  }
  
  incoming.hp_fraction = (incoming.hp_fraction - (hazard_damage / incoming_hp) >= 0) ? incoming.hp_fraction - (hazard_damage / incoming_hp) : 0.0f;
  if (incoming.hp_fraction == 0) incoming.fainted = true;
}

TurnResolution resolve_turn(BattleState& state, ActionId my_action, ActionId opp_action, Rng& rng) {
  bool my_first = my_side_first(state, my_action, opp_action, rng);

  Side first_side = my_first ? Side::Me : Side::Opp;
  Side second_side = my_first ? Side::Opp : Side::Me;
  ActionId first_action = my_first ? my_action : opp_action;
  ActionId second_action = my_first ? opp_action : my_action;

  apply_action(state, first_side, first_action, rng);

  bool second_active_fainted = (second_side == Side::Me)
      ? state.my_team[state.my_active_slot].fainted
      : state.opp_team[state.opp_active_slot].fainted;
  // A fainted mon can't execute a move (pre-empted by the first mover's
  // faster/higher-priority KO) - but it CAN still be the switch target of
  // its own side's pending switch action, so only a move is skipped here.
  if (is_switch_action(second_action) || !second_active_fainted) {
    apply_action(state, second_side, second_action, rng);
  }

  bool my_fainted = state.my_team[state.my_active_slot].fainted;
  bool opp_fainted = state.opp_team[state.opp_active_slot].fainted;

  // kTerminal is only ever decided from my_team (always fully revealed) -
  // see this header's own doc comment on why checking opp_team the same
  // way would silently overclaim knowledge Tier 1's opponent model
  // doesn't have (a fully-fainted REVEALED opp_team doesn't mean the
  // opponent's real team is empty).
  if (my_fainted) {
    bool my_team_wiped = true;
    for (const PokemonSlot& slot : state.my_team) {
      if (!slot.fainted) { my_team_wiped = false; break; }
    }
    if (my_team_wiped) return TurnResolution::kTerminal;
  }

  if (my_fainted && opp_fainted) return TurnResolution::kBothFainted;
  if (my_fainted) return TurnResolution::kMyFainted;
  if (opp_fainted) return TurnResolution::kOppFainted;
  return TurnResolution::kContinue;
}

}  // namespace be
