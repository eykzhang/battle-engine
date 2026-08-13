#include "be/eval.hpp"

#include <stdexcept>

#include "be/pokedex_table.hpp"

#include <algorithm>

#include <cmath>

namespace be {

namespace {

float single_type_multiplier(Type defender_type, Type attacker_type) {
  if (defender_type == Type::None) return 1.0f;  // no second type -> no-op factor
  return kTypeChart[static_cast<int>(defender_type)][static_cast<int>(attacker_type)];
}

}  // namespace

float type_matchup_score(const PokemonSlot& mon, const PokemonSlot& opponent) {
  Type o_1 = mon.type1;
  Type o_2 = mon.type2;
  Type d_1 = opponent.type1;
  Type d_2 = opponent.type2;
  float off = single_type_multiplier(d_1, o_1) * single_type_multiplier(d_2, o_1);
  if (o_2 != Type::None) off = std::max(off, single_type_multiplier(d_1, o_2) * single_type_multiplier(d_2, o_2));
  float def = single_type_multiplier(o_1, d_1) * single_type_multiplier(o_2, d_1);
  if (d_2 != Type::None) def = std::max(def, single_type_multiplier(o_1, d_2) * single_type_multiplier(o_2, d_2));
  return off - def;
}

float hazard_score(const SideConditions& conditions) {
  float ret = (conditions.spikes_layers * kHazardLayerWeight) + (conditions.toxic_spikes_layers * kHazardLayerWeight);
  if (conditions.stealth_rock) ret += kHazardPresenceWeight;
  if (conditions.sticky_web) ret += kHazardPresenceWeight;
  return ret;
}

float speed_control_score(const PokemonSlot& my_active, const PokemonSlot& opp_active) {
  int my_speed;
  int opp_speed;
  if (my_active.spe_stat >= 0) my_speed = my_active.spe_stat;
  else my_speed = std::floor(std::floor(2 * my_active.base_stats.spe + 31) * my_active.level / 100) + 5;
  if (opp_active.spe_stat >= 0) opp_speed = opp_active.spe_stat;
  else opp_speed = std::floor(std::floor(2 * opp_active.base_stats.spe + 31) * opp_active.level / 100) + 5;
  
  float my_boost;
  float opp_boost;
  if (my_active.boost_spe >= 0) my_boost = float(2 + my_active.boost_spe) / 2;
  else my_boost = 2 / float(2 - my_active.boost_spe);
  if (my_active.status == Status::Par) my_boost *= 0.5;
  if (opp_active.boost_spe >= 0) opp_boost = float(2 + opp_active.boost_spe) / 2;
  else opp_boost = 2 / float(2 - opp_active.boost_spe);
  if (opp_active.status == Status::Par) opp_boost *= 0.5;

  if (my_speed * my_boost == opp_speed * opp_boost) return 0.0f;
  else if (my_speed * my_boost > opp_speed * opp_boost) return 1.0f;
  else return -1.0f;
}

float evaluate(const BattleState& state) {
  float my_hp_frac = 0.0f;
  float opp_hp_frac = 0.0f;
  int my_alive = 0;
  int opp_alive = 0;
  int my_afflicted = 0;
  int opp_afflicted = 0;
  float my_hazard = hazard_score(state.my_hazards);
  float opp_hazard = hazard_score(state.opp_hazards);
  float score;

  for (const PokemonSlot& slot : state.my_team) {
    if (slot.fainted || !slot.revealed) continue;
    my_hp_frac += slot.hp_fraction;
    my_alive++;
    if (slot.status != Status::None && slot.status != Status::Fnt) my_afflicted++;
  }
  for (const PokemonSlot& slot : state.opp_team) {
    if (slot.fainted || !slot.revealed) continue;
    opp_hp_frac += slot.hp_fraction;
    opp_alive++;
    if (slot.status != Status::None && slot.status != Status::Fnt) opp_afflicted++;
  }

  score = kHpFractionWeight * (my_hp_frac - opp_hp_frac);
  score += kAliveCountWeight * (my_alive - opp_alive);
  score += kStatusWeight * (opp_afflicted - my_afflicted);
  score += opp_hazard - my_hazard;
  if (state.my_active_slot > -1 && state.opp_active_slot > -1) {
    score += kTypeMatchupWeight * type_matchup_score(state.my_team[state.my_active_slot], state.opp_team[state.opp_active_slot]);
    score += kSpeedControlWeight * speed_control_score(state.my_team[state.my_active_slot], state.opp_team[state.opp_active_slot]);
  }
  return score;
}

}  // namespace be
