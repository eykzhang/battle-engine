#include "be/battle_state.hpp"

#include <stdexcept>

namespace be {

bool slot_invariants_check(const PokemonSlot& slot) {
  if (slot.hp_fraction > 1 || slot.hp_fraction < 0) return false;
  if (slot.boost_spe > 6 || slot.boost_spe < -6) return false;
  if (slot.fainted && slot.hp_fraction != 0) return false;
  return true;
}

bool is_valid(const BattleState& state) {
  if (state.my_active_slot < -1 || state.my_active_slot > 5) return false;
  if (state.opp_active_slot < -1 || state.opp_active_slot > 5) return false;
  for (const PokemonSlot& slot : state.my_team) {
    if (slot.revealed && !slot_invariants_check(slot)) return false;
  }
  for (const PokemonSlot& slot : state.opp_team) {
    if (slot.revealed && !slot_invariants_check(slot)) return false;
  }
  if (state.my_active_slot > -1 && (state.my_team[state.my_active_slot].fainted || !state.my_team[state.my_active_slot].revealed)) return false;
  if (state.opp_active_slot > -1 && (state.opp_team[state.opp_active_slot].fainted || !state.opp_team[state.opp_active_slot].revealed)) return false;
  return true;
}

}  // namespace be
