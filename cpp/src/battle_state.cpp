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
  // TODO(you): implement per battle_state.hpp's doc comment on is_valid().
  // Suggested shape: a small helper lambda/function that checks one
  // PokemonSlot's own invariants (hp_fraction in [0,1], boost_spe in
  // [-6,6], fainted implies hp_fraction == 0), called for every slot in
  // both my_team and opp_team (only slots with revealed == true need to
  // satisfy the hp_fraction/boost_spe/fainted checks - an unrevealed slot
  // is still at its default-constructed values and shouldn't fail
  // validation just for being unrevealed). Then separately check
  // my_active_slot/opp_active_slot: -1 is always fine, otherwise the index
  // must be in [0, 6) AND that slot must be revealed and not fainted.
  //
  // tests/test_battle_state.cpp has real fixtures for both the valid and
  // each-invariant-violated cases - run `ctest --test-dir cpp/build` (after
  // rebuilding) to check your implementation against them.
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
