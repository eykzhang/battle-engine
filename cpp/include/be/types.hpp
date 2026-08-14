// Shared POD types used across the Phase 4 C++ engine. Kept dependency-free
// (no includes beyond <cstdint>) so every other header can include this
// without pulling in anything heavier.
#pragma once

#include <cstdint>

namespace be {

// The 18 real gen-9 attacking/defending types (Stellar excluded - it's
// Terastallization-only and poke-env's own type chart doesn't include it
// either; PokemonType.damage_multiplier special-cases it to return 1.0
// unconditionally, see dump_pokedex.py's docstring). `None` is a sentinel
// for "no second type" (single-typed Pokemon), not a real 19th type.
// Ordering matches scripts/dump_pokedex.py's _TYPE_ORDER exactly - the type
// chart table is indexed by this enum's underlying value, so don't reorder
// this without regenerating pokedex_table.cpp.
enum class Type : uint8_t {
  Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground,
  Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, Fairy,
  None,
};

// Mirrors poke-env's Status enum values actually reachable in a real
// battle (battle_engine/evaluation.py checks `status not in (None, FNT)`,
// i.e. treats "no status" and "fainted" as the two non-afflicted cases -
// None here plays that same role for an un-afflicted, non-fainted mon).
enum class Status : uint8_t { None, Brn, Frz, Par, Psn, Tox, Slp, Fnt };

// A move's damage category (M5). Physical/Special use different attack/
// defense stat pairs (mirrors damage.py's _attack_defense_stats); Status
// moves deal no direct damage - Tier 1's forward model treats a Status
// move as a turn-cost-only no-op (see forward_model.hpp's module comment
// for why the move's real field effects - stat boosts, hazard setup,
// status infliction - are Tier 2, deferred).
enum class MoveCategory : uint8_t { Physical, Special, Status };

// Base stats in Showdown's own hp/atk/def/spa/spd/spe order (matches
// scripts/dump_pokedex.py's _STAT_ORDER) - a species' fixed dex values, not
// a specific Pokemon's actual computed stats (see estimate_stat's role in
// battle_engine/damage.py for how base stats + level + known/unknown EVs
// combine into a real stat estimate; PokemonSlot in battle_state.hpp is
// where that distinction lives for a specific in-battle Pokemon).
struct StatBlock {
  int hp;
  int atk;
  int def;
  int spa;
  int spd;
  int spe;
};

// team-preview-order index (matches ActionId 0-5's switch targets once
// action.hpp lands at M5) - a fixed slot 0-5 within one side's 6-Pokemon
// team, NOT a species identity. -1 is the "no active Pokemon" sentinel used
// by BattleState::my_active_slot / opp_active_slot.
using TeamSlot = int8_t;

}  // namespace be
