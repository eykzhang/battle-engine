// M4: a battle-state representation with just enough derived data to port
// evaluation.py's hand-crafted eval - HP fractions, fainted, status, types,
// boosts/speed, hazards. Deliberately NOT the full 665-dim encode() vector
// (that's the deferred M4b enhancement track) - see
// plans/precious-crafting-bachman.md's M4 section for the scope rationale.
#pragma once

#include <array>

#include "be/types.hpp"

namespace be {

// Hazards evaluation.py's evaluate() actually scores (via _hazard_score) -
// Spikes/Toxic Spikes are stack counts (more layers = worse), Stealth
// Rock/Sticky Web are presence-only. Other side conditions (Reflect, Light
// Screen, Tailwind, ...) aren't read by evaluate() at all and are out of
// scope for this struct - a real, deliberate v1 scope match to the Python
// eval this milestone ports, not an oversight. Real showdown stack limits:
// Spikes 0-3, Toxic Spikes 0-2.
struct SideConditions {
  uint8_t spikes_layers = 0;
  uint8_t toxic_spikes_layers = 0;
  bool stealth_rock = false;
  bool sticky_web = false;
};

// One Pokemon slot within a fixed, 6-long, team-preview-ordered array (see
// BattleState's own comment for why that ordering is the one invariant that
// matters here). `revealed` is what lets opp_team model "we don't know this
// slot yet" - poke-env's battle.opponent_team is a dict with only revealed
// entries; a fixed std::array<PokemonSlot, 6> needs an explicit flag to
// recover that same distinction (unrevealed, NOT "revealed but full HP").
// my_team should always have all 6 slots revealed by the time team preview
// ends (it's your own team).
//
// spe_stat mirrors damage.py's estimate_stat's known-vs-estimated
// distinction for Speed specifically (the only stat evaluate() needs):
// -1 means "not known exactly" (true for the opponent's Pokemon, whose EVs/
// nature/IVs aren't revealed) and should be estimated from base_stats.spe +
// level via the same max-IV/0-EV/neutral-nature formula
// battle_engine/damage.py's estimate_stat() uses:
//   floor(floor(2*base_spe + 31) * level / 100) + 5
// A non-negative value is the real, known stat (always true for your own
// team).
struct PokemonSlot {
  bool revealed = false;
  bool fainted = false;
  int level = 100;
  float hp_fraction = 1.0f;
  Status status = Status::None;
  Type type1 = Type::None;
  Type type2 = Type::None;
  StatBlock base_stats{};
  int8_t boost_spe = 0;  // stat stage, real Showdown range is [-6, 6]
  int spe_stat = -1;     // -1 = unknown, estimate from base_stats.spe + level
};

// team-preview-order arrays: my_team[i] / opp_team[i] correspond to
// ActionId's fixed switch targets 0-5 (see plans/precious-crafting-bachman.md's
// "Fixed, state-independent action scheme" - action.hpp lands at M5 and
// reuses this same indexing, doesn't renumber). This is the ONE ordering
// stored here. Any future species-alphabetical view (needed only once M4b's
// exact encode() port exists, to match encoding.py's bench-sort convention)
// must be computed on demand from this array, never stored as a second
// ordering - don't add one.
struct BattleState {
  std::array<PokemonSlot, 6> my_team{};
  std::array<PokemonSlot, 6> opp_team{};
  TeamSlot my_active_slot = -1;   // index into my_team, or -1 if none active
  TeamSlot opp_active_slot = -1;  // index into opp_team, or -1 if none active
  SideConditions my_hazards{};
  SideConditions opp_hazards{};
};

// True iff `state` satisfies BattleState's structural invariants:
// - my_active_slot/opp_active_slot are either -1, or a valid index (0-5)
//   into a slot that is both revealed and not fainted (an inactive or
//   unrevealed slot can never be the active one)
// - every revealed slot's hp_fraction is in [0, 1]
// - every revealed slot's boost_spe is in [-6, 6] (Showdown's real stat
//   stage range)
// - a fainted slot's hp_fraction is exactly 0
// Not called anywhere yet - this milestone just establishes it truthfully,
// cheap enough to assert at every forward_model transition once M5 lands.
bool is_valid(const BattleState& state);

}  // namespace be
