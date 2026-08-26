// M4: a battle-state representation with just enough derived data to port
// evaluation.py's hand-crafted eval - HP fractions, fainted, status, types,
// boosts/speed, hazards. M4b (below) extends this with everything
// encoding.py's encode() additionally needs - species/item/ability/
// protect_counter, the remaining 6 boost stats, a movedex-derived move
// summary, state-level weather/terrain, and full 8-token hazard richness -
// see plans/precious-crafting-bachman.md's M4/M4b sections for the scope
// rationale, and encode_native()'s own doc comment below for the M4b port.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "be/pokedex_table.hpp"  // kNumTypes - sizes MoveSummary::move_types
#include "be/types.hpp"

namespace be {

// M4b: max non-active team members encode()'s "my bench" ever encodes -
// mirrors encoding.py's MAX_BENCH exactly (always 5 for a real 6-Pokemon
// team minus the active one; padding only matters for a team smaller than
// 6, e.g. a test fixture).
inline constexpr int kMaxBench = 5;

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

  // M4b: real turn numbers for encode()'s single-most-recent-hazard view
  // (encoding.py's _poke_env_hazards) - -1 means "not active." Verified
  // against poke-env's real STACKABLE_CONDITIONS (side_condition.py): only
  // Spikes/Toxic Spikes (above) are stack-counted: every other side
  // condition, including stealth_rock/sticky_web above (which
  // default_eval only needs as booleans), is turn-tracked. This struct
  // deliberately stores ALL 8 tokens' real state - encode_native() derives
  // the single-most-recent view from it on demand (see that function's own
  // doc comment for the exact algorithm, ported from _poke_env_hazards),
  // never a pre-reduced copy - matches this file's "one ordering stored,
  // views computed on demand" invariant (see BattleState's own comment).
  int stealth_rock_turn = -1;
  int sticky_web_turn = -1;
  int reflect_turn = -1;
  int light_screen_turn = -1;
  int aurora_veil_turn = -1;
  int tailwind_turn = -1;
};

// M4b: a Pokemon's known moveset summarized into the hand-engineered
// features encode() actually uses (see encoding.py's module docstring, the
// _MOVES_DEX comment block, for why move IDENTITY isn't encoded directly -
// too many distinct values for a hand-built vector - and what each of
// these means mechanically). Computed ONCE by battle_engine/mcts_player.py
// at translation time (reusing encoding.py's own already-verified
// _move_summary_features directly, NOT re-derived here against
// movedex_table.hpp - that table's MovedexEntry has no heal/sideCondition/
// selfSwitch/boosts/target flags, and movedex_table.hpp/its generator are
// out of this phase's file scope to extend) and stored as plain data -
// PokemonSlot's own precedent (spe_stat, base_stats) for "a derived value,
// computed once, stored rather than recomputed." Stays correct through
// Tier 1's whole forward-model simulation: nothing in forward_model.cpp
// mutates a Pokemon's moveset/item/ability, so a value fixed at
// translation time remains valid at every simulated search-tree node.
struct MoveSummary {
  bool has_recovery = false;
  bool has_hazard_setup = false;
  bool has_hazard_removal = false;
  bool has_setup_boost = false;
  bool has_pivot = false;
  bool has_priority = false;
  int max_base_power = 0;
  // Multi-hot over the 18 real gen-9 types, indexed by be::Type's own
  // declared value (types.hpp) - NOT encoding.py's _ALL_TYPES order
  // (poke-env's PokemonType, alphabetical, 20 members). encode_native()
  // applies the one shared be::Type -> _ALL_TYPES permutation to this (and
  // to a mon's own type1/type2) when building the output vector - kept in
  // one place rather than duplicated per reader.
  std::array<bool, kNumTypes> move_types{};
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

  // M5: this slot's known moveset, in reveal order, as movedex_table.hpp
  // lookup_movedex() keys (e.g. "stealthrock") - "" (empty string) is the
  // "not yet known" sentinel, same convention as `revealed` itself. For
  // my_team this should always be all 4 real slots filled in by the time
  // team preview ends (own team, no hidden information); for opp_team,
  // only moves actually seen used so far are known - action.hpp's
  // legal_actions() restricts Side::Opp's move actions to the non-empty
  // entries here, the same "revealed-only" Tier 1 limitation named for
  // opponent switches in plans/precious-crafting-bachman.md's Scope
  // decision, applied to moves for the same reason (no ground-truth
  // answer for what's still hidden).
  std::array<std::string, 4> moves{};

  // M4b additions - everything encode_native() needs beyond the above.
  // "" is the not-yet-revealed/unknown sentinel throughout, matching
  // `moves`' own existing convention.
  std::string species;   // base_species identity (NOT species/name - see
                          // this header's own comment on why, at the field
                          // that consumes it: encode_native()'s
                          // species-sorted bench view)
  std::string item;      // "" = no item held OR not yet revealed
  std::string ability;   // "" = no ability OR not yet revealed
  int protect_counter = 0;
  // The remaining 6 of encode()'s 7 boost dimensions (boost_spe above is
  // the pre-existing M4 field, kept as-is - not renamed, so no existing
  // reader of it needs to change).
  int8_t boost_atk = 0;
  int8_t boost_def = 0;
  int8_t boost_spa = 0;
  int8_t boost_spd = 0;
  int8_t boost_accuracy = 0;
  int8_t boost_evasion = 0;
  MoveSummary move_summary{};
};

// M4b: state-level (not per-side - weather/terrain affect both players
// equally in real Showdown) single-valued fields, mirroring encoding.py's
// own single-valued weather/terrain semantics (_WEATHER_NAMES/
// _TERRAIN_NAMES, each with real vocabulary that already excludes anything
// but a real weather/terrain - non-terrain Field entries like Trick Room/
// Gravity are out of scope here too, same as encoding.py's own
// _poke_env_terrain). Order matches _WEATHER_NAMES/_TERRAIN_NAMES exactly -
// encode_native()'s one-hot output depends on it.
enum class Weather : uint8_t { None, Sandstorm, RainDance, SunnyDay, Snow };
enum class Terrain : uint8_t { None, Electric, Grassy, Misty, Psychic };

// team-preview-order arrays: my_team[i] / opp_team[i] correspond to
// ActionId's fixed switch targets 0-5 (see plans/precious-crafting-bachman.md's
// "Fixed, state-independent action scheme" - action.hpp lands at M5 and
// reuses this same indexing, doesn't renumber). This is the ONE ordering
// stored here. The species-alphabetical bench view M4b's encode() port
// needs is computed on demand from this array by encode_native() itself
// (battle_state.cpp), never stored as a second ordering - don't add one.
struct BattleState {
  std::array<PokemonSlot, 6> my_team{};
  std::array<PokemonSlot, 6> opp_team{};
  TeamSlot my_active_slot = -1;   // index into my_team, or -1 if none active
  TeamSlot opp_active_slot = -1;  // index into opp_team, or -1 if none active
  SideConditions my_hazards{};
  SideConditions opp_hazards{};
  Weather weather = Weather::None;
  Terrain terrain = Terrain::None;
  // Found 2026-08-25 debugging a real multi-hour benchmark stall: "my"
  // side's active Pokemon can be alive and healthy (a real, valid
  // my_active_slot) while STILL having zero legal moves this turn -
  // poke-env's own battle.force_switch fires whenever a pivot move
  // (U-turn/Volt Switch/Baton Pass/...) just resolved and only a
  // replacement switch is a legal response, common in real gen9ou play.
  // Before this field existed, my_active_slot being a real (non-fainted)
  // index was the ONLY signal legal_actions() (action.hpp) used to decide
  // whether moves were offered - collapsing "there's a well-defined
  // active Pokemon" and "moves are a legal action category this turn"
  // into one bit lost the pivot case, offering real-illegal MOVE actions
  // from a Pokemon that was mid-switch, not mid-decision. Setting
  // my_active_slot itself to -1 for this case (the first fix attempted)
  // is WRONG for a different reason: legal_actions()'s switch-exclusion
  // ("can't switch into the slot that's already active") reads
  // my_active_slot too, so wiping it also wipes that exclusion, making
  // "switch into yourself" look like a legal target. Hence a dedicated
  // field instead of overloading my_active_slot again.
  //
  // "My"-side-only (no opp_force_switch): poke-env's Battle object has no
  // equivalent visibility into whether the OPPONENT is mid-pivot, and
  // this project's own forward model (forward_model.cpp) doesn't
  // simulate pivot moves at all, so no internally-generated state ever
  // needs to set this true for either side - it is populated ONLY by
  // battle_engine/mcts_player.py's translator, from the real root
  // battle.force_switch, and ONLY for "my" side. mcts.cpp's search()/
  // search_puct() explicitly clear it back to false on the per-simulation
  // state copy right after using the root's own value once (see their own
  // comments) - it must never leak into a simulated DESCENDANT node's own
  // legal_actions() computation, since the forward model has no way to
  // know whether a hypothetical future turn would also be pivot-forced.
  bool my_force_switch = false;
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
//
// M4b note: deliberately NOT extended to validate the 6 new boost_* fields
// or MoveSummary - that's not part of this milestone's Produces contract,
// and cpp/tests/test_battle_state.cpp (which would need a new test for
// such an extension) is out of this milestone's file scope. Every new
// field's default already satisfies the existing checks, so this is a
// no-op gap, not a silently-introduced one.
bool is_valid(const BattleState& state);

// M4b: exact length of encode_native()'s output - must equal both
// encoding.VECTOR_LEN (Python) and data/cpp_weights/ppo.bin's header
// vector_len field (tests/test_native_encoding.py checks both; see that
// file for the 3-way cross-check this constant exists to make possible).
// Computed from the same named pieces encoding.py's own VECTOR_LEN is
// (not a bare literal - see this project's "never invent a magic number"
// standard), so a future encode() change and a future encode_native()
// change can each be checked against the same breakdown by eye.
inline constexpr int kEncodeNumStatuses = 6;      // BRN,FRZ,PAR,PSN,SLP,TOX (excludes None/Fnt)
inline constexpr int kEncodeNumAllTypes = 20;     // poke-env's PokemonType width (18 real + ???  + Stellar) - NOT be::Type's 18, see kTypeToAllTypesIndex in battle_state.cpp
inline constexpr int kEncodeNumBoosts = 7;        // atk,def,spa,spd,spe,accuracy,evasion
inline constexpr int kEncodeNumBaseStats = 6;     // hp,atk,def,spa,spd,spe
inline constexpr int kEncodeItemVocabSize = 20;   // must match kItemVocab's length in battle_state.cpp
inline constexpr int kEncodeNumHazardTokens = 8;
inline constexpr int kEncodeNumWeatherTokens = 4;
inline constexpr int kEncodeNumTerrainTokens = 4;

inline constexpr int kPokemonVecLen =
    1                              // known
    + 1                            // hp_fraction
    + 1                            // fainted
    + kEncodeNumStatuses
    + kEncodeNumAllTypes           // own types multi-hot
    + kEncodeNumBoosts
    + kEncodeNumBaseStats
    + (kEncodeItemVocabSize + 1)   // item one-hot + "other known item" bucket
    + 5                            // has_recovery/has_hazard_setup/has_hazard_removal/has_setup_boost/has_pivot
    + 1                            // has_priority
    + 1                            // max_base_power (normalized)
    + kEncodeNumAllTypes           // move type coverage
    + 1;                           // protect_counter (normalized)

inline constexpr int kEncodeVectorLen =
    kPokemonVecLen * (1 + kMaxBench + 1)  // my active, my bench, opponent active
    + 1                                    // opponent fraction remaining
    + 2 * kEncodeNumHazardTokens
    + kEncodeNumWeatherTokens
    + kEncodeNumTerrainTokens
    + 1                                    // active-vs-active type matchup score
    + 2;                                   // my_active_hazard_immune, opp_active_hazard_immune

// M4b: bit-for-bit C++ port of battle_engine/encoding.py's encode() (read
// that module's docstring in full before touching this function - every
// named simplification there - hazard single-most-recent semantics,
// species-sorted bench view, move-summary derivation, opponent-bench
// omission, type-immunity-ability handling, ... - is preserved here
// exactly, not reinterpreted). Operates purely on `state`: the
// species/item/ability/move_summary fields PokemonSlot now carries are
// computed ONCE by battle_engine/mcts_player.py's translator (reusing
// encoding.py's own verified helpers directly - see this milestone's
// Decision Log entry for why, not re-derived against a less-complete C++
// movedex) and stay correct through Tier 1's whole forward-model
// simulation (moves/items/abilities never change turn-to-turn in this
// project's current mechanics scope), so this function needs no
// additional inputs beyond `state`.
//
// Throws std::invalid_argument if either side has no active Pokemon
// (my_active_slot/opp_active_slot == -1) - mirrors encoding.py's
// battle_view_from_poke_env's ValueError for the identical team-preview
// edge case (pybind11 auto-translates std::invalid_argument to a real
// Python ValueError - verified directly, a closer match here than
// PolicyWeights::load's std::runtime_error/RuntimeError precedent).
// Phase 5's PUCT search never calls this on such a state (see
// mcts.hpp's kForcedSwitch handling, which continues expansion past a
// missing-active-mon node rather than evaluating it) - this guard exists
// for this function's own correctness at its own boundary, not because a
// caller depends on catching it.
std::vector<float> encode_native(const BattleState& state);

// M6b: species-sorted non-active BENCH POSITIONS for state.my_team, as
// team-preview slot INDICES (0-5), not PokemonSlot pointers -
// encode_native()'s own file-local species_sorted_bench() (battle_state.cpp)
// returns pointers, which is sufficient for building the observation vector
// but can't recover which team-preview slot (== ActionId) a sorted position
// came from. action.hpp's Metamon-mapping functions need exactly that
// recovery, so this is exposed here rather than re-deriving the same sort a
// second time in action.cpp - a real divergence risk this project already
// treats as a corruption vector (see this file's own hazard/species-sort
// commentary elsewhere), not a hypothetical concern. Same sort as
// species_sorted_bench() by construction (that function is implemented in
// terms of this one, in battle_state.cpp) - base_species order, active
// excluded, fainted members still counted as a real position (they still
// occupy a real bench "position" the PPO actor's own distribution was
// trained against). -1 at a position means "no real bench member there"
// (fewer than kMaxBench non-active revealed slots - a bench<5 team; every
// real Showdown team has exactly 6, so this only reachably fires against a
// hand-built test fixture, never a live battle).
//
// Always operates on state.my_team/my_active_slot, never opp_team - see
// mcts.hpp's own doc comment on why the OPPONENT's own prior is computed by
// passing a mirror()-ed BattleState through this exact same "my side" logic
// rather than this function taking a Side parameter (which would also force
// this header to depend on action.hpp's Side enum - action.hpp depends on
// battle_state.hpp, not the reverse, and this design keeps it that way).
std::array<int, kMaxBench> species_sorted_bench_slots(const BattleState& state);

// M6b: swaps my_*/opp_* identity throughout `state` - my_team<->opp_team,
// my_active_slot<->opp_active_slot, my_hazards<->opp_hazards. weather/
// terrain are state-level (affect both sides equally in real Showdown, per
// this header's own Weather/Terrain comment above) and stay unchanged.
//
// Used to compute the OPPONENT's own PUCT prior/value through the exact
// same encode_native()/actor-forward-pass path used for "my" side (see
// mcts.hpp's own doc comment on why this mirrored-state approximation is
// the chosen, named, accepted M6b design - not a separate mechanism, and
// not a silent one: the mirrored "my bench" is the opponent's actual
// revealed-only bench, not full information, the same Tier-1 asymmetry
// legal_actions() already accepts for opponent switches/moves).
//
// Involutive: mirror(mirror(state)) == state, field-for-field - verified by
// cpp/tests/test_battle_state.cpp's own round-trip test. No BattleState
// operator== exists (nor is one added here - not needed by any caller
// outside that one test, which compares fields directly instead).
BattleState mirror(const BattleState& state);

}  // namespace be
