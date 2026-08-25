#include "be/battle_state.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "be/pokedex_table.hpp"

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

// ---------------------------------------------------------------------------
// M4b: encode_native() - see battle_state.hpp's doc comment for the
// contract. Everything below is a bit-for-bit port of a specific piece of
// battle_engine/encoding.py; each helper names which one.
// ---------------------------------------------------------------------------

namespace {

// Position of each be::Type value (types.hpp's declaration order, 0-17)
// within encoding.py's _ALL_TYPES - poke-env's own PokemonType enum,
// ALPHABETICAL by name, 20 entries (18 real types + THREE_QUESTION_MARKS +
// STELLAR). Verified directly against a real `list(PokemonType)` dump, not
// assumed. Positions 18/19 (???, Stellar) have no be::Type counterpart and
// are structurally unreachable from this table - they stay 0 in the output
// vector, matching real gameplay (no live Tier-1 state ever has a ??? or
// Stellar-typed active mon; Terastallization isn't modeled this phase).
constexpr int kTypeToAllTypesIndex[kNumTypes] = {
    12,  // Normal
    6,   // Fire
    17,  // Water
    3,   // Electric
    9,   // Grass
    11,  // Ice
    5,   // Fighting
    13,  // Poison
    10,  // Ground
    7,   // Flying
    14,  // Psychic
    0,   // Bug
    15,  // Rock
    8,   // Ghost
    2,   // Dragon
    1,   // Dark
    16,  // Steel
    4,   // Fairy
};

// Position of each be::Status value (types.hpp's declaration order, 0-7:
// None,Brn,Frz,Par,Psn,Tox,Slp,Fnt) within encoding.py's _STATUSES list
// (BRN,FRZ,PAR,PSN,SLP,TOX - note Slp/Tox are swapped relative to
// be::Status's own order). -1 for None and Fnt: both correctly produce "no
// status bit set," reproducing encode()'s own `status if status != FNT
// else None` override without a separate conditional (an unrevealed OR a
// fainted mon's status one-hot block is all zero either way).
constexpr int kStatusToStatusesIndex[8] = {
    -1,  // None
    0,   // Brn
    1,   // Frz
    2,   // Par
    3,   // Psn
    5,   // Tox
    4,   // Slp
    -1,  // Fnt
};

// encoding.py's _ITEM_VOCAB, verbatim (order matters - it's the one-hot
// index).
const std::array<std::string, 20> kItemVocab = {
    "leftovers", "heavydutyboots", "rockyhelmet", "boosterenergy", "lifeorb",
    "choiceband", "choicescarf", "choicespecs", "assaultvest", "airballoon",
    "focussash", "wellspringmask", "toxicorb", "loadeddice", "lightclay",
    "heatrock", "weaknesspolicy", "damprock", "eviolite", "blacksludge",
};

constexpr float kBaseStatScale = 255.0f;      // encoding.py's _BASE_STAT_SCALE
constexpr float kMaxBasePowerScale = 250.0f;  // encoding.py's _MAX_BASE_POWER_SCALE
constexpr float kProtectCounterScale = 5.0f;  // encoding.py's _PROTECT_COUNTER_SCALE

// encoding.py's _TYPE_IMMUNITY_ABILITIES, verbatim.
const std::unordered_map<std::string, Type> kTypeImmunityAbilities = {
    {"levitate", Type::Ground},   {"waterabsorb", Type::Water},
    {"stormdrain", Type::Water},  {"dryskin", Type::Water},
    {"voltabsorb", Type::Electric}, {"lightningrod", Type::Electric},
    {"motordrive", Type::Electric}, {"sapsipper", Type::Grass},
    {"flashfire", Type::Fire},    {"eartheater", Type::Ground},
    {"wellbakedbody", Type::Fire},
};

float single_type_multiplier(Type defender, Type attacker) {
  if (defender == Type::None) return 1.0f;  // no second type -> no-op factor
  return kTypeChart[static_cast<int>(defender)][static_cast<int>(attacker)];
}

// Ports encoding.py's _type_multiplier: raw type-chart product for
// `attacking` against a (d1, d2) defender, corrected for a known
// type-immunity ability or Wonder Guard (blocks anything that isn't
// already super-effective).
float type_multiplier(Type attacking, Type d1, Type d2, const std::string& defending_ability) {
  if (defending_ability == "wonderguard") {
    float raw = single_type_multiplier(d1, attacking) * single_type_multiplier(d2, attacking);
    return raw > 1.0f ? raw : 0.0f;
  }
  auto it = kTypeImmunityAbilities.find(defending_ability);
  if (it != kTypeImmunityAbilities.end() && it->second == attacking) return 0.0f;
  return single_type_multiplier(d1, attacking) * single_type_multiplier(d2, attacking);
}

float best_type_multiplier(Type t1, Type t2_maybe_none, Type target1, Type target2,
                            const std::string& target_ability) {
  float best = type_multiplier(t1, target1, target2, target_ability);
  if (t2_maybe_none != Type::None) {
    best = std::max(best, type_multiplier(t2_maybe_none, target1, target2, target_ability));
  }
  return best;
}

// Ports encoding.py's _active_matchup_score: offense - defense, each the
// best multiplier available across a dual typing, each side's known
// ability folded in via type_multiplier above.
float active_matchup_score(const PokemonSlot& my_active, const PokemonSlot& opp_active) {
  if (my_active.type1 == Type::None || opp_active.type1 == Type::None) return 0.0f;
  float offense = best_type_multiplier(my_active.type1, my_active.type2, opp_active.type1,
                                        opp_active.type2, opp_active.ability);
  float defense = best_type_multiplier(opp_active.type1, opp_active.type2, my_active.type1,
                                        my_active.type2, my_active.ability);
  return offense - defense;
}

// Ports encoding.py's _is_hazard_immune - see that function's own doc
// comment for the full mechanical accounting (Heavy-Duty Boots, Flying,
// Levitate; what's deliberately still out of scope).
bool is_hazard_immune(const PokemonSlot& mon) {
  if (mon.item == "heavydutyboots") return true;
  if (mon.type1 == Type::None) return false;  // types unknown - never a false immunity signal
  if (mon.type1 == Type::Flying || mon.type2 == Type::Flying) return true;
  return mon.ability == "levitate";
}

// Ports encoding.py's _poke_env_hazards: the single most-recently-set
// hazard/screen token (index into the fixed 8-token order stealthrock=0,
// spikes=1, toxicspikes=2, stickyweb=3, reflect=4, lightscreen=5,
// auroraveil=6, tailwind=7 - matches _HAZARD_TOKENS exactly), or nullopt if
// none active. Turn-tracked tokens are ranked by turn number, strictly
// greater than the current best - reproducing Python max()'s first-wins
// tie-break by NOT replacing on an equal turn, iterating in _HAZARD_TOKENS'
// own relative order (stealthrock, stickyweb, reflect, lightscreen,
// auroraveil, tailwind - spikes/toxicspikes excluded, they're
// stack-tracked). Falls back to a stack-tracked token (spikes checked
// before toxicspikes, same fixed order) only when no turn-tracked token is
// active.
std::optional<int> most_recent_hazard_index(const SideConditions& sc) {
  struct Entry { int index; int turn; };
  const Entry turn_tracked[] = {
      {0, sc.stealth_rock_turn}, {3, sc.sticky_web_turn},  {4, sc.reflect_turn},
      {5, sc.light_screen_turn}, {6, sc.aurora_veil_turn}, {7, sc.tailwind_turn},
  };
  int best_index = -1;
  int best_turn = -1;
  for (const Entry& e : turn_tracked) {
    if (e.turn > best_turn) {
      best_turn = e.turn;
      best_index = e.index;
    }
  }
  if (best_index >= 0) return best_index;
  if (sc.spikes_layers > 0) return 1;
  if (sc.toxic_spikes_layers > 0) return 2;
  return std::nullopt;
}

std::array<float, kEncodeNumHazardTokens> hazard_vector(const SideConditions& sc) {
  std::array<float, kEncodeNumHazardTokens> vec{};
  std::optional<int> idx = most_recent_hazard_index(sc);
  if (idx) vec[*idx] = 1.0f;
  return vec;
}

std::array<float, kEncodeNumWeatherTokens> weather_vector(Weather w) {
  std::array<float, kEncodeNumWeatherTokens> vec{};
  switch (w) {
    case Weather::Sandstorm: vec[0] = 1.0f; break;
    case Weather::RainDance: vec[1] = 1.0f; break;
    case Weather::SunnyDay:  vec[2] = 1.0f; break;
    case Weather::Snow:      vec[3] = 1.0f; break;
    case Weather::None:      break;
  }
  return vec;
}

std::array<float, kEncodeNumTerrainTokens> terrain_vector(Terrain t) {
  std::array<float, kEncodeNumTerrainTokens> vec{};
  switch (t) {
    case Terrain::Electric: vec[0] = 1.0f; break;
    case Terrain::Grassy:   vec[1] = 1.0f; break;
    case Terrain::Misty:    vec[2] = 1.0f; break;
    case Terrain::Psychic:  vec[3] = 1.0f; break;
    case Terrain::None:     break;
  }
  return vec;
}

std::vector<float> item_vector(const std::string& item) {
  std::vector<float> vec(kItemVocab.size() + 1, 0.0f);
  if (item.empty()) return vec;
  auto it = std::find(kItemVocab.begin(), kItemVocab.end(), item);
  if (it != kItemVocab.end()) {
    vec[static_cast<size_t>(std::distance(kItemVocab.begin(), it))] = 1.0f;
  } else {
    vec.back() = 1.0f;  // known item, outside the curated vocab
  }
  return vec;
}

// Ports encoding.py's _move_summary_vector: 6 flags (including
// has_priority) + max_base_power (normalized) + move-type coverage
// multi-hot, in that exact order.
std::vector<float> move_summary_vector(const MoveSummary& ms) {
  std::vector<float> vec;
  vec.reserve(6 + 1 + kEncodeNumAllTypes);
  vec.push_back(ms.has_recovery ? 1.0f : 0.0f);
  vec.push_back(ms.has_hazard_setup ? 1.0f : 0.0f);
  vec.push_back(ms.has_hazard_removal ? 1.0f : 0.0f);
  vec.push_back(ms.has_setup_boost ? 1.0f : 0.0f);
  vec.push_back(ms.has_pivot ? 1.0f : 0.0f);
  vec.push_back(ms.has_priority ? 1.0f : 0.0f);
  vec.push_back(static_cast<float>(ms.max_base_power) / kMaxBasePowerScale);
  std::array<float, kEncodeNumAllTypes> coverage{};
  for (int t = 0; t < kNumTypes; ++t) {
    if (ms.move_types[static_cast<size_t>(t)]) coverage[static_cast<size_t>(kTypeToAllTypesIndex[t])] = 1.0f;
  }
  vec.insert(vec.end(), coverage.begin(), coverage.end());
  return vec;
}

// Ports encoding.py's _encode_pokemon. An unrevealed slot short-circuits to
// an all-zero block matching PokemonView.unknown()'s exact sentinel -
// deliberately NOT built by reading a default-constructed PokemonSlot's
// fields (its hp_fraction defaults to 1.0, chosen for default_eval, which
// never reads an unrevealed slot's hp_fraction at all - reusing that
// default here would silently produce a non-zero "unknown" block).
std::vector<float> encode_pokemon_slot(const PokemonSlot& slot) {
  if (!slot.revealed) return std::vector<float>(static_cast<size_t>(kPokemonVecLen), 0.0f);

  std::vector<float> vec;
  vec.reserve(static_cast<size_t>(kPokemonVecLen));

  vec.push_back(1.0f);  // known - guaranteed true, see the !revealed early-return above
  vec.push_back(slot.hp_fraction);
  vec.push_back(slot.fainted ? 1.0f : 0.0f);

  std::array<float, kEncodeNumStatuses> status_vec{};
  int status_idx = kStatusToStatusesIndex[static_cast<int>(slot.status)];
  if (status_idx >= 0) status_vec[static_cast<size_t>(status_idx)] = 1.0f;
  vec.insert(vec.end(), status_vec.begin(), status_vec.end());

  std::array<float, kEncodeNumAllTypes> types_vec{};
  if (slot.type1 != Type::None) types_vec[static_cast<size_t>(kTypeToAllTypesIndex[static_cast<int>(slot.type1)])] = 1.0f;
  if (slot.type2 != Type::None) types_vec[static_cast<size_t>(kTypeToAllTypesIndex[static_cast<int>(slot.type2)])] = 1.0f;
  vec.insert(vec.end(), types_vec.begin(), types_vec.end());

  // Boosts, in encoding.py's _BOOST_NAMES order (atk,def,spa,spd,spe,
  // accuracy,evasion) - NOT this struct's own declaration order.
  const int8_t boosts[kEncodeNumBoosts] = {
      slot.boost_atk, slot.boost_def, slot.boost_spa, slot.boost_spd,
      slot.boost_spe, slot.boost_accuracy, slot.boost_evasion,
  };
  for (int8_t b : boosts) vec.push_back(static_cast<float>(b) / 6.0f);

  // Base stats, in encoding.py's _STAT_NAMES order (hp,atk,def,spa,spd,spe)
  // - matches StatBlock's own field order already.
  const int stats[kEncodeNumBaseStats] = {
      slot.base_stats.hp, slot.base_stats.atk, slot.base_stats.def,
      slot.base_stats.spa, slot.base_stats.spd, slot.base_stats.spe,
  };
  for (int s : stats) vec.push_back(static_cast<float>(s) / kBaseStatScale);

  std::vector<float> item_vec = item_vector(slot.item);
  vec.insert(vec.end(), item_vec.begin(), item_vec.end());

  std::vector<float> move_vec = move_summary_vector(slot.move_summary);
  vec.insert(vec.end(), move_vec.begin(), move_vec.end());

  float clamped_protect = std::min(static_cast<float>(slot.protect_counter), kProtectCounterScale);
  vec.push_back(clamped_protect / kProtectCounterScale);

  assert(vec.size() == static_cast<size_t>(kPokemonVecLen));
  return vec;
}

// Ports encoding.py's battle_view_from_poke_env's bench construction
// (sorted by base_species, active excluded) + _pad_bench: my_team's
// species-alphabetical non-active revealed members, padded to kMaxBench
// with an unrevealed sentinel slot. Opponent's bench is never encoded at
// all (matches encode()'s own asymmetry - opp gets only opp_active +
// opp_remaining_fraction), so this only ever runs over my_team.
std::array<const PokemonSlot*, kMaxBench> species_sorted_bench(const BattleState& state) {
  std::vector<const PokemonSlot*> bench;
  bench.reserve(6);
  for (int i = 0; i < 6; ++i) {
    if (i == state.my_active_slot) continue;
    if (!state.my_team[static_cast<size_t>(i)].revealed) continue;
    bench.push_back(&state.my_team[static_cast<size_t>(i)]);
  }
  std::sort(bench.begin(), bench.end(), [](const PokemonSlot* a, const PokemonSlot* b) {
    return a->species < b->species;
  });

  static const PokemonSlot kUnknownSlot{};  // revealed=false -> encode_pokemon_slot's all-zero path
  std::array<const PokemonSlot*, kMaxBench> result{};
  for (int i = 0; i < kMaxBench; ++i) {
    result[static_cast<size_t>(i)] =
        (i < static_cast<int>(bench.size())) ? bench[static_cast<size_t>(i)] : &kUnknownSlot;
  }
  return result;
}

}  // namespace

std::vector<float> encode_native(const BattleState& state) {
  if (state.my_active_slot < 0 || state.opp_active_slot < 0) {
    throw std::invalid_argument(
        "encode_native requires both active Pokemon to be chosen (not team "
        "preview) - no active_pokemon means there's no well-defined 'bench' "
        "to encode yet (mirrors encoding.py's battle_view_from_poke_env)");
  }

  const PokemonSlot& my_active = state.my_team[static_cast<size_t>(state.my_active_slot)];
  const PokemonSlot& opp_active = state.opp_team[static_cast<size_t>(state.opp_active_slot)];

  std::vector<float> vec;
  vec.reserve(static_cast<size_t>(kEncodeVectorLen));
  auto append = [&vec](const std::vector<float>& part) { vec.insert(vec.end(), part.begin(), part.end()); };
  auto append_arr = [&vec](const auto& part) { vec.insert(vec.end(), part.begin(), part.end()); };

  append(encode_pokemon_slot(my_active));
  for (const PokemonSlot* slot : species_sorted_bench(state)) append(encode_pokemon_slot(*slot));
  append(encode_pokemon_slot(opp_active));

  int opp_fainted = 0;
  for (const PokemonSlot& slot : state.opp_team) {
    if (slot.fainted) opp_fainted++;
  }
  vec.push_back(static_cast<float>(6 - opp_fainted) / 6.0f);

  append_arr(hazard_vector(state.my_hazards));
  append_arr(hazard_vector(state.opp_hazards));
  append_arr(weather_vector(state.weather));
  append_arr(terrain_vector(state.terrain));

  vec.push_back(active_matchup_score(my_active, opp_active));
  vec.push_back(is_hazard_immune(my_active) ? 1.0f : 0.0f);
  vec.push_back(is_hazard_immune(opp_active) ? 1.0f : 0.0f);

  assert(vec.size() == static_cast<size_t>(kEncodeVectorLen));
  return vec;
}

}  // namespace be
