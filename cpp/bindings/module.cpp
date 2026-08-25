// M1: toolchain bring-up (ping() - proves CMake + C++20 + pybind11 v3 + the
// editable-install import path work end-to-end, see
// plans/precious-crafting-bachman.md's M1). M5: enough of BattleState +
// legal_actions() exposed to Python for tests/test_native_legality.py to
// build a real BattleState from a poke-env battle
// (battle_engine/mcts_player.py's translator) and cross-check
// legal_actions() against poke-env's own available_moves/
// available_switches. This stays thin glue only, per the plan's own
// module-boundary intent - no engine logic lives in this file.
//
// PokemonSlot::moves and BattleState::my_team/opp_team are fixed-size
// std::array members - pybind11/stl.h's array caster converts them to/from
// a plain Python list, but ONLY as a whole-container copy on each
// attribute access, not a live reference. That means
// `state.my_team[0].hp_fraction = 0.5` from Python would silently mutate a
// throwaway copy and do nothing - always build a full PokemonSlot (or a
// full 6-element list of them) in Python first, then assign the WHOLE
// list back to state.my_team, never index into a container property
// in-place. battle_engine/mcts_player.py's translator follows this
// convention; tests/test_native_legality.py relies on it too.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "be/action.hpp"
#include "be/battle_state.hpp"
#include "be/mcts.hpp"
#include "be/mlp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "battle_engine native C++ extension (Phase 4)";
  m.def("ping", []() { return 42; }, "M1 toolchain smoke test - returns 42.");

  py::enum_<be::Type>(m, "Type")
      .value("NORMAL", be::Type::Normal)
      .value("FIRE", be::Type::Fire)
      .value("WATER", be::Type::Water)
      .value("ELECTRIC", be::Type::Electric)
      .value("GRASS", be::Type::Grass)
      .value("ICE", be::Type::Ice)
      .value("FIGHTING", be::Type::Fighting)
      .value("POISON", be::Type::Poison)
      .value("GROUND", be::Type::Ground)
      .value("FLYING", be::Type::Flying)
      .value("PSYCHIC", be::Type::Psychic)
      .value("BUG", be::Type::Bug)
      .value("ROCK", be::Type::Rock)
      .value("GHOST", be::Type::Ghost)
      .value("DRAGON", be::Type::Dragon)
      .value("DARK", be::Type::Dark)
      .value("STEEL", be::Type::Steel)
      .value("FAIRY", be::Type::Fairy)
      // "NONE", not "None" - `None` is a Python keyword and can't be used
      // as an enum member's attribute name (be::Type::None the C++
      // identifier is unaffected; only the exposed Python-side name changes).
      .value("NONE", be::Type::None);

  py::enum_<be::Status>(m, "Status")
      .value("NONE", be::Status::None)
      .value("BRN", be::Status::Brn)
      .value("FRZ", be::Status::Frz)
      .value("PAR", be::Status::Par)
      .value("PSN", be::Status::Psn)
      .value("TOX", be::Status::Tox)
      .value("SLP", be::Status::Slp)
      .value("FNT", be::Status::Fnt);

  py::enum_<be::Side>(m, "Side")
      .value("ME", be::Side::Me)
      .value("OPP", be::Side::Opp);

  // M4b: state-level weather/terrain - see battle_state.hpp's own comment
  // on why these are single-valued fields, not per-side.
  py::enum_<be::Weather>(m, "Weather")
      .value("NONE", be::Weather::None)
      .value("SANDSTORM", be::Weather::Sandstorm)
      .value("RAINDANCE", be::Weather::RainDance)
      .value("SUNNYDAY", be::Weather::SunnyDay)
      .value("SNOW", be::Weather::Snow);

  py::enum_<be::Terrain>(m, "Terrain")
      .value("NONE", be::Terrain::None)
      .value("ELECTRIC", be::Terrain::Electric)
      .value("GRASSY", be::Terrain::Grassy)
      .value("MISTY", be::Terrain::Misty)
      .value("PSYCHIC", be::Terrain::Psychic);

  py::class_<be::StatBlock>(m, "StatBlock")
      .def(py::init<>())
      .def_readwrite("hp", &be::StatBlock::hp)
      .def_readwrite("atk", &be::StatBlock::atk)
      // "def_", not "def" - `def` is a Python keyword and can't be used as
      // an attribute name in assignment syntax (`block.def = 5` is a
      // SyntaxError); the C++ field itself is still named `def`.
      .def_readwrite("def_", &be::StatBlock::def)
      .def_readwrite("spa", &be::StatBlock::spa)
      .def_readwrite("spd", &be::StatBlock::spd)
      .def_readwrite("spe", &be::StatBlock::spe);

  py::class_<be::SideConditions>(m, "SideConditions")
      .def(py::init<>())
      .def_readwrite("spikes_layers", &be::SideConditions::spikes_layers)
      .def_readwrite("toxic_spikes_layers", &be::SideConditions::toxic_spikes_layers)
      .def_readwrite("stealth_rock", &be::SideConditions::stealth_rock)
      .def_readwrite("sticky_web", &be::SideConditions::sticky_web)
      // M4b: real turn numbers for encode_native()'s single-most-recent-
      // hazard derivation - -1 = not active. See battle_state.hpp's own
      // comment on the turn-tracked vs. stack-tracked split.
      .def_readwrite("stealth_rock_turn", &be::SideConditions::stealth_rock_turn)
      .def_readwrite("sticky_web_turn", &be::SideConditions::sticky_web_turn)
      .def_readwrite("reflect_turn", &be::SideConditions::reflect_turn)
      .def_readwrite("light_screen_turn", &be::SideConditions::light_screen_turn)
      .def_readwrite("aurora_veil_turn", &be::SideConditions::aurora_veil_turn)
      .def_readwrite("tailwind_turn", &be::SideConditions::tailwind_turn);

  // M4b: a Pokemon's movedex-derived move summary - see battle_state.hpp's
  // own comment on why this is computed once in Python (mcts_player.py)
  // rather than re-derived in C++. move_types is a fixed
  // std::array<bool, 18> - same whole-container-only caveat as
  // PokemonSlot::moves below (assign a fresh 18-bool Python list, never
  // index into the property in place).
  py::class_<be::MoveSummary>(m, "MoveSummary")
      .def(py::init<>())
      .def_readwrite("has_recovery", &be::MoveSummary::has_recovery)
      .def_readwrite("has_hazard_setup", &be::MoveSummary::has_hazard_setup)
      .def_readwrite("has_hazard_removal", &be::MoveSummary::has_hazard_removal)
      .def_readwrite("has_setup_boost", &be::MoveSummary::has_setup_boost)
      .def_readwrite("has_pivot", &be::MoveSummary::has_pivot)
      .def_readwrite("has_priority", &be::MoveSummary::has_priority)
      .def_readwrite("max_base_power", &be::MoveSummary::max_base_power)
      .def_readwrite("move_types", &be::MoveSummary::move_types);

  py::class_<be::PokemonSlot>(m, "PokemonSlot")
      .def(py::init<>())
      .def_readwrite("revealed", &be::PokemonSlot::revealed)
      .def_readwrite("fainted", &be::PokemonSlot::fainted)
      .def_readwrite("level", &be::PokemonSlot::level)
      .def_readwrite("hp_fraction", &be::PokemonSlot::hp_fraction)
      .def_readwrite("status", &be::PokemonSlot::status)
      .def_readwrite("type1", &be::PokemonSlot::type1)
      .def_readwrite("type2", &be::PokemonSlot::type2)
      .def_readwrite("base_stats", &be::PokemonSlot::base_stats)
      .def_readwrite("boost_spe", &be::PokemonSlot::boost_spe)
      .def_readwrite("spe_stat", &be::PokemonSlot::spe_stat)
      // moves is a fixed std::array<std::string, 4> - assign a Python list
      // of EXACTLY 4 strings (use "" for an unknown/not-yet-revealed
      // slot), never fewer/more (pybind11's array caster raises on a
      // length mismatch rather than padding or truncating).
      .def_readwrite("moves", &be::PokemonSlot::moves)
      // M4b additions - see battle_state.hpp's own PokemonSlot comment.
      .def_readwrite("species", &be::PokemonSlot::species)
      .def_readwrite("item", &be::PokemonSlot::item)
      .def_readwrite("ability", &be::PokemonSlot::ability)
      .def_readwrite("protect_counter", &be::PokemonSlot::protect_counter)
      .def_readwrite("boost_atk", &be::PokemonSlot::boost_atk)
      .def_readwrite("boost_def", &be::PokemonSlot::boost_def)
      .def_readwrite("boost_spa", &be::PokemonSlot::boost_spa)
      .def_readwrite("boost_spd", &be::PokemonSlot::boost_spd)
      .def_readwrite("boost_accuracy", &be::PokemonSlot::boost_accuracy)
      .def_readwrite("boost_evasion", &be::PokemonSlot::boost_evasion)
      .def_readwrite("move_summary", &be::PokemonSlot::move_summary);

  py::class_<be::BattleState>(m, "BattleState")
      .def(py::init<>())
      // Whole-container access only - see this file's own module comment.
      .def_readwrite("my_team", &be::BattleState::my_team)
      .def_readwrite("opp_team", &be::BattleState::opp_team)
      .def_readwrite("my_active_slot", &be::BattleState::my_active_slot)
      .def_readwrite("opp_active_slot", &be::BattleState::opp_active_slot)
      .def_readwrite("my_hazards", &be::BattleState::my_hazards)
      .def_readwrite("opp_hazards", &be::BattleState::opp_hazards)
      .def_readwrite("weather", &be::BattleState::weather)
      .def_readwrite("terrain", &be::BattleState::terrain);

  m.def("is_valid", &be::is_valid, py::arg("state"));

  // M4b: bit-for-bit C++ port of encoding.py's encode() - see
  // battle_state.hpp's own doc comment for the full contract, including
  // the std::invalid_argument (-> a real Python ValueError, verified) raised when either
  // side has no active Pokemon.
  m.def("encode_native", &be::encode_native, py::arg("state"));
  m.attr("ENCODE_VECTOR_LEN") = int(be::kEncodeVectorLen);

  m.attr("NUM_SWITCH_ACTIONS") = int(be::kNumSwitchActions);
  m.attr("MOVE_ACTION_OFFSET") = int(be::kMoveActionOffset);
  m.attr("NUM_MOVE_ACTIONS") = int(be::kNumMoveActions);

  m.def("legal_actions", &be::legal_actions, py::arg("state"), py::arg("side"),
        "Legal ActionIds (see action.hpp) for `side` in `state`, using the "
        "fixed M5 action scheme - ActionId 0-5 switch, 6-9 move slot.");

  // M7: NO_ACTION mirrors mcts.hpp's kNoAction sentinel (-1) - a real
  // search() result's best_action is always >= 0, so a caller (MctsPlayer)
  // checks equality/sign against this to detect "no legal action" and fall
  // back to a safe default order rather than propagating an invalid one.
  m.attr("NO_ACTION") = int(be::kNoAction);

  py::class_<be::SearchResult>(m, "SearchResult")
      .def_readonly("best_action", &be::SearchResult::best_action)
      .def_readonly("root_visit_distribution", &be::SearchResult::root_visit_distribution);

  // M7: exposes M6's search() with default_eval fixed C++-side, NOT the raw
  // be::search signature - that takes an EvalFn (std::function<float(const
  // BattleState&)>), and pybind11/functional.h would happily accept a
  // Python callable there. A Python leaf_eval invoked while the GIL is
  // released (below) would be a genuine use-of-Python-without-the-GIL
  // crash, not a hypothetical - so this lambda's signature deliberately
  // has no callable parameter at all; default_eval is the only leaf
  // evaluator this binding can ever call. py::call_guard (not manual
  // py::gil_scoped_release RAII) releases the GIL for exactly the call's
  // duration and reacquires it before the return value is converted back
  // to Python - per mcts.hpp's own M7 latency note: choose_move runs on
  // poke-env's asyncio loop, so a long blocking call here must not stall
  // every other concurrently-running battle.
  m.def(
      "search",
      [](const be::BattleState& state, int n_simulations, uint64_t seed) {
        return be::search(state, be::default_eval, n_simulations, seed);
      },
      py::arg("state"), py::arg("n_simulations"), py::arg("seed"),
      py::call_guard<py::gil_scoped_release>(),
      "Runs M6's MCTS/DUCT search from `state` (n_simulations sims, seeded "
      "for determinism) using the fixed C++-side default_eval leaf "
      "evaluator - no Python callable leaf_eval is accepted, see this "
      "binding's own comment in module.cpp for why.");

  // M3: exposes mlp.hpp's forward pass + loader so
  // tests/test_native_forward_pass.py can call the exact same C++ code
  // path Phase 5's PUCT expansion will use, not a reimplementation of it
  // in the binding layer. No GIL release here (unlike search() above) -
  // one forward pass is a short, bounded computation, not the kind of
  // long blocking call that would stall poke-env's asyncio loop for other
  // concurrently-running battles.
  // Exposes only in_dim/out_dim, not the raw weight/bias vectors - a
  // caller (the parity test) needs to confirm the loaded shapes, never
  // needs to read the weights themselves back out from Python.
  py::class_<be::MlpLayer>(m, "MlpLayer")
      .def_readonly("in_dim", &be::MlpLayer::in_dim)
      .def_readonly("out_dim", &be::MlpLayer::out_dim);

  py::class_<be::MlpWeights>(m, "MlpWeights")
      .def_readonly("layer0", &be::MlpWeights::layer0)
      .def_readonly("layer1", &be::MlpWeights::layer1)
      .def_readonly("layer2", &be::MlpWeights::layer2)
      .def("forward", &be::MlpWeights::forward, py::arg("input"),
           "Runs the fixed 3-layer (in->128->64->out) forward pass, ReLU "
           "between hidden layers, none after the final layer. "
           "input.size() must equal this branch's declared input width - "
           "a caller bug otherwise, not checked here.");

  py::class_<be::PolicyWeights>(m, "PolicyWeights")
      .def_readonly("actor", &be::PolicyWeights::actor)
      .def_readonly("critic", &be::PolicyWeights::critic)
      .def_static("load", &be::PolicyWeights::load, py::arg("path"),
                  "Loads both branches from one ppo.bin-format file (see "
                  "mlp.hpp's PolicyWeights::load doc comment for the exact "
                  "byte layout and validation performed). Raises "
                  "RuntimeError (via pybind11's automatic std::runtime_error "
                  "translation) on a missing/truncated/malformed file.");
}
