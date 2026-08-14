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
      .def_readwrite("sticky_web", &be::SideConditions::sticky_web);

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
      .def_readwrite("moves", &be::PokemonSlot::moves);

  py::class_<be::BattleState>(m, "BattleState")
      .def(py::init<>())
      // Whole-container access only - see this file's own module comment.
      .def_readwrite("my_team", &be::BattleState::my_team)
      .def_readwrite("opp_team", &be::BattleState::opp_team)
      .def_readwrite("my_active_slot", &be::BattleState::my_active_slot)
      .def_readwrite("opp_active_slot", &be::BattleState::opp_active_slot)
      .def_readwrite("my_hazards", &be::BattleState::my_hazards)
      .def_readwrite("opp_hazards", &be::BattleState::opp_hazards);

  m.def("is_valid", &be::is_valid, py::arg("state"));

  m.attr("NUM_SWITCH_ACTIONS") = int(be::kNumSwitchActions);
  m.attr("MOVE_ACTION_OFFSET") = int(be::kMoveActionOffset);
  m.attr("NUM_MOVE_ACTIONS") = int(be::kNumMoveActions);

  m.def("legal_actions", &be::legal_actions, py::arg("state"), py::arg("side"),
        "Legal ActionIds (see action.hpp) for `side` in `state`, using the "
        "fixed M5 action scheme - ActionId 0-5 switch, 6-9 move slot.");
}
