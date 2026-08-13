// M1: toolchain bring-up only. ping() proves CMake + C++20 + pybind11 v3 +
// the editable-install import path work end-to-end before any real engine
// logic exists (see plans/precious-crafting-bachman.md, M1). Real bindings
// (BattleState, search(), ...) get added here starting M4/M6 - this file
// stays thin glue only, per the plan's own module-boundary intent.
#include <pybind11/pybind11.h>

// namespace py = pybind11; add back once real bindings (BattleState, etc.)
// need py:: types beyond what PYBIND11_MODULE's macro-generated `m` covers.

PYBIND11_MODULE(_native, m) {
  m.doc() = "battle_engine native C++ extension (Phase 4)";
  m.def("ping", []() { return 42; }, "M1 toolchain smoke test - returns 42.");
}
