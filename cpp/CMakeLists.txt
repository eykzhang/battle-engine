cmake_minimum_required(VERSION 3.20)
project(battle_engine_native CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Debug preset with sanitizers on: for hand-written tree/pointer code
# (unique_ptr, raw indices into BattleState arrays) this catches
# use-after-free/out-of-bounds immediately instead of as a confusing crash
# weeks later. Enabled whenever CMAKE_BUILD_TYPE=Debug (scripts/build_cpp.sh
# passes this by default); Release builds skip it for real speed.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)

# Pinned to real tags, verified via the GitHub releases API at the time this
# was written (2026-08-12) - not copied from a tutorial. pybind11 v3.x is a
# major-version jump from the v2.x line most pybind11 tutorials reference.
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.15.3
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")

# be_core grows one target_sources() call at a time as each milestone lands
# (M4: battle_state, eval, pokedex_table; M5: forward_model, action,
# movedex_table; M6: mcts; M2/M3: nnue) - deliberately not all pre-declared
# in one shot, since CMake errors on a source file that doesn't exist yet.
add_library(be_core STATIC
  src/battle_state.cpp
  src/eval.cpp
  src/pokedex_table.cpp
  src/action.cpp
  src/forward_model.cpp
  src/movedex_table.cpp
)
target_include_directories(be_core PUBLIC include)

# pybind11_add_module's output directory is pointed straight at
# battle_engine/ (the Python package) - the compiled extension landing there
# just works with no reinstall step, since pyproject.toml has no
# [build-system] table (a plain PEP 660 editable install resolving into the
# source tree - confirmed before this was written, not assumed).
pybind11_add_module(_native bindings/module.cpp)
target_link_libraries(_native PRIVATE be_core)
set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../battle_engine"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../battle_engine"
)

enable_testing()
add_subdirectory(tests)
