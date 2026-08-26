add_executable(be_tests
  test_battle_state.cpp
  test_eval.cpp
  test_action.cpp
  test_forward_model.cpp
  test_mcts.cpp
  test_mlp.cpp
)
target_link_libraries(be_tests PRIVATE be_core Catch2::Catch2WithMain)

# M3: test_mlp.cpp's DW-3.2/DW-3.3-adjacent Catch2 tests load the REAL
# data/cpp_weights/ppo.bin (Phase 2's output) for the microbenchmark and as
# a positive-path load() smoke check - a compile-time absolute path avoids
# any dependence on ctest's own working directory (which isn't
# cpp/tests/, and isn't guaranteed stable across CMake versions/generators).
target_compile_definitions(be_tests PRIVATE
  BE_TEST_PPO_WEIGHTS_PATH="${CMAKE_SOURCE_DIR}/../data/cpp_weights/ppo.bin")

include(CTest)
include(Catch)
catch_discover_tests(be_tests)
